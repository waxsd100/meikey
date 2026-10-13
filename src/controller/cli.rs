//! CLI: 引数を読んで、探索（[`Searcher`]）と保存（[`KeyRepository`]）を呼ぶ
//!
//! 探索と保存は trait 越しに受け取るので、テストでは mockall のモックに差し替えて、
//! 鍵を計算せず・ファイルに触れずに、呼ぶ順番・渡す設定・エラーの扱いを確かめられる

use crate::controller::format;
use crate::model::backend::{Compress, Mul, Simd};
use crate::model::key::check_comment;
use crate::model::pattern::{Find, Pattern};
use crate::model::search::{SearchConfig, Stats};
use crate::repository::key_file::{self, KeyFiles, KeyRepository};
use crate::service::search::{Parallel, Searcher};
use clap::Parser;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

/// 公開鍵に好きな文字列を刻む ed25519 SSH 鍵ジェネレーター
// 非公開の型なので、--document-private-items のときだけ検査される。expect だとそれ以外で未達になるので allow
#[allow(
    rustdoc::invalid_html_tags,
    reason = "ヘルプの文面（`<OUTPUT>` など）は clap がそのまま表示するので、HTML として書かない"
)]
#[derive(Parser)]
#[command(name = "meikey", version)]
struct Cli {
    /// 探索せずに、AVX2などの高速化機能が使えるか事前チェックして表示する
    #[arg(long)]
    check: bool,

    /// 公開鍵に入れたい文字列（A-Z a-z 0-9 + /、最大42文字）
    #[arg(required_unless_present = "check")]
    target: Option<String>,

    /// 大文字小文字を区別する（英字8文字なら256倍遅くなる）
    #[arg(short = 's', long)]
    case_sensitive: bool,

    /// スレッド数 [既定: 全コア]
    #[arg(short = 'j', long, value_parser = clap::value_parser!(u16).range(1..))]
    threads: Option<u16>,

    /// 秘密鍵の出力先（公開鍵は <OUTPUT>.pub）
    #[arg(short, long, default_value = "id_ed25519_meikey")]
    output: PathBuf,

    /// 鍵のコメント
    #[arg(short = 'C', long, default_value = "")]
    comment: String,

    /// 既存のファイルを上書きする
    #[arg(short, long)]
    force: bool,

    /// 探索せず、指定秒数だけ速度を測る（0.5 のような小数も可）
    #[arg(long, value_name = "SECS", value_parser = parse_secs, allow_negative_numbers = true)]
    bench: Option<Duration>,

    /// 基点倍算に定数時間の実装（curve25519-dalek）を使う（高速化前の倍算）。数倍遅いが、
    /// 同じマシン上の他プロセスからキャッシュのタイミングで秘密鍵を推測される余地をなくす
    #[arg(long)]
    constant_time: bool,

    /// 逆元を 256 鍵ごとに 1 回ではなく、1 鍵ごとに計算する（高速化前の圧縮。比較用）。
    /// --constant-time・--simd off・--no-memmem と全部付けると、ed25519-dalek で1鍵ずつ計算する高速化前と同じになる
    #[arg(long)]
    no_batch: bool,

    /// 使う SIMD の上限（比較用）。CPU が対応していなければ、対応している中で一番上のものに下がる。
    /// avx512-ifma・avx512 は8鍵ずつ、avx2 は4鍵ずつ同時に計算し、off は1鍵ずつ（高速化前の SHA-512 と倍算）。
    /// amx は体の乗算を AMX のタイル演算でする実験的な経路（IFMA より遅い。Linux のみ。feature amx を
    /// 有効にしてビルドしたときだけ指定できる）
    #[arg(long, value_enum, value_name = "LEVEL", default_value_t = SimdArg::Avx512Ifma)]
    simd: SimdArg,

    /// 公開鍵の中を memmem（SIMD の部分文字列検索）で探さず、1文字ずつずらして比べる
    /// （高速化前の検索。比較用）
    #[arg(long)]
    no_memmem: bool,

    /// CPU ではなく GPU（CUDA）で探す。--threads・--constant-time・--no-batch・--simd・--no-memmem は
    /// CPU 用なので使わない
    #[cfg(feature = "cuda")]
    #[arg(long, conflicts_with = "both")]
    gpu: bool,

    /// GPU と CPU の両方で同時に探す（CPU のスレッド数の既定は 全コア − 1。1 本は GPU の操作に使う）
    #[cfg(feature = "cuda")]
    #[arg(long)]
    both: bool,

    /// GPU の表の幅 W（2〜16）。大きいほど点の足し算が減り、表は大きくなる（12 で約 5.8 MB、16 で約 67 MB）。
    /// 既定の 12 から変えると、起動時に NVRTC でカーネルをコンパイルする
    #[cfg(feature = "cuda")]
    #[arg(
        long,
        value_name = "W",
        default_value_t = crate::service::gpu::DEFAULT_WIDTH,
        value_parser = clap::value_parser!(u32).range(2..=16)
    )]
    gpu_width: u32,

    /// GPU の1スレッドが、逆元を1回にまとめて計算する鍵の数（1〜64）。
    /// 既定の 16 から変えると、起動時に NVRTC でカーネルをコンパイルする
    #[cfg(feature = "cuda")]
    #[arg(
        long,
        value_name = "N",
        default_value_t = crate::service::gpu::DEFAULT_BATCH,
        value_parser = clap::value_parser!(u32).range(1..=64)
    )]
    gpu_batch: u32,
}

/// 探すのに使うもの（既定は CPU、--gpu で GPU、--both で両方）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu,
    #[cfg(feature = "cuda")]
    Both,
}

/// `--simd` の値
#[derive(Clone, Copy, clap::ValueEnum)]
enum SimdArg {
    #[cfg(feature = "amx")]
    Amx,
    #[value(name = "avx512-ifma")]
    Avx512Ifma,
    Avx512,
    Avx2,
    Off,
}

impl From<SimdArg> for Simd {
    fn from(s: SimdArg) -> Simd {
        match s {
            #[cfg(feature = "amx")]
            SimdArg::Amx => Simd::Amx,
            SimdArg::Avx512Ifma => Simd::Avx512Ifma,
            SimdArg::Avx512 => Simd::Avx512,
            SimdArg::Avx2 => Simd::Avx2,
            SimdArg::Off => Simd::Off,
        }
    }
}

/// 引数を読んで実行し、終了コードを返す
pub fn main() -> ExitCode {
    let cli = Cli::parse();
    match execute(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("エラー: {e}");
            ExitCode::FAILURE
        }
    }
}

/// 探すのに使うもの（CPU・GPU・両方）を決めて実行する
fn execute(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    match device(cli) {
        Device::Cpu => run(cli, &Parallel, &KeyFiles),
        #[cfg(feature = "cuda")]
        Device::Gpu => run(
            cli,
            &crate::service::gpu::Gpu::new(0, gpu_options(cli))?,
            &KeyFiles,
        ),
        #[cfg(feature = "cuda")]
        Device::Both => {
            let gpu = crate::service::gpu::Gpu::new(0, gpu_options(cli))?;
            run(cli, &crate::service::search::Both(gpu, Parallel), &KeyFiles)
        }
    }
}

#[cfg(feature = "cuda")]
fn gpu_options(cli: &Cli) -> crate::service::gpu::GpuOptions {
    crate::service::gpu::GpuOptions {
        width: cli.gpu_width,
        batch: cli.gpu_batch,
    }
}

fn device(cli: &Cli) -> Device {
    #[cfg(feature = "cuda")]
    if cli.both {
        return Device::Both;
    } else if cli.gpu {
        return Device::Gpu;
    }
    let _ = cli;
    Device::Cpu
}

fn run(
    cli: &Cli,
    searcher: &impl Searcher,
    repo: &impl KeyRepository,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = SearchConfig::new(None);
    if let Some(n) = cli.threads {
        config.threads = n.into();
    } else if device(cli) != Device::Cpu {
        // GPU と両方で探すときは、GPU を操作するスレッドの分を空ける
        // （GPU だけのときは CPU で探さないので、この値は使わない）
        config.threads = config.threads.saturating_sub(1).max(1);
    }
    if cli.constant_time {
        config.backend.mul = Mul::ConstantTime;
    }
    if cli.no_batch {
        config.backend.compress = Compress::Each;
    }
    config.backend.simd = cli.simd.into();
    if cli.no_memmem {
        config.find = Find::Naive;
    }

    if cli.check {
        print_rows(&searcher.status(&config));
        return Ok(());
    }

    let target = cli.target.as_ref().unwrap();
    let pattern = Pattern::new(target, cli.case_sensitive)?;
    check_comment(&cli.comment)?;
    config.pattern = Some(pattern.clone());

    let mut rows = searcher.status(&config);
    rows.extend([
        (
            "文字列",
            format!(
                "{}（大小文字の区別{}）",
                pattern.as_str(),
                if pattern.case_sensitive() {
                    "あり"
                } else {
                    "なし"
                }
            ),
        ),
        (
            "期待試行",
            format!("約 {}", format::count(pattern.expected_tries())),
        ),
    ]);
    print_rows(&rows);

    match cli.bench {
        Some(limit) => bench(searcher, config, &pattern, limit),
        None => generate(cli, searcher, repo, config, &pattern),
    }
}

/// 0 より大きい秒数（小数可）
fn parse_secs(s: &str) -> Result<Duration, String> {
    let secs: f64 = s.parse().map_err(|_| format!("秒数ではない: {s:?}"))?;
    Duration::try_from_secs_f64(secs)
        .ok()
        .filter(|d| !d.is_zero())
        .ok_or_else(|| format!("0 より大きい秒数が必要: {s:?}"))
}

fn bench(
    searcher: &impl Searcher,
    mut config: SearchConfig,
    pattern: &Pattern,
    limit: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    // 文字列の判定まで含めて測る（当たっても止まらない）
    config.bench = true;
    config.time_limit = Some(limit);
    config.progress_interval = Duration::from_secs(1);

    print_rows(&[("計測", format!("{} 秒", limit.as_secs_f64()))]);
    let mut last = Stats {
        tries: 0,
        elapsed: Duration::ZERO,
    };
    let out = searcher.run(&config, &mut |now: Stats| {
        let delta = since(now, last);
        eprintln!(
            "{:>5.1}s  {} 鍵/秒",
            now.elapsed.as_secs_f64(),
            format::count(delta.rate())
        );
        last = now;
    });

    let rate = out.stats.rate();
    println!(
        "{}",
        format::row("速度", &format!("{} 鍵/秒", format::count(rate)))
    );
    println!(
        "{}",
        format::row(
            "期待時間",
            &format!(
                "約 {}",
                format::duration_secs(pattern.expected_tries() / rate)
            )
        )
    );
    Ok(())
}

fn generate(
    cli: &Cli,
    searcher: &impl Searcher,
    repo: &impl KeyRepository,
    config: SearchConfig,
    pattern: &Pattern,
) -> Result<(), Box<dyn std::error::Error>> {
    if !cli.force
        && let Some(p) = repo.existing(&cli.output)?
    {
        // 何時間も回したあとで書き込めない、を防ぐため先に確認する
        // （書き込み時も確かめ、間に作られたファイルは上書きしない）
        return Err(format!("既に存在: {}（--force で上書き）", p.display()).into());
    }

    let expected = pattern.expected_tries();

    let mut shown = false;
    let out = searcher.run(&config, &mut |s: Stats| {
        shown = true;
        let remaining = remaining_secs(expected, s);
        eprint!(
            "\r経過 {}  試行 {}  {} 鍵/秒  残り約 {}    ",
            format::duration(s.elapsed),
            format::count(s.tries as f64),
            format::count(s.rate()),
            format::duration_secs(remaining),
        );
        let _ = io::stderr().flush();
    });
    if shown {
        // 進捗の行を改行で終える
        eprintln!();
    }

    let key = out.key.ok_or("鍵が見つからなかった")?;
    repo.save(&key, &cli.comment, &cli.output, cli.force)?;

    print_rows(&[
        (
            "試行",
            format!("{} 鍵", format::count(out.stats.tries as f64)),
        ),
        ("経過", format::duration(out.stats.elapsed)),
    ]);
    let line = key.public_openssh(&cli.comment);
    println!("{line}");
    if let Some(r) = pattern.find(&key.public_b64()) {
        // "ssh-ed25519 " の分だけずらして、一致した場所に ^ を付ける
        let offset = "ssh-ed25519 ".len();
        eprintln!("{}{}", " ".repeat(offset + r.start), "^".repeat(r.len()));
    }
    print_rows(&[
        ("秘密鍵", cli.output.display().to_string()),
        (
            "公開鍵",
            key_file::pub_path(&cli.output).display().to_string(),
        ),
        (
            "パスフレーズ",
            format!("なし（設定: ssh-keygen -p -f {}）", cli.output.display()),
        ),
    ]);
    Ok(())
}

/// 前回の進捗 `last` から `now` までの増分（計測中、1秒ごとの速度に使う）
fn since(now: Stats, last: Stats) -> Stats {
    Stats {
        tries: now.tries - last.tries,
        elapsed: now.elapsed - last.elapsed,
    }
}

/// 今の速さのまま、期待試行回数 `expected` に届くまでの残り秒数（もう超えていれば 0）
fn remaining_secs(expected: f64, s: Stats) -> f64 {
    (expected - s.tries as f64).max(0.0) / s.rate()
}

/// 状態の行を標準エラーに（`ラベル  値`）
fn print_rows(rows: &[(&str, String)]) {
    for (label, value) in rows {
        eprintln!("{}", format::row(label, value));
    }
}

#[cfg(test)]
mod tests {
    //! 探索（[`MockSearcher`]）と保存（[`MockKeyRepository`]）を mockall のモックに差し替えて、
    //! controller の配線だけを確かめる。期待していない呼び出しが来ると、モックがその場でパニックする
    use super::*;
    use crate::model::backend::Backend;
    use crate::model::key::VanityKey;
    use crate::model::search::Outcome;
    use crate::repository::key_file::MockKeyRepository;
    use crate::service::search::MockSearcher;
    use mockall::Sequence;
    use mockall::predicate::eq;
    use rstest::{fixture, rstest};
    use similar_asserts::assert_eq;
    use std::iter::once;
    use std::path::Path;

    /// 表示用の名前だけ答える探索のモック（run の期待は各テストで書く）
    #[fixture]
    fn searcher() -> MockSearcher {
        let mut s = MockSearcher::new();
        s.expect_status()
            .return_const(vec![("テスト", "探索".to_string())]);
        s
    }

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(once("meikey").chain(args.iter().copied())).unwrap()
    }

    /// 保存のモック（期待は各テストで書く）
    #[fixture]
    fn repo() -> MockKeyRepository {
        MockKeyRepository::new()
    }

    /// 見つかったことにする鍵と、その公開鍵に含まれる文字列
    #[fixture]
    fn found() -> (VanityKey, String) {
        let key = VanityKey::from_seed([7; 32]);
        let target = String::from_utf8(key.public_b64()[10..14].to_vec()).unwrap();
        (key, target)
    }

    /// 試行回数と経過時間
    #[fixture]
    fn stats(#[default(0)] tries: u64, #[default(0)] millis: u64) -> Stats {
        Stats {
            tries,
            elapsed: Duration::from_millis(millis),
        }
    }

    /// 計測中の1秒ごとの速度は、前回からの増分で出す
    #[rstest]
    #[case::first(stats(1_000, 1_000), stats(0, 0), stats(1_000, 1_000))]
    #[case::later(stats(5_000, 3_000), stats(2_000, 2_000), stats(3_000, 1_000))]
    fn since_is_delta(#[case] now: Stats, #[case] last: Stats, #[case] want: Stats) {
        assert_eq!(since(now, last), want);
    }

    /// 残り時間 = 期待試行回数までの残り ÷ 今の速さ。超えていれば 0
    #[rstest]
    #[case::half_way(1_000.0, stats(500, 1_000), 1.0)]
    #[case::just_started(1_000.0, stats(100, 2_000), 18.0)]
    #[case::reached(1_000.0, stats(1_000, 1_000), 0.0)]
    #[case::beyond(1_000.0, stats(3_000, 1_000), 0.0)]
    fn remaining_secs_at_current_rate(
        #[case] expected: f64,
        #[case] now: Stats,
        #[case] want: f64,
    ) {
        let got = remaining_secs(expected, now);
        assert_eq!((got - want).abs() < 1e-9, true, "{got}");
    }

    /// 既存ファイルの確認（--force なら省く）→ 探索 → 保存 の順に呼び、見つかった鍵を
    /// 指定どおりのパス・コメント・上書き指定で保存する
    #[rstest]
    fn generate_saves_found_key(
        #[values(false, true)] force: bool,
        mut searcher: MockSearcher,
        found: (VanityKey, String),
        mut repo: MockKeyRepository,
    ) {
        let (key, target) = found;
        let mut args = vec![target.as_str(), "-o", "out/id_test", "-C", "コメント"];
        if force {
            args.push("--force");
        }
        let cli = parse(&args);
        let mut seq = Sequence::new();

        repo.expect_existing()
            .with(eq(Path::new("out/id_test")))
            .times(usize::from(!force))
            .in_sequence(&mut seq)
            .returning(|_| Ok(None));
        let (want, t) = (key.clone(), target.clone());
        searcher
            .expect_run()
            .withf(move |c, _| {
                c.pattern.as_ref().is_some_and(|p| p.as_str() == t)
                    && !c.bench
                    && c.time_limit.is_none()
            })
            .times(1)
            .in_sequence(&mut seq)
            .returning(move |_, on_progress| {
                // 進捗表示も通す（時間・試行回数が増えていく普通の呼ばれ方）
                on_progress.report(stats(1_000, 100));
                on_progress.report(stats(5_000, 200));
                Outcome {
                    key: Some(want.clone()),
                    stats: stats(5_100, 210),
                }
            });
        repo.expect_save()
            .withf(move |k, comment, private, overwrite| {
                *k == key
                    && comment == "コメント"
                    && private == Path::new("out/id_test")
                    && *overwrite == force
            })
            .times(1)
            .in_sequence(&mut seq)
            .returning(|_, _, _, _| Ok(()));

        run(&cli, &searcher, &repo).unwrap();
    }

    /// 出力先が既にあれば（--force なし）、何時間も探す前に止まり、探索も保存もしない
    #[rstest]
    fn existing_output_stops_before_search(
        #[values("id_test", "id_test.pub")] already: &'static str,
        mut searcher: MockSearcher,
        mut repo: MockKeyRepository,
    ) {
        let cli = parse(&["abc", "-o", "id_test"]);

        repo.expect_existing()
            .times(1)
            .returning(move |_| Ok(Some(PathBuf::from(already))));
        repo.expect_save().never();

        searcher.expect_run().never();

        let e = run(&cli, &searcher, &repo).unwrap_err().to_string();
        assert_eq!(e.contains(already) && e.contains("--force"), true, "{e}");
    }

    /// 既存ファイルを確かめられなければ（権限など）、そのエラーで止まり、探索しない
    #[rstest]
    fn existing_check_error_stops_before_search(
        mut searcher: MockSearcher,
        mut repo: MockKeyRepository,
    ) {
        let cli = parse(&["abc"]);

        repo.expect_existing()
            .returning(|_| Err(io::Error::new(io::ErrorKind::PermissionDenied, "読めない")));

        searcher.expect_run().never();

        let e = run(&cli, &searcher, &repo).unwrap_err().to_string();
        assert_eq!(e, "読めない");
    }

    /// 見つからずに終わった（時間切れ・パニック後の停止など）なら、エラーにして何も保存しない
    #[rstest]
    fn not_found_saves_nothing(mut searcher: MockSearcher, mut repo: MockKeyRepository) {
        let cli = parse(&["abc"]);

        repo.expect_existing().returning(|_| Ok(None));
        repo.expect_save().never();

        searcher.expect_run().returning(|_, _| Outcome {
            key: None,
            stats: stats(10, 5),
        });

        let e = run(&cli, &searcher, &repo).unwrap_err().to_string();
        assert_eq!(e, "鍵が見つからなかった");
    }

    /// 保存に失敗したら（探索中に作られたファイル・書き込み権限など）、そのエラーを返す
    #[rstest]
    #[case::created_meanwhile(io::ErrorKind::AlreadyExists)]
    #[case::permission(io::ErrorKind::PermissionDenied)]
    fn save_error_is_returned(
        #[case] kind: io::ErrorKind,
        mut searcher: MockSearcher,
        found: (VanityKey, String),
        mut repo: MockKeyRepository,
    ) {
        let (key, target) = found;
        let cli = parse(&[&target]);

        repo.expect_existing().returning(|_| Ok(None));
        repo.expect_save()
            .times(1)
            .returning(move |_, _, _, _| Err(io::Error::new(kind, "書けない")));

        searcher.expect_run().returning(move |_, _| Outcome {
            key: Some(key.clone()),
            stats: stats(10, 5),
        });

        let e = run(&cli, &searcher, &repo).unwrap_err();
        let e = e.downcast_ref::<io::Error>().expect("io::Error のまま返す");
        assert_eq!(e.kind(), kind);
    }

    /// --bench は「当たっても止まらない・時間で止まる・1秒ごとに進捗」の設定で探索し、
    /// ファイルには一切触れない（`repo` に期待を書いていないので、呼べばパニックする）
    #[rstest]
    #[case::half_second("0.5", 500)]
    #[case::two_seconds("2", 2_000)]
    fn bench_measures_without_files(
        #[case] secs: &str,
        #[case] millis: u64,
        mut searcher: MockSearcher,
        repo: MockKeyRepository,
    ) {
        let cli = parse(&["--bench", secs, "abc", "--force"]);

        searcher
            .expect_run()
            .withf(move |c, _| {
                c.bench
                    && c.time_limit == Some(Duration::from_millis(millis))
                    && c.progress_interval == Duration::from_secs(1)
                    && c.pattern.is_some()
            })
            .times(1)
            .returning(move |_, on_progress| {
                on_progress.report(stats(1_000, 1_000));
                on_progress.report(stats(2_500, 2_000));
                Outcome {
                    key: None,
                    stats: stats(3_000, millis),
                }
            });

        run(&cli, &searcher, &repo).unwrap();
    }

    /// --check は対応状況を表示するだけで、探索も保存もしない（文字列も要らない）
    #[rstest]
    #[case::alone(&["--check"])]
    #[case::with_target(&["--check", "abc", "--bench", "1"])]
    fn check_only_reports(#[case] args: &[&str], searcher: MockSearcher, repo: MockKeyRepository) {
        run(&parse(args), &searcher, &repo).unwrap();
    }

    /// 文字列・コメントがおかしければ、ファイルを確かめる前・探索する前に止まる
    #[rstest]
    #[case::invalid_char(&["ab-c"], "使えない文字: '-'")]
    #[case::too_long(&["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"], "長すぎる")]
    #[case::newline_in_comment(&["abc", "-C", "a\nb"], "")]
    #[case::nul_in_comment(&["abc", "-C", "a\0b", "--bench", "1"], "")]
    fn invalid_input_stops_early(
        #[case] args: &[&str],
        #[case] message: &str,
        searcher: MockSearcher,
        repo: MockKeyRepository,
    ) {
        let e = run(&parse(args), &searcher, &repo).unwrap_err().to_string();
        assert_eq!(e.contains(message), true, "{e}");
    }

    /// --simd amx は feature amx が有効なときだけ受け付け、探索の設定に届く
    #[rstest]
    fn simd_amx_needs_feature() {
        let parsed = Cli::try_parse_from(["meikey", "abc", "--simd", "amx"]);
        #[cfg(feature = "amx")]
        assert_eq!(Simd::from(parsed.unwrap().simd), Simd::Amx);
        #[cfg(not(feature = "amx"))]
        assert_eq!(parsed.is_err(), true);
    }

    /// 既定は CPU、--gpu で GPU、--both で両方。--gpu と --both は一緒に付けられない
    #[cfg(feature = "cuda")]
    #[rstest]
    #[case::default(&[], Some(Device::Cpu))]
    #[case::gpu(&["--gpu"], Some(Device::Gpu))]
    #[case::both(&["--both"], Some(Device::Both))]
    #[case::conflict(&["--gpu", "--both"], None)]
    fn device_from_flags(#[case] options: &[&str], #[case] want: Option<Device>) {
        let parsed = Cli::try_parse_from(["meikey", "abc"].iter().chain(options));
        assert_eq!(parsed.ok().map(|c| device(&c)), want);
    }

    /// --gpu-width・--gpu-batch は範囲内の値だけ受け付け、GPU の設定に届く（無ければ既定）
    #[cfg(feature = "cuda")]
    #[rstest]
    #[case::defaults(&[], Some((12, 16)))]
    #[case::wide(&["--gpu-width", "16", "--gpu-batch", "32"], Some((16, 32)))]
    #[case::smallest(&["--gpu-width", "2", "--gpu-batch", "1"], Some((2, 1)))]
    #[case::largest_batch(&["--gpu-batch", "64"], Some((12, 64)))]
    #[case::width_too_large(&["--gpu-width", "17"], None)]
    #[case::width_too_small(&["--gpu-width", "1"], None)]
    #[case::batch_zero(&["--gpu-batch", "0"], None)]
    #[case::batch_too_large(&["--gpu-batch", "65"], None)]
    fn gpu_options_from_flags(#[case] options: &[&str], #[case] want: Option<(u32, u32)>) {
        let parsed = Cli::try_parse_from(["meikey", "abc", "--gpu"].iter().chain(options));
        assert_eq!(
            parsed
                .ok()
                .map(|c| gpu_options(&c))
                .map(|o| (o.width, o.batch)),
            want
        );
    }

    /// --gpu・--both は feature cuda が無ければ受け付けない
    #[cfg(not(feature = "cuda"))]
    #[rstest]
    fn gpu_flags_need_feature(
        #[values(&["--gpu"], &["--both"], &["--gpu-width", "16"], &["--gpu-batch", "32"])]
        flags: &[&str],
    ) {
        let args = ["meikey", "abc"].iter().chain(flags);
        assert_eq!(Cli::try_parse_from(args).is_err(), true);
    }

    /// CPU のスレッド数: -j が無ければ、CPU だけなら全コア、GPU と両方なら 全コア − 1（最低 1）。-j があればそのまま
    #[rstest]
    #[case::cpu(&[], |all| all)]
    #[case::cpu_explicit(&["-j", "3"], |_| 3)]
    #[cfg_attr(feature = "cuda", case::both(&["--both"], |all: usize| all.saturating_sub(1).max(1)))]
    #[cfg_attr(feature = "cuda", case::both_explicit(&["--both", "-j", "3"], |_| 3))]
    fn cpu_threads(
        #[case] options: &[&str],
        #[case] want: fn(usize) -> usize,
        mut searcher: MockSearcher,
        repo: MockKeyRepository,
    ) {
        let want = want(SearchConfig::new(None).threads);
        let mut args = vec!["abc", "--bench", "1"];
        args.extend(options);

        searcher
            .expect_run()
            .withf(move |c, _| c.threads == want)
            .times(1)
            .returning(|_, _| Outcome {
                key: None,
                stats: stats(10, 5),
            });

        run(&parse(&args), &searcher, &repo).unwrap();
    }

    /// 各オプションが探索の設定（[`SearchConfig`]）にそのまま届く
    #[rstest]
    #[case::defaults(&[], |c: &SearchConfig| {
        c.backend == Backend::FAST
            && c.find == Find::Memmem
            && !c.pattern.as_ref().unwrap().case_sensitive()
    })]
    #[case::threads(&["-j", "3"], |c: &SearchConfig| c.threads == 3)]
    #[case::case_sensitive(&["-s"], |c: &SearchConfig| c.pattern.as_ref().unwrap().case_sensitive())]
    #[case::constant_time(&["--constant-time"], |c: &SearchConfig| c.backend.mul == Mul::ConstantTime)]
    #[case::no_batch(&["--no-batch"], |c: &SearchConfig| c.backend.compress == Compress::Each)]
    #[case::simd_avx512(&["--simd", "avx512"], |c: &SearchConfig| c.backend.simd == Simd::Avx512)]
    #[case::simd_avx2(&["--simd", "avx2"], |c: &SearchConfig| c.backend.simd == Simd::Avx2)]
    #[case::simd_off(&["--simd", "off"], |c: &SearchConfig| c.backend.simd == Simd::Off)]
    #[case::no_memmem(&["--no-memmem"], |c: &SearchConfig| c.find == Find::Naive)]
    #[case::baseline(
        &["--constant-time", "--no-batch", "--simd", "off", "--no-memmem"],
        |c: &SearchConfig| c.backend == Backend::BASELINE && c.find == Find::Naive
    )]
    fn options_reach_search_config(
        #[case] options: &[&str],
        #[case] want: fn(&SearchConfig) -> bool,
        #[values(false, true)] bench: bool,
        mut searcher: MockSearcher,
        mut repo: MockKeyRepository,
    ) {
        let mut args = vec!["abc"];
        args.extend(options);
        if bench {
            args.extend(["--bench", "1"]);
        }

        repo.expect_existing().returning(|_| Ok(None));

        searcher
            .expect_run()
            .withf(move |c, _| want(c))
            .times(1)
            .returning(|_, _| Outcome {
                key: None,
                stats: stats(10, 5),
            });

        let result = run(&parse(&args), &searcher, &repo);
        // 見つからない結果を返しているので、生成ならエラー・計測なら成功
        assert_eq!(result.is_ok(), bench);
    }
}
