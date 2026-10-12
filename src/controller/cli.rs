//! CLI: 引数を読んで、探索（[`crate::service::search`]）と保存（[`crate::repository::key_file`]）を呼ぶ

use crate::controller::format;
use crate::model::backend::{Compress, Mul, Simd};
use crate::model::key::check_comment;
use crate::model::pattern::{Find, Pattern};
use crate::model::search::{SearchConfig, Stats};
use crate::repository::key_file;
use crate::service::search;
use clap::Parser;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

/// 公開鍵に好きな文字列を刻む ed25519 SSH 鍵ジェネレーター
// ヘルプの文面（`<OUTPUT>` など）は clap がそのまま表示するので、rustdoc の HTML の検査は外す
#[allow(rustdoc::invalid_html_tags)]
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

    /// 圧縮の逆元を256鍵でまとめず、1鍵ずつ計算する（高速化前の圧縮。比較用）。
    /// --constant-time・--simd off・--no-memmem と全部付けると、ed25519-dalek で1鍵ずつ計算する高速化前と同じになる
    #[arg(long)]
    no_batch: bool,

    /// 使う SIMD の上限（比較用）。CPU が対応していなければ、対応している中で一番上のものに下がる。
    /// avx512-ifma・avx512 は8鍵ずつ、avx2 は4鍵ずつ同時に計算し、off は1鍵ずつ（高速化前の SHA-512 と倍算）
    #[arg(long, value_enum, value_name = "LEVEL", default_value_t = SimdArg::Avx512Ifma)]
    simd: SimdArg,

    /// 公開鍵の中を memmem（SIMD の部分文字列検索）で探さず、1文字ずつずらして比べる
    /// （高速化前の検索。比較用）
    #[arg(long)]
    no_memmem: bool,
}

/// `--simd` の値
#[derive(Clone, Copy, clap::ValueEnum)]
enum SimdArg {
    #[value(name = "avx512-ifma")]
    Avx512Ifma,
    Avx512,
    Avx2,
    Off,
}

impl From<SimdArg> for Simd {
    fn from(s: SimdArg) -> Simd {
        match s {
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
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("エラー: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = SearchConfig::new(None);
    if let Some(n) = cli.threads {
        config.threads = n.into();
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
        check_features(&config);
        return Ok(());
    }

    let target = cli.target.as_ref().unwrap();
    let pattern = Pattern::new(target, cli.case_sensitive)?;
    check_comment(&cli.comment)?;
    config.pattern = Some(pattern.clone());

    check_features(&config);

    match cli.bench {
        Some(limit) => bench(config, &pattern, limit),
        None => generate(cli, config, &pattern),
    }
}
/// 0 より大きい秒数（小数可）
fn parse_secs(s: &str) -> Result<Duration, String> {
    let secs: f64 = s
        .parse()
        .map_err(|_| format!("{s:?} は秒数ではありません"))?;
    Duration::try_from_secs_f64(secs)
        .ok()
        .filter(|d| !d.is_zero())
        .ok_or_else(|| format!("{s:?} は 0 より大きい秒数にしてください"))
}

fn bench(
    mut config: SearchConfig,
    pattern: &Pattern,
    limit: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    // 文字列の判定まで含めて測る（当たっても止まらない）
    config.bench = true;
    config.time_limit = Some(limit);
    config.progress_interval = Duration::from_secs(1);

    eprintln!(
        "{}スレッド・{}・{} で{}秒間計測します",
        config.threads,
        config.backend,
        config.find,
        limit.as_secs_f64()
    );
    let mut last = Stats {
        tries: 0,
        elapsed: Duration::ZERO,
    };
    let out = search::run(&config, |now| {
        let delta = Stats {
            tries: now.tries - last.tries,
            elapsed: now.elapsed - last.elapsed,
        };
        eprintln!(
            "{:>5.1}s  {} 鍵/秒",
            now.elapsed.as_secs_f64(),
            format::count(delta.rate())
        );
        last = now;
    });

    let rate = out.stats.rate();
    println!(
        "速度: {} 鍵/秒（{}スレッド）",
        format::count(rate),
        config.threads
    );
    println!(
        "「{}」の期待時間: 約 {}",
        pattern.as_str(),
        format::duration_secs(pattern.expected_tries() / rate)
    );
    Ok(())
}

fn generate(
    cli: &Cli,
    config: SearchConfig,
    pattern: &Pattern,
) -> Result<(), Box<dyn std::error::Error>> {
    if !cli.force
        && let Some(p) = key_file::existing(&cli.output)?
    {
        // 何時間も回したあとで書き込めない、を防ぐため先に確認する
        // （書き込み時も確かめ、間に作られたファイルは上書きしない）
        return Err(format!("{} は既に存在します（上書きするなら --force）", p.display()).into());
    }

    let expected = pattern.expected_tries();
    eprintln!(
        "「{}」（大小文字{}）を{}スレッドで探します。期待試行回数: 約{}",
        pattern.as_str(),
        if pattern.case_sensitive() {
            "区別あり"
        } else {
            "区別なし"
        },
        config.threads,
        format::count(expected),
    );

    let out = search::run(&config, |s| {
        let remaining = (expected - s.tries as f64).max(0.0) / s.rate();
        eprint!(
            "\r{}  試行 {}  {} 鍵/秒  期待値まで残り約 {}    ",
            format::duration(s.elapsed),
            format::count(s.tries as f64),
            format::count(s.rate()),
            format::duration_secs(remaining),
        );
        let _ = io::stderr().flush();
    });
    eprintln!();

    let key = out.key.ok_or("鍵が見つかりませんでした")?;
    key_file::save(&key, &cli.comment, &cli.output, cli.force)?;

    eprintln!(
        "見つかりました！（{} 鍵、{}）",
        format::count(out.stats.tries as f64),
        format::duration(out.stats.elapsed)
    );
    let line = key.public_openssh(&cli.comment);
    println!("{line}");
    if let Some(r) = pattern.find(&key.public_b64()) {
        // "ssh-ed25519 " の分だけずらして、一致した場所に ^ を付ける
        let offset = "ssh-ed25519 ".len();
        eprintln!("{}{}", " ".repeat(offset + r.start), "^".repeat(r.len()));
    }
    eprintln!("秘密鍵: {}", cli.output.display());
    eprintln!("公開鍵: {}", key_file::pub_path(&cli.output).display());
    eprintln!(
        "※ パスフレーズなしです。使う前に ssh-keygen -p -f {} で設定してください",
        cli.output.display()
    );
    Ok(())
}

fn check_features(config: &SearchConfig) {
    eprintln!("高速化機能の対応状況:");
    let used = config.backend.simd.effective();
    for simd in Simd::ALL {
        let what = match simd {
            Simd::Avx512Ifma => "8鍵同時、52 ビットの積和で倍算",
            Simd::Avx512 => "8鍵同時",
            Simd::Avx2 => "4鍵同時",
            Simd::Off => continue,
        };
        let state = match (simd == used, simd.supported()) {
            (true, _) => "対応（使います）",
            (false, true) => "対応",
            (false, false) => "非対応",
        };
        eprintln!("- {}（{what}）: {state}", simd.name());
    }
    eprintln!("- 計算バックエンド: {}", config.backend);
    eprintln!("- 検索アルゴリズム: {}", config.find);
    eprintln!();
}
