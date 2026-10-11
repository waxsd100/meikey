use clap::Parser;
use meikey::{Compress, Mul, Pattern, SearchConfig, Stats, VanityKey, format, search};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

/// 公開鍵に好きな文字列を刻む ed25519 SSH 鍵ジェネレーター
#[derive(Parser)]
#[command(name = "meikey", version)]
struct Cli {
    /// 公開鍵に入れたい文字列（A-Z a-z 0-9 + /、最大42文字）
    target: String,

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

    /// 基点倍算に定数時間の実装（curve25519-dalek）を使う（高速化前の倍算）。約3倍遅いが、
    /// 同じマシン上の他プロセスからキャッシュのタイミングで秘密鍵を推測される余地をなくす
    #[arg(long)]
    constant_time: bool,

    /// 圧縮の逆元を256鍵でまとめず、1鍵ずつ計算する（高速化前の圧縮。比較用）。
    /// --constant-time と両方付けると、ed25519-dalek で1鍵ずつ計算する高速化前と同じになる
    #[arg(long)]
    no_batch: bool,
}

fn main() -> ExitCode {
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
    let pattern = Pattern::new(&cli.target, cli.case_sensitive)?;
    meikey::key::check_comment(&cli.comment)?;
    let mut config = SearchConfig::new(Some(pattern.clone()));
    if let Some(n) = cli.threads {
        config.threads = n.into();
    }
    if cli.constant_time {
        config.backend.mul = Mul::ConstantTime;
    }
    if cli.no_batch {
        config.backend.compress = Compress::Each;
    }

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
    config.pattern = None;
    config.time_limit = Some(limit);
    config.progress_interval = Duration::from_secs(1);

    eprintln!(
        "{}スレッド・{} で{}秒間計測します",
        config.threads,
        config.backend,
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
    let pub_path = cli.output.with_added_extension("pub");
    if !cli.force {
        // 何時間も回したあとで書き込めない、を防ぐため先に確認する
        // （書き込み時も create_new で、間に作られたファイルは上書きしない）
        for p in [&cli.output, &pub_path] {
            (!p.try_exists()?).ok_or_else(|| {
                format!("{} は既に存在します（上書きするなら --force）", p.display())
            })?;
        }
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
    save(&key, cli, &pub_path)?;

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
    eprintln!("公開鍵: {}", pub_path.display());
    eprintln!(
        "※ パスフレーズなしです。使う前に ssh-keygen -p -f {} で設定してください",
        cli.output.display()
    );
    Ok(())
}

fn save(key: &VanityKey, cli: &Cli, pub_path: &Path) -> io::Result<()> {
    let private = key.private_openssh(&cli.comment, rand::random());
    write_file(&cli.output, &private, 0o600, cli.force)?;
    write_file(
        pub_path,
        &(key.public_openssh(&cli.comment) + "\n"),
        0o644,
        cli.force,
    )
}

/// ファイルを新しく作って書く。`overwrite` なら既存のファイルを先に消し、そうでなければ
/// 既存のファイルがあると失敗する（どちらも create_new）。
/// `mode` は Unix のパーミッション（秘密鍵は 0600 で所有者だけ読める）。既存のファイルに上書きすると
/// そのファイルのパーミッション（0644 など）が残ってしまうので、必ず新しいファイルとして作る
fn write_file(path: &Path, content: &str, mode: u32, overwrite: bool) -> io::Result<()> {
    if overwrite
        && let Err(e) = std::fs::remove_file(path)
        && e.kind() != io::ErrorKind::NotFound
    {
        return Err(e);
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, mode);
    #[cfg(not(unix))]
    let _ = mode;
    opts.open(path)?.write_all(content.as_bytes())
}
