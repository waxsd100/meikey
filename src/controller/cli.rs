//! CLI: 引数を読んで、探索（[`Searcher`]）と保存（[`KeyRepository`]）を呼ぶ
//!
//! 探索と保存は trait 越しに受け取るので、テストでは mockall のモックに差し替えて、
//! 鍵を計算せず・ファイルに触れずに、呼ぶ順番・渡す設定・エラーの扱いを確かめられる

use crate::controller::format;
use crate::lang::{Lang, current, t, tf};
use crate::model::backend::{Compress, Mul, Simd};
use crate::model::key::check_comment;
use crate::model::pattern::{Find, Mode, Pattern};
use crate::model::search::{SearchConfig, SeedSource, Stats};
use crate::repository::key_file::{KeyFiles, KeyRepository, pub_path};
use crate::service::search::{Parallel, Searcher};
use clap::Parser;
// --gpu-limbs の値（"10"・"8" など）を Limbs にする map のため
#[cfg(feature = "cuda")]
use clap::builder::TypedValueParser as _;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// 公開鍵に好きな文字列を刻む ed25519 SSH 鍵ジェネレーター。
///
/// ヘルプの文は言語ごとに [`HELP`] にあり、[`command`] で付ける（ここのフィールドには書かない）
#[derive(Parser)]
#[command(name = "meikey", version)]
// --gpu-width・--gpu-batch・--gpu-limbs は GPU で探すとき（--gpu か --both）だけ付けられる
#[cfg_attr(
    feature = "cuda",
    command(group(clap::ArgGroup::new("gpu_device").args(["gpu", "both"])))
)]
struct Cli {
    #[arg(long)]
    check: bool,

    #[arg(required_unless_present = "check")]
    target: Option<String>,

    #[arg(short = 's', long)]
    case_sensitive: bool,

    #[arg(long, conflicts_with = "regex")]
    suffix: bool,

    #[arg(long)]
    regex: bool,

    #[arg(short = 'j', long, value_parser = |s: &str| int_in(s, 1, u16::MAX.into()).map(|n| n as u16))]
    threads: Option<u16>,

    #[arg(short, long, default_value = "id_ed25519_meikey")]
    output: PathBuf,

    #[arg(short = 'C', long, default_value = "", hide_default_value = true)]
    comment: String,

    #[arg(short, long)]
    force: bool,

    #[arg(long, value_name = "SECS", value_parser = parse_secs, allow_negative_numbers = true)]
    bench: Option<Duration>,

    #[arg(long)]
    low_security: bool,

    #[arg(long)]
    no_batch: bool,

    #[arg(long, value_enum, value_name = "LEVEL", default_value_t = SimdArg::Avx512Ifma)]
    simd: SimdArg,

    #[arg(long)]
    no_memmem: bool,

    // GPU の計算は定数時間でないので、低セキュリティモードのときだけ使える
    #[cfg(feature = "cuda")]
    #[arg(long, conflicts_with = "both", requires = "low_security")]
    gpu: bool,

    #[cfg(feature = "cuda")]
    #[arg(long, requires = "low_security")]
    both: bool,

    #[cfg(feature = "cuda")]
    #[arg(
        long,
        value_name = "W",
        value_parser = |s: &str| int_in(s, 2, 20),
        requires = "gpu_device"
    )]
    gpu_width: Option<u32>,

    #[cfg(feature = "cuda")]
    #[arg(
        long,
        value_name = "N",
        value_parser = |s: &str| int_in(s, 1, 64),
        requires = "gpu_device"
    )]
    gpu_batch: Option<u32>,

    #[cfg(feature = "cuda")]
    #[arg(
        long,
        value_name = "L",
        value_parser = clap::builder::PossibleValuesParser::new(["10", "10u", "8", "10k", "10ku", "5f"])
            .map(|s| s.parse::<crate::service::gpu::Limbs>().expect("--gpu-limbs の値")),
        requires = "gpu_device"
    )]
    gpu_limbs: Option<crate::service::gpu::Limbs>,
}

/// コマンドの説明（日本語・英語）
const ABOUT: (&str, &str) = (
    "公開鍵に好きな文字列を刻む ed25519 SSH 鍵ジェネレーター",
    "An ed25519 SSH key generator that puts the text you choose into the public key",
);

/// 引数ごとのヘルプの文（[`Cli`] のフィールド名、日本語、英語）。
/// 日本語のヘルプの見出しは、この順に付け直した引数の順に並ぶので、「引数」（target）を先に置く
const HELP: &[(&str, &str, &str)] = &[
    (
        "target",
        "公開鍵に入れたい文字列（A-Z a-z 0-9 + /、最大42文字。--regex なら正規表現）",
        "Text to put in the public key (A-Z a-z 0-9 + /, up to 42 characters; a regular expression with --regex)",
    ),
    (
        "check",
        "探索せずに、AVX2などの高速化機能が使えるか事前チェックして表示する",
        "Show which speedups (AVX2 and so on) can be used, without searching",
    ),
    (
        "case_sensitive",
        "大文字小文字を区別する（英字8文字なら256倍遅くなる）",
        "Match upper and lower case exactly (8 letters make it 256 times slower)",
    ),
    (
        "suffix",
        "公開鍵の最後が文字列で終わる鍵だけを探す（既定は、固定部分の後ろのどこかに含む鍵）",
        "Look only for keys whose public key ends with the text (by default, the text can be anywhere after the fixed part)",
    ),
    (
        "regex",
        "文字列を正規表現として、固定部分の後ろの 43 文字に一致する鍵を探す（^ は固定部分の直後、$ は最後。\
         -s を付けなければ大小文字を区別しない）",
        "Treat the text as a regular expression and look for keys whose 43 characters after the fixed part \
         match it (^ is right after the fixed part and $ is the end; case-insensitive unless -s is given)",
    ),
    (
        "threads",
        "スレッド数 [既定: 全コア]",
        "Number of threads [default: all cores]",
    ),
    (
        "output",
        "秘密鍵の出力先（公開鍵は <OUTPUT>.pub）",
        "Where to write the private key (the public key goes to <OUTPUT>.pub)",
    ),
    ("comment", "鍵のコメント", "Comment for the key"),
    (
        "force",
        "既存のファイルを上書きする",
        "Overwrite existing files",
    ),
    (
        "bench",
        "探索せず、指定秒数だけ速度を測る（0.5 のような小数も可）",
        "Measure the speed for this many seconds instead of searching (fractions such as 0.5 are allowed)",
    ),
    (
        "low_security",
        "低セキュリティモード: 速さを優先する。基点倍算を定数時間でない表の倍算にし（数倍速いが、同じマシン上の\
         他プロセスからキャッシュのタイミングで秘密鍵を推測される余地がある）、シードは鍵ごとの ChaCha20 ではなく、\
         OS の乱数（スレッドごとに1回）に通し番号を足したものにする。--gpu・--both はこのモードでだけ使える\
         （GPU の計算は定数時間でない）。付けなければ、倍算は定数時間の実装（AVX2・AVX-512 では表の行の\
         全エントリを読んで選ぶもの、SIMD を使わなければ curve25519-dalek）",
        "Low-security mode: prioritize speed. Use the table-based base-point multiplication, which is \
         not constant-time (several times faster, but leaves room for other processes on the same \
         machine to infer the private key from cache timing), and make seeds by adding a counter to \
         OS randomness (once per thread) instead of ChaCha20 for each key. --gpu and --both work only \
         in this mode (the GPU computation is not constant-time). Without it, the multiplication \
         is constant-time (with AVX2 or AVX-512, a table walk that reads every entry of each row; \
         without SIMD, curve25519-dalek)",
    ),
    (
        "no_batch",
        "逆元を 256 鍵ごとに 1 回ではなく、1 鍵ごとに計算する（高速化前の圧縮。比較用）。\
         --simd off・--no-memmem と全部付けると（--low-security なし）、ed25519-dalek で1鍵ずつ計算する高速化前と\
         同じになる",
        "Compute the inverse for every key instead of once per 256 keys (the compression before \
         the speedups; for comparison). With --simd off and --no-memmem as well (without \
         --low-security), keys are computed one at a time with ed25519-dalek, the same as before \
         the speedups",
    ),
    (
        "simd",
        "使う SIMD の上限（比較用）。CPU が対応していなければ、対応している中で一番上のものに下がる。\
         avx512-ifma・avx512 は8鍵ずつ、avx2 は4鍵ずつ同時に計算し、off は1鍵ずつ（高速化前の SHA-512 と倍算）。\
         amx は体の乗算を AMX のタイル演算でする実験的な経路（IFMA より遅い。Linux のみ。feature amx を\
         有効にしてビルドしたときだけ指定できる）",
        "Highest SIMD level to use (for comparison). If the CPU does not support it, the highest \
         supported level below it is used. avx512-ifma and avx512 compute 8 keys at a time, avx2 4, \
         and off 1 (the SHA-512 and multiplication before the speedups). amx is an experimental path \
         that multiplies field elements with AMX tile operations (slower than IFMA; Linux only; \
         available only when built with the amx feature)",
    ),
    (
        "no_memmem",
        "公開鍵の中を memmem（SIMD の部分文字列検索）で探さず、1文字ずつずらして比べる（高速化前の検索。比較用）",
        "Compare at each position in turn instead of searching the public key with memmem \
         (SIMD substring search) (the search before the speedups; for comparison)",
    ),
    (
        "gpu",
        "CPU ではなく GPU（CUDA）で探す（--low-security と一緒に付ける。GPU が複数あれば全部使う）。\
         --threads・--no-batch・--simd・--no-memmem は CPU 用なので使わない",
        "Search on the GPU (CUDA) instead of the CPU (use with --low-security; all GPUs are used if \
         there are several). --threads, --no-batch, --simd and --no-memmem are for the CPU and are \
         not used",
    ),
    (
        "both",
        "GPU と CPU の両方で同時に探す（--low-security と一緒に付ける。CPU のスレッド数の既定は 全コア − GPU の数。\
         GPU 1 枚に 1 本を操作に使う）",
        "Search on the GPUs and the CPU at the same time (use with --low-security; the default \
         number of CPU threads is all cores − the number of GPUs; one thread drives each GPU)",
    ),
    (
        "gpu_width",
        "GPU の表の幅 W（2〜20）。大きいほど点の足し算が減り、表は大きくなる（12 で約 4.1 MB、16 で約 50 MB、\
         20 で約 650 MB）。--gpu-width・--gpu-batch・--gpu-limbs を付けなければ、探す前にこの GPU で測って\
         一番速い組み合わせを使う",
        "Width W of the GPU table (2-20). Larger widths need fewer point additions and a bigger \
         table (about 4.1 MB at 12, about 50 MB at 16, about 650 MB at 20). Without --gpu-width, \
         --gpu-batch and --gpu-limbs, the fastest combination is measured on this GPU before \
         searching",
    ),
    (
        "gpu_batch",
        "GPU の1スレッドが、逆元を1回にまとめて計算する鍵の数（1〜64）。16・32 以外は、起動時に NVRTC で\
         カーネルをコンパイルする",
        "Number of keys per GPU thread whose inverses are computed together (1-64). Values other \
         than 16 and 32 are compiled with NVRTC at startup",
    ),
    (
        "gpu_limbs",
        "GPU の体演算の種類（10 = 25.5 ビット × 10、8 = 32 ビット × 8、10k = 26 ビット × 10 の積を Karatsuba 法で、\
         5f = 51 ビット × 5 の積を倍精度浮動小数点数で。10u・10ku は 10・10k で、表をリムの形のまま持つ）。\
         速いものは GPU で違う",
        "Kind of GPU field arithmetic (10 = 25.5 bits x 10, 8 = 32 bits x 8, 10k = 26 bits x 10 \
         multiplied with Karatsuba, 5f = 51 bits x 5 multiplied in double precision; 10u and 10ku are \
         10 and 10k with the table stored as limbs). Which is fastest depends on the GPU",
    ),
];

/// 今の言語のヘルプ・見出しを付けた、引数の読み方。日本語では clap が英語で付ける
/// 「Usage:」「Options:」「[default: …]」「[possible values: …]」と -h・-V の説明も日本語にする
fn command() -> clap::Command {
    use clap::{Arg, ArgAction, CommandFactory};
    let lang = current();
    let pick = |ja: &'static str, en: &'static str| if lang == Lang::Ja { ja } else { en };
    let mut cmd = Cli::command()
        .about(pick(ABOUT.0, ABOUT.1))
        .long_about(None);
    for &(id, ja, en) in HELP {
        // feature で無くなる引数（--gpu など）は飛ばす
        if cmd.get_arguments().any(|a| a.get_id() == id) {
            cmd = cmd.mut_arg(id, |a| a.help(pick(ja, en)));
        }
    }
    if lang == Lang::En {
        return cmd;
    }
    cmd = cmd
        .help_template("{about-with-newline}\n使い方: {usage}\n\n{all-args}{after-help}")
        .disable_help_flag(true)
        .disable_version_flag(true)
        .arg(
            Arg::new("help")
                .short('h')
                .long("help")
                .action(ArgAction::Help)
                .help("ヘルプを表示する"),
        )
        .arg(
            Arg::new("version")
                .short('V')
                .long("version")
                .action(ArgAction::Version)
                .help("版を表示する"),
        );
    cmd.mut_args(|a| {
        let mut help = a.get_help().map(|h| h.to_string()).unwrap_or_default();
        let heading = if a.is_positional() {
            "引数"
        } else {
            "オプション"
        };
        // 値を取らない引数（--force など）には既定値・値の一覧を付けない（clap も付けない）
        if !a.get_action().takes_values() {
            return a.help(help).help_heading(heading);
        }
        let defaults: Vec<String> = a
            .get_default_values()
            .iter()
            .map(|v| v.to_string_lossy().into_owned())
            .filter(|v| !v.is_empty())
            .collect();
        if !defaults.is_empty() {
            help += &format!(" [既定: {}]", defaults.join(", "));
        }
        let values: Vec<String> = a
            .get_possible_values()
            .iter()
            .filter(|v| !v.is_hide_set())
            .map(|v| v.get_name().to_string())
            .collect();
        if !values.is_empty() {
            help += &format!(" [使える値: {}]", values.join(", "));
        }
        a.help(help)
            .help_heading(heading)
            .hide_default_value(true)
            .hide_possible_values(true)
    })
}

/// 引数を読む。誤りがあれば、今の言語で説明して終わる（終了コード 2）
fn parse_from<I, T>(args: I) -> Result<Cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    use clap::FromArgMatches;
    Cli::from_arg_matches(&command().try_get_matches_from(args)?)
}

/// 引数の誤りを説明して終わる。日本語のときは clap の英語の説明を日本語にする
/// （日本語にしていない種類の誤りと、ヘルプ・版の表示は clap のまま）
fn exit_with(e: clap::Error) -> ! {
    if current() == Lang::Ja
        && let Some(message) = japanese_error(&e)
    {
        // その誤りに合わせた使い方（無ければ全体の使い方）
        let usage = e
            .get(clap::error::ContextKind::Usage)
            .map_or_else(|| command().render_usage().to_string(), |u| u.to_string());
        let usage = usage.strip_prefix("Usage: ").unwrap_or(&usage);
        eprintln!("エラー: {message}\n\n使い方: {usage}\n\n詳しくは --help");
        std::process::exit(2);
    }
    e.exit()
}

/// clap の誤りの日本語の説明（日本語にしていない種類なら None）
fn japanese_error(e: &clap::Error) -> Option<String> {
    use clap::error::{ContextKind, ErrorKind};
    let get = |k| e.get(k).map(|v| v.to_string()).unwrap_or_default();
    let (arg, value) = (get(ContextKind::InvalidArg), get(ContextKind::InvalidValue));
    let message = match e.kind() {
        ErrorKind::MissingRequiredArgument => format!("必要な引数が無い: {arg}"),
        ErrorKind::UnknownArgument => {
            let mut m = format!("知らない引数: '{arg}'");
            if let Some(s) = e.get(ContextKind::SuggestedArg) {
                m += &format!("（もしかして '{s}'）");
            }
            m
        }
        ErrorKind::InvalidValue if value.is_empty() => format!("'{arg}' に値が要る"),
        ErrorKind::InvalidValue => {
            let mut m = format!("'{arg}' に使えない値: '{value}'");
            if let Some(v) = e.get(ContextKind::ValidValue) {
                m += &format!("（使える値: {v}）");
            }
            m
        }
        ErrorKind::ValueValidation => {
            let source = std::error::Error::source(e)
                .map(|s| s.to_string())
                .unwrap_or_default();
            format!("'{arg}' の値 '{value}' が正しくない: {source}")
        }
        ErrorKind::ArgumentConflict => {
            format!(
                "'{arg}' と '{}' は一緒に付けられない",
                get(ContextKind::PriorArg)
            )
        }
        _ => return None,
    };
    Some(message)
}

/// `lo` 以上 `hi` 以下の整数
fn int_in(s: &str, lo: u32, hi: u32) -> Result<u32, ValueError> {
    s.parse()
        .ok()
        .filter(|n| (lo..=hi).contains(n))
        .ok_or(ValueError::NotInRange { lo, hi })
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

mod error;
mod interactive;

pub use error::{CliError, ValueError};
pub use interactive::AskError;

/// 引数を読んで実行し、終了コードを返す。引数なしで端末から起動したら、質問に答えてもらって実行する
pub fn main() -> ExitCode {
    let interactive = interactive::wanted();
    let result = match interactive {
        true => interactive::run(),
        false => execute(&parse_from(std::env::args_os()).unwrap_or_else(|e| exit_with(e))),
    };
    let code = match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}", tf!("エラー: {e}", "error: {e}"));
            ExitCode::FAILURE
        }
    };
    if interactive {
        interactive::finish();
    }
    code
}

/// 探すのに使うもの（CPU・GPU・両方）を決めて実行する
fn execute(cli: &Cli) -> Result<(), CliError> {
    // Windows で、前面にないと数秒で CPU を絞られる（探索が数分の一に遅くなる）のを防ぐ
    crate::service::power::keep_full_speed();
    match device(cli) {
        Device::Cpu => run(cli, &Parallel, &KeyFiles),
        #[cfg(feature = "cuda")]
        Device::Gpu => run(cli, &gpus(cli)?, &KeyFiles),
        #[cfg(feature = "cuda")]
        Device::Both => run(
            cli,
            &crate::service::search::Both(gpus(cli)?, Parallel),
            &KeyFiles,
        ),
    }
}

/// 全部の GPU を使えるようにする。設定を測ると数秒かかるので、引数の誤りや既にある出力先は先に確かめる。
/// 設定を測るときは、同じ型の GPU（名前と世代が同じ）では最初の1枚で測った設定を使う
#[cfg(feature = "cuda")]
fn gpus(cli: &Cli) -> Result<crate::service::search::All<crate::service::gpu::Gpu>, CliError> {
    use crate::service::gpu::{Gpu, GpuSetting, device_count};
    if !cli.check {
        pattern(cli)?;
        check_comment(&cli.comment)?;
        if cli.bench.is_none() {
            check_output(cli, &KeyFiles)?;
        }
    }
    let setting = gpu_options(cli);
    // GPU が使えるとわかって、測り始めるときに知らせる（測ると数秒かかるので）
    let notify = |n: usize| {
        eprintln!(
            "{}",
            tf!(
                "GPU の設定を測っています（{} 通り）",
                "Measuring GPU settings ({} combinations)",
                n
            )
        )
    };
    // ドライバや GPU が無ければ、1 枚目を開くところでそのエラーになる
    let count = device_count().max(1);
    let mut out: Vec<Gpu> = Vec::new();
    for i in 0..count {
        let info = Gpu::info_of(i)?;
        let same = out
            .iter()
            .find(|g| g.info().name == info.name && g.info().compute == info.compute);
        let gpu = match (setting, same) {
            (GpuSetting::Fastest, Some(g)) => {
                Gpu::new(i, GpuSetting::Fixed(g.options()), |_| {})?.same_as_another()
            }
            _ => Gpu::new(i, setting, notify)?,
        };
        out.push(gpu);
    }
    if out.len() > 1 {
        out = out
            .into_iter()
            .enumerate()
            .map(|(i, g)| g.numbered(i))
            .collect();
    }
    Ok(crate::service::search::All(out))
}

/// GPU の設定の決め方: --gpu-width・--gpu-batch・--gpu-limbs のどれかがあればそれ（無いものは既定）、
/// --check は測らずに既定、それ以外はこの GPU で測って一番速いもの
#[cfg(feature = "cuda")]
fn gpu_options(cli: &Cli) -> crate::service::gpu::GpuSetting {
    use crate::service::gpu::{GpuOptions, GpuSetting};
    let default = GpuOptions::default();
    match (cli.gpu_width, cli.gpu_batch, cli.gpu_limbs) {
        (None, None, None) if cli.check => GpuSetting::Default,
        (None, None, None) => GpuSetting::Fastest,
        (width, batch, limbs) => GpuSetting::Fixed(GpuOptions::new(
            width.unwrap_or(default.width),
            batch.unwrap_or(default.batch),
            limbs.unwrap_or(default.limbs),
        )),
    }
}

/// GPU を操作するスレッドの数（GPU の数。分からなければ 1）
fn gpu_threads() -> usize {
    #[cfg(feature = "cuda")]
    return crate::service::gpu::device_count().max(1);
    #[cfg(not(feature = "cuda"))]
    1
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

fn run(cli: &Cli, searcher: &impl Searcher, repo: &impl KeyRepository) -> Result<(), CliError> {
    let mut config = SearchConfig::new(None);
    if let Some(n) = cli.threads {
        config.threads = n.into();
    } else if device(cli) != Device::Cpu {
        // GPU と両方で探すときは、GPU を操作するスレッドの分（GPU 1 枚に 1 本）を空ける
        // （GPU だけのときは CPU で探さないので、この値は使わない）
        config.threads = config.threads.saturating_sub(gpu_threads()).max(1);
    }
    // 既定は定数時間の倍算と ChaCha20 のシード。低セキュリティモードでは表の倍算と通し番号のシード
    match cli.low_security {
        true => config.seeds = SeedSource::Counter,
        false => config.backend.mul = Mul::ConstantTime,
    }
    if cli.no_batch {
        config.backend.compress = Compress::Each;
    }
    config.backend.simd = cli.simd.into();
    if cli.no_memmem {
        config.find = Find::Naive;
    }

    if cli.check {
        let mut rows = searcher.status(&config);
        rows.push(seed_row(&config));
        print_rows(&rows);
        return Ok(());
    }

    let pattern = pattern(cli)?;
    check_comment(&cli.comment)?;
    config.pattern = Some(pattern.clone());

    let mut rows = searcher.status(&config);
    rows.push(seed_row(&config));
    rows.extend([
        (t!("文字列", "Text"), describe(&pattern)),
        (
            t!("期待試行", "Est. tries"),
            about(&format::count(pattern.expected_tries())),
        ),
    ]);
    print_rows(&rows);

    match cli.bench {
        Some(limit) => bench(searcher, config, &pattern, limit),
        None => generate(cli, searcher, repo, config, &pattern),
    }
}

/// 探す文字列と探し方（--suffix・--regex）
fn pattern(cli: &Cli) -> Result<Pattern, CliError> {
    let mode = match (cli.suffix, cli.regex) {
        (true, _) => Mode::Suffix,
        (_, true) => Mode::Regex,
        _ => Mode::Contains,
    };
    Ok(Pattern::with_mode(
        cli.target.as_ref().unwrap(),
        cli.case_sensitive,
        mode,
    )?)
}

/// 状態の表示の「文字列」の値（探し方と、大小文字を区別するか）
fn describe(pattern: &Pattern) -> String {
    let case = match pattern.case_sensitive() {
        true => t!("大小文字の区別あり", "case-sensitive"),
        false => t!("大小文字の区別なし", "case-insensitive"),
    };
    match pattern.mode() {
        Mode::Contains => tf!("{}（{case}）", "{} ({case})", pattern.as_str()),
        mode => tf!(
            "{}（{}・{case}）",
            "{} ({}, {case})",
            pattern.as_str(),
            mode.name()
        ),
    }
}

/// 0 より大きい秒数（小数可）
fn parse_secs(s: &str) -> Result<Duration, ValueError> {
    let secs: f64 = s.parse().map_err(|_| ValueError::NotSeconds)?;
    Duration::try_from_secs_f64(secs)
        .ok()
        .filter(|d| !d.is_zero())
        .ok_or(ValueError::NotPositive)
}

/// 状態の表示の「シード」の行（低セキュリティモードかどうかも分かるように）
fn seed_row(config: &SearchConfig) -> (&'static str, String) {
    let how = match config.seeds {
        SeedSource::ChaCha20 => t!("鍵ごとに ChaCha20", "ChaCha20 for each key"),
        SeedSource::Counter => t!(
            "乱数 + 通し番号（低セキュリティモード）",
            "random + counter (low-security mode)"
        ),
    };
    (t!("シード", "Seeds"), how.to_string())
}

/// 「約 …」（期待試行・期待時間・残り時間）
fn about(value: &str) -> String {
    tf!("約 {value}", "about {value}")
}

/// 「… 鍵/秒」
fn per_sec(rate: f64) -> String {
    tf!("{} 鍵/秒", "{} keys/s", format::count(rate))
}

fn bench(
    searcher: &impl Searcher,
    mut config: SearchConfig,
    pattern: &Pattern,
    limit: Duration,
) -> Result<(), CliError> {
    // 文字列の判定まで含めて測る（当たっても止まらない）
    config.bench = true;
    config.time_limit = Some(limit);
    config.progress_interval = Duration::from_secs(1);

    print_rows(&[(
        t!("計測", "Measure"),
        tf!("{} 秒", "{} s", limit.as_secs_f64()),
    )]);
    let mut last = Stats {
        tries: 0,
        elapsed: Duration::ZERO,
    };
    let out = searcher.run(&config, &mut |now: Stats| {
        let delta = since(now, last);
        eprintln!(
            "{:>5.1}s  {}",
            now.elapsed.as_secs_f64(),
            per_sec(delta.rate())
        );
        last = now;
    });

    let rate = out.stats.rate();
    println!("{}", format::row(t!("速度", "Speed"), &per_sec(rate)));
    println!(
        "{}",
        format::row(
            t!("期待時間", "Est. time"),
            &about(&format::duration_secs(pattern.expected_tries() / rate))
        )
    );
    Ok(())
}

/// 出力先に鍵のファイルが既にある（--force でなければ）か、そのディレクトリに書けなければエラー
fn check_output(cli: &Cli, repo: &impl KeyRepository) -> Result<(), CliError> {
    if !cli.force
        && let Some(p) = repo.existing(&cli.output)?
    {
        return Err(CliError::OutputExists(p));
    }
    repo.check_writable(&cli.output)
        .map_err(CliError::NotWritable)
}

/// 出力先に保存できなかったときに、代わりに保存する先（今のディレクトリ、一時ディレクトリの順）。
/// 名前は出力先の名前と、保存した時刻（Unix 時間の秒）と乱数。一時ディレクトリはほかのユーザーも
/// ファイルを作れるので、先に同じ名前を作られて保存を邪魔されないよう、推測できない名前にする
fn fallback_paths(output: &Path) -> Vec<PathBuf> {
    let name = output
        .file_name()
        .map_or_else(|| "id_ed25519_meikey".into(), |n| n.to_string_lossy());
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let name = format!("{name}-{secs}.{:08x}", rand::random::<u32>());
    vec![PathBuf::from(&name), std::env::temp_dir().join(&name)]
}

fn generate(
    cli: &Cli,
    searcher: &impl Searcher,
    repo: &impl KeyRepository,
    config: SearchConfig,
    pattern: &Pattern,
) -> Result<(), CliError> {
    // 何時間も回したあとで書き込めない、を防ぐため先に確認する
    // （書き込み時も確かめ、間に作られたファイルは上書きしない）
    check_output(cli, repo)?;

    let expected = pattern.expected_tries();

    let mut shown = false;
    let out = searcher.run(&config, &mut |s: Stats| {
        shown = true;
        let remaining = remaining_secs(expected, s);
        let (elapsed, tries, rate, remaining) = (
            format::duration(s.elapsed),
            format::count(s.tries as f64),
            per_sec(s.rate()),
            format::duration_secs(remaining),
        );
        eprint!(
            "\r{}    ",
            tf!(
                "経過 {elapsed}  試行 {tries}  {rate}  残り約 {remaining}",
                "Elapsed {elapsed}  Tries {tries}  {rate}  About {remaining} left"
            )
        );
        let _ = io::stderr().flush();
    });
    if shown {
        // 進捗の行を改行で終える
        eprintln!();
    }

    let key = out.key.ok_or(CliError::NotFound)?;
    if let Err(e) = repo.save(&key, &cli.comment, &cli.output, cli.force) {
        // 探す前に確かめても、探している間に書けなくなることはある。見つけた鍵を失わないよう、
        // 代わりの場所に保存してからエラーにする
        println!("{}", key.public_openssh(&cli.comment));
        let saved = fallback_paths(&cli.output)
            .into_iter()
            .find(|p| repo.save(&key, &cli.comment, p, false).is_ok());
        return Err(match saved {
            Some(private) => CliError::SavedElsewhere { source: e, private },
            None => CliError::NotSaved(e),
        });
    }

    print_rows(&[
        (
            t!("試行", "Tries"),
            tf!("{} 鍵", "{} keys", format::count(out.stats.tries as f64)),
        ),
        (t!("経過", "Elapsed"), format::duration(out.stats.elapsed)),
    ]);
    let line = key.public_openssh(&cli.comment);
    println!("{line}");
    if let Some(r) = pattern.find(&key.public_b64()) {
        // "ssh-ed25519 " の分だけずらして、一致した場所に ^ を付ける
        let offset = "ssh-ed25519 ".len();
        eprintln!("{}{}", " ".repeat(offset + r.start), "^".repeat(r.len()));
    }
    print_rows(&[
        (
            t!("秘密鍵", "Private key"),
            cli.output.display().to_string(),
        ),
        (
            t!("公開鍵", "Public key"),
            pub_path(&cli.output).display().to_string(),
        ),
        (
            t!("パスフレーズ", "Passphrase"),
            tf!(
                "なし（設定: ssh-keygen -p -f {}）",
                "none (to set one: ssh-keygen -p -f {})",
                cli.output.display()
            ),
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
    #[cfg(feature = "cuda")]
    use crate::service::gpu::{GpuOptions, GpuSetting, Limbs};
    use crate::service::search::MockSearcher;
    use mockall::Sequence;
    use mockall::predicate::eq;
    use rstest::{fixture, rstest};
    use similar_asserts::assert_eq;
    use std::iter::once;
    use std::path::Path;

    /// 鍵のファイルの読み書きのエラー（モックが返すもの）
    fn io_error(path: &str, message: &str) -> crate::repository::key_file::KeyFileError {
        crate::repository::key_file::KeyFileError::Io {
            path: path.into(),
            source: io::Error::other(message.to_string()),
        }
    }

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

    /// 既存ファイルの確認（--force なら省く）→ 書けるかの確認 → 探索 → 保存 の順に呼び、見つかった鍵を
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
        repo.expect_check_writable()
            .with(eq(Path::new("out/id_test")))
            .times(1)
            .in_sequence(&mut seq)
            .returning(|_| Ok(()));
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
        repo.expect_check_writable().never();
        repo.expect_save().never();

        searcher.expect_run().never();

        let e = run(&cli, &searcher, &repo).unwrap_err();
        assert_eq!(
            matches!(&e, CliError::OutputExists(p) if p == Path::new(already)),
            true,
            "{e:?}"
        );
        let e = e.to_string();
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
            .returning(|_| Err(io_error("id_ed25519_meikey", "読めない")));

        searcher.expect_run().never();

        let e = run(&cli, &searcher, &repo).unwrap_err();
        assert_eq!(matches!(e, CliError::KeyFile(_)), true, "{e:?}");
        assert_eq!(e.to_string(), "id_ed25519_meikey: 読めない");
    }

    /// 出力先のディレクトリに書けなければ（無い・権限など）、探す前にそのエラーで止まる
    #[rstest]
    fn unwritable_output_stops_before_search(
        mut searcher: MockSearcher,
        mut repo: MockKeyRepository,
    ) {
        let cli = parse(&["abc", "-o", "nodir/id_test"]);

        repo.expect_existing().returning(|_| Ok(None));
        repo.expect_check_writable()
            .with(eq(Path::new("nodir/id_test")))
            .times(1)
            .returning(|_| Err(io_error("nodir", "無い")));
        repo.expect_save().never();
        searcher.expect_run().never();

        let e = run(&cli, &searcher, &repo).unwrap_err();
        assert_eq!(matches!(e, CliError::NotWritable(_)), true, "{e:?}");
        assert_eq!(e.to_string(), "保存できない: nodir: 無い");
    }

    /// 見つからずに終わった（時間切れ・パニック後の停止など）なら、エラーにして何も保存しない
    #[rstest]
    fn not_found_saves_nothing(mut searcher: MockSearcher, mut repo: MockKeyRepository) {
        let cli = parse(&["abc"]);

        repo.expect_existing().returning(|_| Ok(None));
        repo.expect_check_writable().returning(|_| Ok(()));
        repo.expect_save().never();

        searcher.expect_run().returning(|_, _| Outcome {
            key: None,
            stats: stats(10, 5),
        });

        let e = run(&cli, &searcher, &repo).unwrap_err();
        assert_eq!(matches!(e, CliError::NotFound), true, "{e:?}");
        assert_eq!(e.to_string(), "鍵が見つからなかった");
    }

    /// 保存に失敗したら（探索中に作られたファイル・書き込み権限など）、見つけた鍵を代わりの場所
    /// （今のディレクトリ、だめなら一時ディレクトリ）に上書きせずに保存し、元のエラーと保存先を返す
    #[rstest]
    #[case::cwd(&[true], Some(0))]
    #[case::temp_dir(&[false, true], Some(1))]
    #[case::nowhere(&[false, false], None)]
    fn save_error_falls_back(
        #[case] fallback_ok: &'static [bool],
        #[case] saved_at: Option<usize>,
        mut searcher: MockSearcher,
        found: (VanityKey, String),
        mut repo: MockKeyRepository,
    ) {
        let (key, target) = found;
        let cli = parse(&[&target, "-o", "out/id_test"]);
        let fallbacks = fallback_paths(Path::new("out/id_test"));
        assert_eq!(
            fallbacks[0].parent(),
            Some(Path::new("")),
            "1つ目は今のディレクトリ"
        );
        assert_eq!(fallbacks[1].parent(), Some(std::env::temp_dir().as_path()));

        repo.expect_existing().returning(|_| Ok(None));
        repo.expect_check_writable().returning(|_| Ok(()));
        let mut seq = Sequence::new();
        repo.expect_save()
            .withf(|_, _, p, _| p == Path::new("out/id_test"))
            .times(1)
            .in_sequence(&mut seq)
            .returning(|_, _, _, _| Err(io_error("out/id_test", "書けない")));
        for &ok in fallback_ok {
            repo.expect_save()
                .withf(|_, comment, p, overwrite| {
                    comment.is_empty() && p.to_string_lossy().contains("id_test-") && !*overwrite
                })
                .times(1)
                .in_sequence(&mut seq)
                .returning(move |_, _, _, _| match ok {
                    true => Ok(()),
                    false => Err(io_error("x", "x")),
                });
        }
        searcher.expect_run().returning(move |_, _| Outcome {
            key: Some(key.clone()),
            stats: stats(10, 5),
        });

        let e = run(&cli, &searcher, &repo).unwrap_err();
        let message = e.to_string();
        assert_eq!(
            message.starts_with("out/id_test: 書けない。"),
            true,
            "{message}"
        );
        match (saved_at, e) {
            (Some(i), CliError::SavedElsewhere { source, private }) => {
                assert_eq!(source.to_string(), "out/id_test: 書けない");
                // 名前の最後（保存した時刻と乱数）は、呼ぶたびに変わる
                let name = fallbacks[i].to_string_lossy().into_owned();
                let prefix = &name[..name.rfind('-').unwrap() + 1];
                assert_eq!(
                    private.to_string_lossy().starts_with(prefix),
                    true,
                    "{private:?}"
                );
                assert_eq!(message.contains("代わりに"), true, "{message}");
            }
            (None, CliError::NotSaved(source)) => {
                assert_eq!(source.to_string(), "out/id_test: 書けない");
                assert_eq!(message.contains("にも保存できなかった"), true, "{message}");
            }
            (want, e) => panic!("{want:?} 番目に保存したはずが {e:?}"),
        }
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
        let parsed = Cli::try_parse_from(["meikey", "abc", "--low-security"].iter().chain(options));
        assert_eq!(parsed.ok().map(|c| device(&c)), want);
    }

    /// GPU の設定: 何も付けなければ測って一番速いもの、--check なら測らずに既定、--gpu-width・--gpu-batch・
    /// --gpu-limbs は範囲内の値だけ受け付けてそのまま使う（付けなかったものは既定）
    #[cfg(feature = "cuda")]
    #[rstest]
    #[case::fastest(&[], Some(GpuSetting::Fastest))]
    #[case::check(&["--check"], Some(GpuSetting::Default))]
    #[case::all(&["--gpu-width", "18", "--gpu-batch", "64", "--gpu-limbs", "8"], Some(GpuSetting::Fixed(GpuOptions::new(18, 64, Limbs::R32))))]
    #[case::width_only(&["--gpu-width", "14"], Some(GpuSetting::Fixed(GpuOptions::new(14, 32, Limbs::R25))))]
    #[case::batch_only(&["--gpu-batch", "64"], Some(GpuSetting::Fixed(GpuOptions::new(16, 64, Limbs::R25))))]
    #[case::limbs_only(&["--gpu-limbs", "8"], Some(GpuSetting::Fixed(GpuOptions::new(16, 32, Limbs::R32))))]
    #[case::karatsuba(&["--gpu-limbs", "10k"], Some(GpuSetting::Fixed(GpuOptions::new(16, 32, Limbs::K26))))]
    #[case::karatsuba_table(&["--gpu-limbs", "10ku"], Some(GpuSetting::Fixed(GpuOptions::new(16, 32, Limbs::K26Table))))]
    #[case::unpacked(&["--gpu-limbs", "10u"], Some(GpuSetting::Fixed(GpuOptions::new(16, 32, Limbs::R25Table))))]
    #[case::fp64(&["--gpu-limbs", "5f"], Some(GpuSetting::Fixed(GpuOptions::new(16, 32, Limbs::F51))))]
    #[case::check_with_width(&["--check", "--gpu-width", "12"], Some(GpuSetting::Fixed(GpuOptions::new(12, 32, Limbs::R25))))]
    #[case::smallest(&["--gpu-width", "2", "--gpu-batch", "1"], Some(GpuSetting::Fixed(GpuOptions::new(2, 1, Limbs::R25))))]
    #[case::limbs_other(&["--gpu-limbs", "9"], None)]
    #[case::widest(&["--gpu-width", "20"], Some(GpuSetting::Fixed(GpuOptions::new(20, 32, Limbs::R25))))]
    #[case::width_too_large(&["--gpu-width", "21"], None)]
    #[case::width_too_small(&["--gpu-width", "1"], None)]
    #[case::batch_zero(&["--gpu-batch", "0"], None)]
    #[case::batch_too_large(&["--gpu-batch", "65"], None)]
    fn gpu_options_from_flags(#[case] options: &[&str], #[case] want: Option<GpuSetting>) {
        let parsed = Cli::try_parse_from(
            ["meikey", "abc", "--gpu", "--low-security"]
                .iter()
                .chain(options),
        );
        assert_eq!(parsed.ok().map(|c| gpu_options(&c)), want);
    }

    /// --gpu・--both は低セキュリティモードでだけ使える（GPU の計算は定数時間でない）。
    /// --gpu-width・--gpu-batch・--gpu-limbs は GPU で探すときだけ付けられる
    #[cfg(feature = "cuda")]
    #[rstest]
    #[case::gpu_without_low_security(&["--gpu"], false)]
    #[case::both_without_low_security(&["--both"], false)]
    #[case::gpu(&["--gpu", "--low-security"], true)]
    #[case::both(&["--both", "--low-security"], true)]
    #[case::low_security_cpu(&["--low-security"], true)]
    #[case::gpu_and_both(&["--gpu", "--both", "--low-security"], false)]
    #[case::width_without_gpu(&["--gpu-width", "16", "--low-security"], false)]
    #[case::batch_without_gpu(&["--gpu-batch", "32", "--low-security"], false)]
    #[case::width_with_gpu(&["--gpu", "--gpu-width", "16", "--low-security"], true)]
    #[case::batch_with_both(&["--both", "--gpu-batch", "32", "--low-security"], true)]
    #[case::limbs_without_gpu(&["--gpu-limbs", "8", "--low-security"], false)]
    #[case::limbs_with_gpu(&["--gpu", "--gpu-limbs", "8", "--low-security"], true)]
    #[case::check_width_without_gpu(&["--check", "--gpu-width", "16", "--low-security"], false)]
    fn gpu_flag_combinations(#[case] options: &[&str], #[case] accepted: bool) {
        let parsed = Cli::try_parse_from(["meikey", "abc"].iter().chain(options));
        assert_eq!(parsed.is_ok(), accepted, "{:?}", parsed.err());
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
    #[cfg_attr(feature = "cuda", case::both(&["--both", "--low-security"], |all: usize| all.saturating_sub(1).max(1)))]
    #[cfg_attr(feature = "cuda", case::both_explicit(&["--both", "--low-security", "-j", "3"], |_| 3))]
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
    // 既定は定数時間の倍算（ほかの高速化はあり）と ChaCha20 のシード
    #[case::defaults(&[], |c: &SearchConfig| {
        c.backend == Backend { mul: Mul::ConstantTime, ..Backend::FAST }
            && c.find == Find::Memmem
            && c.seeds == SeedSource::ChaCha20
            && !c.pattern.as_ref().unwrap().case_sensitive()
    })]
    // 低セキュリティモードは表の倍算と通し番号のシード
    #[case::low_security(&["--low-security"], |c: &SearchConfig| {
        c.backend == Backend::FAST && c.seeds == SeedSource::Counter
    })]
    #[case::threads(&["-j", "3"], |c: &SearchConfig| c.threads == 3)]
    #[case::case_sensitive(&["-s"], |c: &SearchConfig| c.pattern.as_ref().unwrap().case_sensitive())]
    #[case::no_batch(&["--no-batch"], |c: &SearchConfig| c.backend.compress == Compress::Each)]
    #[case::simd_avx512(&["--simd", "avx512"], |c: &SearchConfig| c.backend.simd == Simd::Avx512)]
    #[case::simd_avx2(&["--simd", "avx2"], |c: &SearchConfig| c.backend.simd == Simd::Avx2)]
    #[case::simd_off(&["--simd", "off"], |c: &SearchConfig| c.backend.simd == Simd::Off)]
    #[case::no_memmem(&["--no-memmem"], |c: &SearchConfig| c.find == Find::Naive)]
    #[case::baseline(
        &["--no-batch", "--simd", "off", "--no-memmem"],
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
        repo.expect_check_writable().returning(|_| Ok(()));

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

    /// --simd に使える値の一覧（amx は feature amx が有効なときだけ）
    const SIMD_VALUES: &str = if cfg!(feature = "amx") {
        "amx, avx512-ifma, avx512, avx2, off"
    } else {
        "avx512-ifma, avx512, avx2, off"
    };

    /// どの言語でも、引数の読み方は clap の検査に通り、どの引数にもその言語のヘルプが付く
    /// （引数を足して HELP に書き忘れると、ここで落ちる）
    #[rstest]
    fn every_argument_has_help(#[values(Lang::Ja, Lang::En)] lang: Lang) {
        crate::lang::set(lang);
        let cmd = command();
        cmd.clone().debug_assert();
        for a in cmd.get_arguments() {
            let id = a.get_id().as_str();
            let help = a.get_help().map(|h| h.to_string()).unwrap_or_default();
            assert_eq!(help.is_empty(), false, "{id} にヘルプが無い");
            let japanese = help.chars().any(|c| !c.is_ascii() && c != '−');
            assert_eq!(japanese, lang == Lang::Ja, "{id}: {help}");
        }
        for &(id, _, _) in HELP {
            let known = cmd.get_arguments().any(|a| a.get_id() == id);
            let gpu_only = matches!(id, "gpu" | "both" | "gpu_width" | "gpu_batch" | "gpu_limbs");
            assert_eq!(
                known || (gpu_only && !cfg!(feature = "cuda")),
                true,
                "{id} は無い引数"
            );
        }
    }

    /// 日本語では、clap の英語の見出し・既定値・値の一覧・-h と -V の説明を日本語にする
    #[rstest]
    #[case::ja(Lang::Ja, &["使い方: meikey", "引数:", "オプション:", "[既定: id_ed25519_meikey]", "[使える値: {SIMD}]", "ヘルプを表示する", "版を表示する"], &["Usage:", "Options:", "[default:", "[possible values:", "Print help"])]
    #[case::en(Lang::En, &["Usage: meikey", "Arguments:", "Options:", "[default: id_ed25519_meikey]", "[possible values: {SIMD}]", "Print help", "Print version"], &["使い方", "既定", "[default: \"\"]"])]
    fn help_in_language(#[case] lang: Lang, #[case] has: &[&str], #[case] lacks: &[&str]) {
        crate::lang::set(lang);
        let help = command().render_long_help().to_string();
        for want in has {
            let want = &want.replace("{SIMD}", SIMD_VALUES);
            assert_eq!(help.contains(want), true, "{want:?} が無い:\n{help}");
        }
        for not in lacks {
            assert_eq!(help.contains(not), false, "{not:?} がある:\n{help}");
        }
        // 引数（TARGET）の見出しがオプションより先
        let (args, options) = match lang {
            Lang::Ja => ("引数:", "オプション:"),
            Lang::En => ("Arguments:", "Options:"),
        };
        assert_eq!(help.find(args) < help.find(options), true, "{help}");
    }

    /// 日本語では clap の引数の誤りを日本語で説明する（英語なら clap のまま = None）
    #[rstest]
    #[case::missing(&[], "必要な引数が無い: <TARGET>")]
    #[case::unknown(&["abc", "--nope"], "知らない引数: '--nope'")]
    #[case::suggestion(&["abc", "--forc"], "知らない引数: '--forc'（もしかして '--force'）")]
    #[case::extra(&["abc", "def"], "知らない引数: 'def'")]
    #[case::no_value(&["abc", "--output"], "'--output <OUTPUT>' に値が要る")]
    #[case::not_in_list(&["abc", "--simd", "avx1024"], "'--simd <LEVEL>' に使えない値: 'avx1024'（使える値: {SIMD}）")]
    #[case::threads_zero(&["abc", "-j", "0"], "'--threads <THREADS>' の値 '0' が正しくない: 1〜65535 の整数が必要")]
    #[case::threads_not_number(&["abc", "-j", "x"], "'--threads <THREADS>' の値 'x' が正しくない: 1〜65535 の整数が必要")]
    #[case::bench_zero(&["abc", "--bench", "0"], "'--bench <SECS>' の値 '0' が正しくない: 0 より大きい秒数が必要")]
    #[case::bench_not_number(&["abc", "--bench", "x"], "'--bench <SECS>' の値 'x' が正しくない: 秒数ではない")]
    #[cfg_attr(feature = "cuda", case::conflict(&["abc", "--low-security", "--gpu", "--both"], "'--gpu' と '--both' は一緒に付けられない"))]
    #[cfg_attr(feature = "cuda", case::gpu_without_low_security(&["abc", "--gpu"], "必要な引数が無い: --low-security"))]
    #[cfg_attr(feature = "cuda", case::gpu_width_alone(&["abc", "--gpu-width", "16"], "必要な引数が無い: <--gpu|--both>"))]
    #[cfg_attr(feature = "cuda", case::gpu_width_range(&["abc", "--low-security", "--gpu", "--gpu-width", "21"], "'--gpu-width <W>' の値 '21' が正しくない: 2〜20 の整数が必要"))]
    fn clap_errors_in_japanese(#[case] args: &[&str], #[case] want: &str) {
        let parse = || {
            parse_from(once("meikey").chain(args.iter().copied()))
                .err()
                .unwrap()
        };
        let want = want.replace("{SIMD}", SIMD_VALUES);
        assert_eq!(japanese_error(&parse()), Some(want));
        // 英語では clap の説明のまま
        crate::lang::set(Lang::En);
        assert_eq!(parse().to_string().is_ascii(), true, "{}", parse());
    }

    /// ヘルプと版の表示は誤りではないので、日本語にせず clap に任せる
    #[rstest]
    #[case::help("--help")]
    #[case::version("--version")]
    fn help_and_version_are_not_errors(#[case] flag: &str) {
        let e = parse_from(["meikey", flag]).err().unwrap();
        assert_eq!(japanese_error(&e), None);
    }

    /// 範囲の整数: 範囲内の整数だけ受け付け、それ以外は範囲を付けた ValueError::NotInRange
    #[rstest]
    #[case::low("1", Some(1))]
    #[case::high("64", Some(64))]
    #[case::below("0", None)]
    #[case::above("65", None)]
    #[case::negative("-1", None)]
    #[case::not_a_number("x", None)]
    #[case::fraction("1.5", None)]
    fn int_in_range(#[case] s: &str, #[case] want: Option<u32>) {
        let want = want.ok_or(ValueError::NotInRange { lo: 1, hi: 64 });
        assert_eq!(int_in(s, 1, 64), want);
    }

    /// 秒数: 0 より大きい数だけ受け付ける。数でなければ NotSeconds、0 以下や大きすぎれば NotPositive
    #[rstest]
    #[case::integer("10", Ok(Duration::from_secs(10)))]
    #[case::fraction("0.5", Ok(Duration::from_millis(500)))]
    #[case::zero("0", Err(ValueError::NotPositive))]
    #[case::negative("-1", Err(ValueError::NotPositive))]
    #[case::too_large("1e30", Err(ValueError::NotPositive))]
    #[case::not_a_number("x", Err(ValueError::NotSeconds))]
    #[case::empty("", Err(ValueError::NotSeconds))]
    fn parse_secs_cases(#[case] s: &str, #[case] want: Result<Duration, ValueError>) {
        assert_eq!(parse_secs(s), want);
    }

    /// 英語では、表示も誤りも英語になる
    #[rstest]
    fn messages_in_english(mut searcher: MockSearcher, mut repo: MockKeyRepository) {
        crate::lang::set(Lang::En);
        let cli = parse(&["abc"]);
        repo.expect_existing()
            .returning(|_| Ok(Some(PathBuf::from("id_ed25519_meikey"))));
        searcher.expect_run().never();
        let e = run(&cli, &searcher, &repo).unwrap_err().to_string();
        assert_eq!(
            e,
            "already exists: id_ed25519_meikey (--force to overwrite)"
        );

        let e = run(&parse(&["ab-c"]), &searcher, &repo)
            .unwrap_err()
            .to_string();
        assert_eq!(e, "invalid character: '-' (allowed: A-Z a-z 0-9 + /)");
        assert_eq!(
            parse_secs("0").unwrap_err().to_string(),
            "must be greater than 0 seconds"
        );
        assert_eq!(per_sec(62_501_000.0), "62.5M keys/s");
        assert_eq!(about("3d 18h"), "about 3d 18h");
    }
}
