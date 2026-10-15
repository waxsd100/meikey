//! GPU（CUDA）で鍵を探す [`Searcher`]（feature `cuda`、`--gpu`）
//!
//! シードの生成から文字列の判定までを GPU のカーネル（`gpu/kernel.cu`）で行い、CPU は起動と結果の確認だけをする。
//!
//! - カーネルの設定（表の幅 W、逆元をまとめる鍵の数、体演算の種類。[`GpuOptions`]）のうち、鍵の数と体演算の
//!   組（[`Build`]）ごとに compute_75（Turing）向けの PTX（`gpu/ptx/`）を埋め込んでおく（[`BUILDS`]。表の幅は
//!   起動の引数）。起動時にドライバがその GPU 向けに変換する（変換結果はドライバがキャッシュするので2回目からは速い）。
//!   要るのは NVIDIA のドライバ（libcuda / nvcuda.dll）だけで、CUDA Toolkit は要らない。PTX は `cargo make ptx` で
//!   kernel.cu から作る
//! - 速い設定は GPU によって違うので、探す前にこの GPU で短く測り、一番速いものを使う（[`GpuSetting::Fastest`]）。
//!   まず幅 16・32 鍵で体演算の種類を比べ、次にその種類で幅と鍵の数を比べ、最後に 1 SM に載せるブロック数
//!   （PTX の `.minnctapersm` を書き換えて、1 スレッドのレジスタの上限を変える）を比べる（3 段で測る）。
//!   `--gpu-width`・`--gpu-batch`・`--gpu-limbs` で決めることもでき、埋め込んでいない組や、ドライバが古くて PTX を
//!   読めないときは、NVRTC（libnvrtc）があればその GPU 向けにコンパイルする
//! - 読み込んだカーネルは、まず少しの鍵を計算させて CPU（ed25519-dalek）と比べ（探索と同じ、ブロックの中で
//!   逆元をまとめる手順でも計算させる）、合わなければその設定は使わない
//!   （[`GpuError::SelfCheck`]。測って選ぶときは飛ばして、状態の表示に出す）
//! - CUDA のライブラリは実行時に読み込むので、ビルドに CUDA は要らない
//! - シードは `ChaCha20(鍵 = 起動ごとの乱数, ブロック番号 = 鍵の通し番号 / 2)` の 64 バイトの、通し番号が
//!   偶数なら前半・奇数なら後半の 32 バイト（1 ブロックで 2 本ぶん）か、低セキュリティモードでは乱数の最後の
//!   8 バイトに通し番号を足したもの。
//!   当たりは通し番号だけを受け取り、CPU で同じシードを作り直して ed25519-dalek で計算し、
//!   文字列が本当に含まれるかを確かめてから返す（CPU 版と同じく、GPU の計算にバグがあっても誤った鍵は返さない）。
//!   GPU は x の符号を計算しないので、符号だけが違って文字列を含まない鍵も知らせてくる。そのときは探し続ける
//! - 櫛形テーブル（comb.rs と同じ作り方、幅 W）は CPU で作り、各値を 32 ビット × 8 語に詰めるか、体演算のリムの
//!   形にして GPU に送る（1 エントリ 96 バイトか 128 バイト）
//! - 1回の起動（カーネル）が `TARGET_LAUNCH`（0.2 秒）前後で終わるように、1スレッドの繰り返し回数を測りながら調整する。
//!   画面をつないだ GPU（Windows のタイムアウト検出の対象）では、画面が固まらないよう `DISPLAY_LAUNCH`（0.05 秒）にする

use crate::lang::{t, tf};
use crate::model::key::{PUBLIC_B64_LEN, VanityKey, encode_public_b64};
use crate::model::pattern::{Mode, Pattern};
use crate::model::search::{Outcome, SearchConfig, SeedSource, Stats, counter_seed};
use crate::service::derive::comb::{for_each_entry_in_row, rows_cols};
use crate::service::derive::field::Fe;
use crate::service::search::{Progress, Searcher};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
    sys::CUdevice_attribute,
};
use cudarc::nvrtc::{CompileError, CompileOptions, compile_ptx_with_opts};
use rand::rngs::ChaCha20Rng;
use rand::{Rng, SeedableRng};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// カーネルのソース
const KERNEL: &str = include_str!("gpu/kernel.cu");

/// 埋め込む PTX の世代。これより新しい GPU でもドライバが変換して動かせる（Turing 〜 Blackwell）
pub const PTX_ARCH: &str = "compute_75";

/// 表の幅 W の既定（`--check` のように測らないとき）
pub const DEFAULT_WIDTH: u32 = 16;

/// 1スレッドが1回にまとめて計算する鍵の数（逆元をまとめる数）の既定（`--check` のように測らないとき）
pub const DEFAULT_BATCH: u32 = 32;

/// 体演算の種類の既定（`--check` のように測らないとき）
pub const DEFAULT_LIMBS: Limbs = Limbs::R25;

/// 選べる表の幅 W
pub const WIDTHS: std::ops::RangeInclusive<u32> = 2..=20;

/// 選べる、逆元をまとめる鍵の数（kernel.cu の MEIKEY_BATCH の範囲）
pub const BATCHES: std::ops::RangeInclusive<u32> = 1..=64;

/// 体演算の種類（kernel.cu の MEIKEY_FIELD と MEIKEY_UNPACKED。`--gpu-limbs` の値は [`Limbs::code`]）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Limbs {
    /// 25.5 ビット × 10 リム（64 ビットの積和）
    R25,
    /// R25 で、表をリムの形のまま持つ
    R25Table,
    /// 32 ビット × 8 リム（桁上がりつきの命令）
    R32,
    /// 26 ビット × 10 リムで、積を Karatsuba 法で
    K26,
    /// K26 で、表をリムの形のまま持つ
    K26Table,
    /// 51 ビット × 5 リムで、積を倍精度浮動小数点数の FMA で（倍精度の速い GPU 向け）
    F51,
}

impl Limbs {
    pub const ALL: [Limbs; 6] = [
        Limbs::R25,
        Limbs::R25Table,
        Limbs::R32,
        Limbs::K26,
        Limbs::K26Table,
        Limbs::F51,
    ];

    /// `--gpu-limbs` の値・PTX のファイル名に使う名前
    pub fn code(self) -> &'static str {
        match self {
            Limbs::R25 => "10",
            Limbs::R25Table => "10u",
            Limbs::R32 => "8",
            Limbs::K26 => "10k",
            Limbs::K26Table => "10ku",
            Limbs::F51 => "5f",
        }
    }

    /// kernel.cu の MEIKEY_FIELD
    fn field(self) -> u32 {
        match self {
            Limbs::R25 | Limbs::R25Table => 10,
            Limbs::R32 => 8,
            Limbs::K26 | Limbs::K26Table => 26,
            Limbs::F51 => 51,
        }
    }

    /// 表の形
    fn layout(self) -> Layout {
        match self {
            Limbs::R25Table => Layout::R25,
            Limbs::K26Table => Layout::K26,
            _ => Layout::Packed,
        }
    }

    /// 倍精度浮動小数点数の速い GPU（A100・H100 など）でだけ測るか
    fn needs_fp64(self) -> bool {
        self == Limbs::F51
    }
}

impl fmt::Display for Limbs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Limbs::R25 => t!("10 リム", "10 limbs"),
            Limbs::R25Table => t!("10 リム・展開した表", "10 limbs, unpacked table"),
            Limbs::R32 => t!("8 リム", "8 limbs"),
            Limbs::K26 => t!("10 リム・Karatsuba", "10 limbs, Karatsuba"),
            Limbs::K26Table => t!(
                "10 リム・Karatsuba・展開した表",
                "10 limbs, Karatsuba, unpacked table"
            ),
            Limbs::F51 => t!("5 リム・FP64", "5 limbs, FP64"),
        })
    }
}

impl FromStr for Limbs {
    type Err = String;

    fn from_str(s: &str) -> Result<Limbs, String> {
        Limbs::ALL
            .into_iter()
            .find(|l| l.code() == s)
            .ok_or_else(|| {
                let codes: Vec<&str> = Limbs::ALL.iter().map(|l| l.code()).collect();
                tf!(
                    "{} のどれかが必要",
                    "must be one of {}",
                    codes.join(t!("・", ", "))
                )
            })
    }
}

/// 表の 1 エントリの形（kernel.cu の ENTRY_WORDS）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// y+x, y-x, 2dxy を 32 ビット × 8 語ずつ（96 バイト）
    Packed,
    /// 25.5 ビット × 10 リムずつと余白 2 語（128 バイト）
    R25,
    /// 26 ビット × 10 リムずつと余白 2 語（128 バイト）
    K26,
}

impl Layout {
    fn entry_words(self) -> usize {
        match self {
            Layout::Packed => 24,
            Layout::R25 | Layout::K26 => 32,
        }
    }
}

/// GPU のカーネルの設定
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuOptions {
    /// 表の幅 W（大きいほど点の足し算が減り、表は大きくなる。W = 12 で約 4.1 MB、16 で約 50 MB、20 で約 650 MB）
    pub width: u32,
    /// 1スレッドが1回にまとめて計算する鍵の数（大きいほど逆元の分担が減り、1スレッドが使うメモリは増える）
    pub batch: u32,
    /// 体演算の種類（どれが速いかは GPU で違う）
    pub limbs: Limbs,
    /// 1 SM に同時に載せたいブロック数（カーネルの .minnctapersm。多いほど 1 スレッドが使えるレジスタは減る）
    pub blocks: u32,
}

/// 1 SM に同時に載せたいブロック数の既定（kernel.cu の __launch_bounds__ と同じ）
pub const DEFAULT_BLOCKS: u32 = 4;

/// 選べる、1 SM に同時に載せたいブロック数
pub const BLOCKS_PER_SM: std::ops::RangeInclusive<u32> = 1..=8;

impl Default for GpuOptions {
    fn default() -> Self {
        GpuOptions::new(DEFAULT_WIDTH, DEFAULT_BATCH, DEFAULT_LIMBS)
    }
}

impl GpuOptions {
    pub const fn new(width: u32, batch: u32, limbs: Limbs) -> Self {
        GpuOptions {
            width,
            batch,
            limbs,
            blocks: DEFAULT_BLOCKS,
        }
    }

    /// 1 SM に同時に載せたいブロック数を変えたもの
    pub const fn with_blocks(self, blocks: u32) -> Self {
        GpuOptions { blocks, ..self }
    }

    /// kernel.cu が受け付ける設定か
    fn in_range(&self) -> bool {
        WIDTHS.contains(&self.width)
            && BATCHES.contains(&self.batch)
            && BLOCKS_PER_SM.contains(&self.blocks)
    }

    /// 表の大きさ（バイト）
    pub fn table_bytes(&self) -> usize {
        let (rows, cols) = rows_cols(self.width as usize);
        rows * cols * self.limbs.layout().entry_words() * 4
    }

    /// コンパイルする単位（鍵の数と体演算の種類）
    pub fn build(&self) -> Build {
        Build {
            batch: self.batch,
            limbs: self.limbs,
        }
    }
}

impl fmt::Display for GpuOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&tf!(
            "W = {}・{} 鍵・{}",
            "W = {}, {} keys, {}",
            self.width,
            self.batch,
            self.limbs
        ))?;
        if self.blocks != DEFAULT_BLOCKS {
            f.write_str(&tf!(
                "・SM に {} ブロック",
                ", {} blocks per SM",
                self.blocks
            ))?;
        }
        Ok(())
    }
}

/// カーネルをコンパイルする単位（[`GpuOptions`] のうち、表の幅を除いたもの）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Build {
    pub batch: u32,
    pub limbs: Limbs,
}

impl Build {
    /// 埋め込む PTX のファイル名（`gpu/ptx/` の中）
    pub fn ptx_file(&self) -> String {
        format!("b{}_l{}.ptx", self.batch, self.limbs.code())
    }

    /// NVRTC に渡すマクロ
    fn defines(&self) -> Vec<String> {
        vec![
            format!("-DMEIKEY_BATCH={}", self.batch),
            format!("-DMEIKEY_FIELD={}", self.limbs.field()),
            format!(
                "-DMEIKEY_UNPACKED={}",
                u32::from(self.limbs.layout() != Layout::Packed)
            ),
        ]
    }
}

impl fmt::Display for Build {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&tf!("{} 鍵・{}", "{} keys, {}", self.batch, self.limbs))
    }
}

/// 測るときの、まとめる鍵の数
pub const MEASURE_BATCHES: [u32; 2] = [32, 16];

/// 測るときの表の幅
pub const MEASURE_WIDTHS: [u32; 5] = [12, 14, 16, 18, 20];

/// PTX を埋め込むもの（鍵の数 × 体演算の種類）。先頭は既定の設定（build.rs の VARIANTS も同じ並び）
pub const BUILDS: [Build; 12] = {
    let mut out = [Build {
        batch: 0,
        limbs: Limbs::R25,
    }; 12];
    let mut i = 0;
    while i < 12 {
        out[i] = Build {
            batch: MEASURE_BATCHES[i / 6],
            limbs: Limbs::ALL[i % 6],
        };
        i += 1;
    }
    out
};

/// [`BUILDS`] の kernel.cu を compute_75 向けにコンパイルした PTX（`cargo make ptx` で作る）。
/// 1行目に kernel.cu と設定のハッシュ（[`kernel_hash`]）を書いておき、ソースと食い違っていないかをテストで確かめる
const PTXS: [&str; 12] = [
    include_str!("gpu/ptx/b32_l10.ptx"),
    include_str!("gpu/ptx/b32_l10u.ptx"),
    include_str!("gpu/ptx/b32_l8.ptx"),
    include_str!("gpu/ptx/b32_l10k.ptx"),
    include_str!("gpu/ptx/b32_l10ku.ptx"),
    include_str!("gpu/ptx/b32_l5f.ptx"),
    include_str!("gpu/ptx/b16_l10.ptx"),
    include_str!("gpu/ptx/b16_l10u.ptx"),
    include_str!("gpu/ptx/b16_l8.ptx"),
    include_str!("gpu/ptx/b16_l10k.ptx"),
    include_str!("gpu/ptx/b16_l10ku.ptx"),
    include_str!("gpu/ptx/b16_l5f.ptx"),
];

/// PTX の各カーネルの「1 SM に同時に載せたいブロック数」（`.minnctapersm`。kernel.cu の __launch_bounds__ の 2 つ目）
/// を `blocks` に書き換える（ドライバが PTX を変換するときの、1 スレッドのレジスタの上限が変わる）
fn with_min_blocks(ptx: &str, blocks: u32) -> String {
    let (from, to) = (
        format!(".minnctapersm {DEFAULT_BLOCKS}"),
        format!(".minnctapersm {blocks}"),
    );
    // 行ごとに比べる（Windows で改行が CRLF になっていても同じ）
    ptx.split_inclusive('\n')
        .map(|line| match line.trim_end() == from {
            true => line.replacen(&from, &to, 1),
            false => line.to_string(),
        })
        .collect()
}

/// `build` の埋め込んだ PTX（埋め込んでいなければ None）
fn embedded_ptx(build: Build) -> Option<&'static str> {
    BUILDS.iter().position(|&b| b == build).map(|i| PTXS[i])
}

/// 1 段目に測る設定: 幅 16・32 鍵で、体演算の種類を全部（倍精度の速い GPU でなければ FP64 は除く）
pub fn first_stage(fp64_fast: bool) -> Vec<GpuOptions> {
    Limbs::ALL
        .into_iter()
        .filter(|l| fp64_fast || !l.needs_fp64())
        .map(|l| GpuOptions::new(DEFAULT_WIDTH, DEFAULT_BATCH, l))
        .collect()
}

/// 2 段目に測る設定: 1 段目で一番速かった体演算の種類で、幅と鍵の数を全部（ブロック数は既定）
pub fn second_stage(limbs: Limbs) -> Vec<GpuOptions> {
    MEASURE_BATCHES
        .into_iter()
        .flat_map(|b| MEASURE_WIDTHS.map(|w| GpuOptions::new(w, b, limbs)))
        .collect()
}

/// 3 段目に測る設定: 2 段目で一番速かった設定で、1 SM に載せたいブロック数を 4・3・2 に
pub fn third_stage(best: GpuOptions) -> Vec<GpuOptions> {
    MEASURE_BLOCKS.map(|b| best.with_blocks(b)).to_vec()
}

/// 3 段目に測る、1 SM に載せたいブロック数（先頭は既定）
pub const MEASURE_BLOCKS: [u32; 3] = [DEFAULT_BLOCKS, 3, 2];

/// 測る設定の数（3 段の合計。表示用）
pub fn measured_count(fp64_fast: bool) -> usize {
    first_stage(fp64_fast).len()
        + MEASURE_BATCHES.len() * MEASURE_WIDTHS.len()
        + MEASURE_BLOCKS.len()
}

/// カーネルの設定の決め方
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSetting {
    /// この GPU で測り、一番速いものを使う（3 段で測る）
    Fastest,
    /// この設定を使う（測らない）
    Fixed(GpuOptions),
    /// 測らずに既定の設定を使う（`--check` のように、探さないとき）
    Default,
}

/// 設定をどう決めたか（表示用）
#[derive(Debug, Clone)]
enum Chosen {
    /// 測って一番速かった。測った設定と速度（鍵/秒）と、計算が CPU と合わなかったので使わなかった設定
    Fastest {
        results: Vec<(GpuOptions, f64)>,
        rejected: Vec<GpuOptions>,
    },
    /// 同じ GPU で測った設定を使った（複数の GPU のうち 2 枚目から）
    Same,
    Fixed,
    Default,
}

/// 1ブロックのスレッド数
const BLOCK: u32 = 128;

/// 読み込んだカーネルを確かめるときのスレッド数（[`Kernel::self_check`]）
const SELF_CHECK_THREADS: u32 = 32;

/// 1回の起動にかける時間の目安（長いほど起動の手間は減るが、当たりや時間切れに気づくのが遅れる）
const TARGET_LAUNCH: Duration = Duration::from_millis(200);

/// 画面をつないだ GPU での目安。Windows は GPU が約 2 秒応答しないとドライバをリセットし、
/// 起動が長いと画面もその間固まるので短くする
const DISPLAY_LAUNCH: Duration = Duration::from_millis(50);

/// 設定を測るときの、1回の起動の長さと回数
const MEASURE_LAUNCH: Duration = Duration::from_millis(100);
const MEASURE_LAUNCHES: u32 = 3;

/// 設定を測るときに探す文字列（判定の分も測るため。当たっても止まらない）
const MEASURE_NEEDLE: &[u8] = b"meikey";

/// 倍精度浮動小数点数の速い GPU とみなす、単精度との速さの比（A100・H100 は 2。一般向けの GPU は 32〜64）
const FP64_FAST_RATIO: i32 = 4;

/// GPU を使えないときのエラー
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// CUDA のドライバ（libcuda・nvcuda.dll）が無い
    #[error("{}", t!(
        "CUDA のドライバが見つからない（NVIDIA の GPU とドライバが必要。libcuda / nvcuda.dll）",
        "CUDA driver not found (an NVIDIA GPU and its driver are needed; libcuda / nvcuda.dll)"
    ))]
    DriverNotFound,
    /// GPU が1枚も無い
    #[error("{}", t!("CUDA の GPU が無い", "no CUDA GPU found"))]
    NoDevice,
    /// NVRTC（libnvrtc）が無い（埋め込む PTX を作るとき）
    #[error("{}", t!("libnvrtc が見つからない", "libnvrtc not found"))]
    NvrtcNotFound,
    /// 表の幅・まとめる鍵の数が範囲の外
    #[error("{}", tf!(
        "表の幅は {}〜{}、まとめる鍵の数は {}〜{}（{}）",
        "the table width must be {}-{} and the keys per batch {}-{} ({})",
        WIDTHS.start(),
        WIDTHS.end(),
        BATCHES.start(),
        BATCHES.end(),
        .0
    ))]
    OutOfRange(GpuOptions),
    /// 読み込んだカーネルで計算した公開鍵が、CPU（ed25519-dalek）で計算したものと合わない
    #[error("{}", tf!(
        "{} の GPU の計算が CPU と合わない（この設定は使えない）",
        "the GPU results of {} differ from the CPU (this setting cannot be used)",
        .0
    ))]
    SelfCheck(GpuOptions),
    /// 埋め込んだ PTX をドライバが読めず（ドライバが古い）、代わりにコンパイルする NVRTC も無い
    #[error("{}", tf!(
        "PTX を読めない（{}。ドライバを更新するか、NVRTC を入れて LD_LIBRARY_PATH / PATH に追加）",
        "cannot load the PTX ({}; update the driver, or install NVRTC and add it to LD_LIBRARY_PATH / PATH)",
        .0
    ))]
    PtxNotLoadable(#[source] DriverError),
    /// 埋め込んでいない設定なのに、コンパイルする NVRTC が無い
    #[error("{}", tf!(
        "{} は埋め込んでいない設定なので NVRTC（libnvrtc）が要る（LD_LIBRARY_PATH / PATH に追加）",
        "{} is not built in, so NVRTC (libnvrtc) is needed (add it to LD_LIBRARY_PATH / PATH)",
        .0
    ))]
    NotEmbedded(Build),
    /// 測る設定が1つも無い
    #[error("{}", t!("測る設定が無い", "no settings to measure"))]
    NoSettings,
    /// カーネルをコンパイルできない（世代ごとの NVRTC のログ）
    #[error("{}", tf!("カーネルのコンパイルに失敗:\n{}", "failed to compile the kernel:\n{}", .0))]
    Compile(String),
    /// CUDA の操作に失敗した
    #[error("CUDA: {0}")]
    Driver(#[from] DriverError),
}

/// GPU の情報
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    /// compute capability（世代）
    pub compute: (i32, i32),
    /// SM（ストリーミングマルチプロセッサ）の数
    pub sms: u32,
    /// メモリ（バイト）
    pub memory: usize,
    /// 倍精度浮動小数点数が速いか（FP64 の体演算も測る）
    pub fp64_fast: bool,
}

impl fmt::Display for GpuInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&tf!(
            "{}（sm_{}{}・SM {}・{:.0} GB）",
            "{} (sm_{}{}, {} SMs, {:.0} GB)",
            self.name,
            self.compute.0,
            self.compute.1,
            self.sms,
            self.memory as f64 / 1e9
        ))
    }
}

/// 1つの設定で読み込んだカーネルと、GPU に送った表
struct Kernel {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    /// 公開鍵を計算して書き出すだけのカーネル（読み込んだときの確認と、テスト用）
    derive: CudaFunction,
    /// 探索のカーネルと同じ手順（ブロックの中で逆元をまとめる）で y を書き出すカーネル（読み込んだときの確認用）
    pooled: CudaFunction,
    /// 表（幅と形が同じ設定で共有する）
    table: Arc<CudaSlice<u32>>,
    options: GpuOptions,
    /// 起動するブロック数
    grid: u32,
}

/// GPU に送った表と、その幅と形
type SentTable = ((u32, Layout), Arc<CudaSlice<u32>>);

/// 読み込んだモジュールと GPU に送った表（測るときに、同じものを何度も作らない）
#[derive(Default)]
struct Loaded {
    /// ((コンパイルの単位, 1 SM に載せたいブロック数), [探索のカーネル, 公開鍵を書き出すカーネル,
    /// 同じ手順で y を書き出すカーネル])
    modules: Vec<((Build, u32), [CudaFunction; 3])>,
    tables: Vec<SentTable>,
}

impl Loaded {
    /// `build` のカーネルを、1 SM に `blocks` ブロック載せる前提で（まだ読み込んでいなければ読み込む）。
    /// 埋め込んだ PTX があればそれを、無いかドライバが読めなければ NVRTC でこの GPU 向けにコンパイルしたものを使う
    fn module(
        &mut self,
        ctx: &Arc<CudaContext>,
        info: &GpuInfo,
        build: Build,
        blocks: u32,
    ) -> Result<[CudaFunction; 3], GpuError> {
        if let Some((_, f)) = self.modules.iter().find(|(k, _)| *k == (build, blocks)) {
            return Ok(f.clone());
        }
        let load =
            |ptx: &str| ctx.load_module(cudarc::nvrtc::Ptx::from_src(with_min_blocks(ptx, blocks)));
        let embedded = embedded_ptx(build).map(load);
        let module = match embedded {
            Some(Ok(m)) => m,
            // 埋め込んでいない組か、ドライバが古くて埋め込んだ PTX を読めない: NVRTC でコンパイルする
            other => {
                // SAFETY: ライブラリを読み込めるか試すだけ
                if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
                    return Err(match other {
                        Some(Err(e)) => GpuError::PtxNotLoadable(e),
                        _ => GpuError::NotEmbedded(build),
                    });
                }
                load(&compile(info.compute, build)?.to_src())?
            }
        };
        let functions = [
            module.load_function("meikey_search")?,
            module.load_function("meikey_derive")?,
            module.load_function("meikey_pooled")?,
        ];
        self.modules.push(((build, blocks), functions.clone()));
        Ok(functions)
    }

    /// 幅 `width`・形 `layout` の表（まだ無ければ作って送る）
    fn table(
        &mut self,
        stream: &Arc<CudaStream>,
        width: u32,
        layout: Layout,
    ) -> Result<Arc<CudaSlice<u32>>, GpuError> {
        if let Some((_, t)) = self.tables.iter().find(|(k, _)| *k == (width, layout)) {
            return Ok(t.clone());
        }
        let table = Arc::new(stream.clone_htod(&table_words(width, layout))?);
        self.tables.push(((width, layout), table.clone()));
        Ok(table)
    }
}

/// kernel.cu の探し方（MODE_CONTAINS など。model::pattern::Mode の順）
const MODE_CONTAINS: u32 = 0;
const MODE_SUFFIX: u32 = 1;
const MODE_REGEX: u32 = 2;

/// GPU に渡す探す文字列（kernel.cu の make_matcher の needle・len・fold・mode）
#[derive(Debug, Clone, PartialEq, Eq)]
struct Needle {
    /// 文字列（大小文字を区別しないなら、GPU 側で base64 を小文字にして比べるので小文字に）か、
    /// 正規表現なら DFA の表（[`crate::model::pattern::dfa::Dfa::gpu_table`]）
    bytes: Vec<u8>,
    len: u32,
    fold: bool,
    mode: u32,
}

impl Needle {
    fn new(pattern: Option<&Pattern>) -> Needle {
        let Some(p) = pattern else {
            // ベンチマークで判定しないとき: 長さ 0 は一致しない
            return Needle {
                bytes: Vec::new(),
                len: 0,
                fold: false,
                mode: MODE_CONTAINS,
            };
        };
        if let Some(dfa) = p.dfa() {
            return Needle {
                bytes: dfa.gpu_table(),
                len: dfa.states() as u32,
                fold: false,
                mode: MODE_REGEX,
            };
        }
        let bytes = match p.case_sensitive() {
            true => p.as_str().as_bytes().to_vec(),
            false => p.as_str().to_ascii_lowercase().into_bytes(),
        };
        Needle {
            len: bytes.len() as u32,
            bytes,
            fold: !p.case_sensitive(),
            mode: match p.mode() {
                Mode::Suffix => MODE_SUFFIX,
                _ => MODE_CONTAINS,
            },
        }
    }
}

/// 探索の1回の起動に渡すもの（カーネルの引数のうち、起動ごとに変わらないもの）
struct Launch {
    key: CudaSlice<u32>,
    /// シードを通し番号で作るか（低セキュリティモード）
    counter: u32,
    needle: CudaSlice<u8>,
    needle_len: u32,
    fold: u32,
    mode: u32,
    stop_on_hit: u32,
    result: CudaSlice<u64>,
    sink: CudaSlice<u32>,
}

impl Kernel {
    /// 設定 `options` のカーネルを読み込み、表を送る（`loaded` に同じものがあればそれを使う）
    fn load(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        info: &GpuInfo,
        options: GpuOptions,
        loaded: &mut Loaded,
    ) -> Result<Kernel, GpuError> {
        if !options.in_range() {
            return Err(GpuError::OutOfRange(options));
        }
        let [func, derive, pooled] = loaded.module(ctx, info, options.build(), options.blocks)?;
        // SM あたりに同時に載るブロック数（レジスタと共有メモリの使用量で決まる）の分だけ起動する
        let per_sm = func
            .occupancy_max_active_blocks_per_multiprocessor(BLOCK, 0, None)
            .unwrap_or(2)
            .max(1);
        let table = loaded.table(stream, options.width, options.limbs.layout())?;
        let kernel = Kernel {
            stream: stream.clone(),
            func,
            derive,
            pooled,
            table,
            options,
            grid: info.sms.max(1) * per_sm,
        };
        kernel.self_check()?;
        Ok(kernel)
    }

    /// 1 ブロック分（[`SELF_CHECK_THREADS`] スレッド）の公開鍵を GPU で計算し、そのうち先頭と最後のスレッドの全部と、
    /// ほかのスレッドの1本目を CPU（ed25519-dalek）で計算し直して比べる。シードの作り方の両方で確かめ、
    /// 合わなければ使えない設定（通し番号は奇数から始めて、シードを ChaCha20 のブロックの後半から取る場合も通す）。
    /// 探索のカーネルと同じ手順（ブロックの中で逆元をまとめる）の y も、[`BLOCK`] スレッドの各スレッドの最初と
    /// 最後の鍵を比べる
    fn self_check(&self) -> Result<(), GpuError> {
        let batch = self.options.batch as usize;
        let keys = SELF_CHECK_THREADS as usize * batch;
        let fail = || Err(GpuError::SelfCheck(self.options));
        for source in [SeedSource::ChaCha20, SeedSource::Counter] {
            let key: [u8; 32] = rand::random();
            let first = (rand::random::<u64>() >> 8) | 1;
            let got = self.derive_on_gpu(&key, source, first, 1, SELF_CHECK_THREADS)?;
            let checked = (0..keys)
                .filter(|i| i % batch == 0 || i / batch == 0 || i / batch == keys / batch - 1);
            for i in checked {
                let want = VanityKey::from_seed(seed(&key, source, first + i as u64));
                if &got.as_chunks::<32>().0[i] != want.public_bytes() {
                    return fail();
                }
            }
        }
        let key: [u8; 32] = rand::random();
        let first = rand::random::<u64>() >> 8;
        let got = self.pooled_on_gpu(&key, first)?;
        for t in 0..BLOCK as usize {
            for j in [0, batch - 1] {
                let i = t * batch + j;
                let mut want =
                    *VanityKey::from_seed(counter_seed(&key, first + i as u64)).public_bytes();
                want[31] &= 0x7f;
                if got.as_chunks::<32>().0[i] != want {
                    return fail();
                }
            }
        }
        Ok(())
    }

    /// meikey_pooled で、通し番号 `first` から 1 ブロック（[`BLOCK`] スレッド）分の公開鍵の y を計算する
    /// （低セキュリティモードのシード）
    fn pooled_on_gpu(&self, key: &[u8; 32], first: u64) -> Result<Vec<u8>, GpuError> {
        let stream = &self.stream;
        let n = BLOCK as usize * self.options.batch as usize;
        let key_dev = stream.clone_htod(&key_words(key))?;
        let counter = 1u32;
        let mut out = stream.alloc_zeros::<u8>(n * 32)?;
        let mut b = stream.launch_builder(&self.pooled);
        b.arg(&*self.table)
            .arg(&self.options.width)
            .arg(&key_dev)
            .arg(&counter)
            .arg(&first)
            .arg(&mut out);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: 引数は kernel.cu の meikey_pooled と同じ並びで、out は 1 ブロックの鍵の分ある
        unsafe { b.launch(cfg)? };
        Ok(stream.clone_dtoh(&out)?)
    }

    /// meikey_derive で、通し番号 `first` から `blocks` × `threads` スレッド分の公開鍵を計算する
    fn derive_on_gpu(
        &self,
        key: &[u8; 32],
        source: SeedSource,
        first: u64,
        blocks: u32,
        threads: u32,
    ) -> Result<Vec<u8>, GpuError> {
        let stream = &self.stream;
        let n = (blocks * threads) as usize * self.options.batch as usize;
        let key_dev = stream.clone_htod(&key_words(key))?;
        let counter = u32::from(source == SeedSource::Counter);
        let mut out = stream.alloc_zeros::<u8>(n * 32)?;
        let mut b = stream.launch_builder(&self.derive);
        b.arg(&*self.table)
            .arg(&self.options.width)
            .arg(&key_dev)
            .arg(&counter)
            .arg(&first)
            .arg(&mut out);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: 引数は kernel.cu の meikey_derive と同じ並びで、out は全スレッドの鍵の分ある
        unsafe { b.launch(cfg)? };
        Ok(stream.clone_dtoh(&out)?)
    }
}

/// GPU で探すときに、カーネルを起動する部分（本物は [`Kernel`]。テストでは、CPU 向けにコンパイルしたカーネルを
/// 呼ぶものと、mockall のモックに差し替えて、探索の流れ（[`search_with`]）と測り方（[`measure`]）を GPU なしで確かめる）
#[cfg_attr(test, mockall::automock(type Launch = ();))]
trait Engine {
    /// 起動に渡すもの（カーネルの引数のうち、起動ごとに変わらないもの）
    type Launch;

    /// 1回の起動で計算する鍵の数（1スレッドの繰り返し1回あたり）
    fn keys_per_iter(&self) -> u64;

    /// 起動に渡すものを用意する。`key` はシードを作る乱数（ChaCha20 の鍵か、通し番号を足す 32 バイト）
    fn prepare(
        &self,
        key: &[u8; 32],
        source: SeedSource,
        needle: &Needle,
        stop_on_hit: bool,
    ) -> Result<Self::Launch, GpuError>;

    /// 通し番号 `base` から、1スレッドが `iters` 回ずつ計算する起動を1回して、終わるまで待つ。
    /// 当たり（x の符号を除いて文字列を含む鍵）があれば、その通し番号を返す
    fn launch(&self, l: &mut Self::Launch, base: u64, iters: u32) -> Result<Option<u64>, GpuError>;
}

impl Engine for Kernel {
    type Launch = Launch;

    fn keys_per_iter(&self) -> u64 {
        u64::from(self.grid) * u64::from(BLOCK) * u64::from(self.options.batch)
    }

    fn prepare(
        &self,
        key: &[u8; 32],
        source: SeedSource,
        needle: &Needle,
        stop_on_hit: bool,
    ) -> Result<Launch, GpuError> {
        let stream = &self.stream;
        Ok(Launch {
            key: stream.clone_htod(&key_words(key))?,
            counter: u32::from(source == SeedSource::Counter),
            needle: stream.clone_htod(&[&needle.bytes[..], &[0; 64]].concat())?,
            needle_len: needle.len,
            fold: u32::from(needle.fold),
            mode: needle.mode,
            stop_on_hit: u32::from(stop_on_hit),
            result: stream.alloc_zeros::<u64>(2)?,
            sink: stream.alloc_zeros::<u32>((self.grid * BLOCK) as usize)?,
        })
    }

    fn launch(&self, l: &mut Launch, base: u64, iters: u32) -> Result<Option<u64>, GpuError> {
        let stream = &self.stream;
        let mut b = stream.launch_builder(&self.func);
        b.arg(&*self.table)
            .arg(&self.options.width)
            .arg(&l.key)
            .arg(&l.counter)
            .arg(&l.needle)
            .arg(&l.needle_len)
            .arg(&l.fold)
            .arg(&l.mode)
            .arg(&base)
            .arg(&iters)
            .arg(&l.stop_on_hit)
            .arg(&mut l.result)
            .arg(&mut l.sink);
        let cfg = LaunchConfig {
            grid_dim: (self.grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: 引数の型と並びは kernel.cu の meikey_search と同じ。sink はスレッド数ぶん、
        // result は 2 語、needle は文字列 + 余白 64 バイト確保してある
        unsafe { b.launch(cfg)? };
        let found = stream.clone_dtoh(&l.result)?; // 終わるまで待つ
        if found[0] != 0 {
            // 次の起動のために戻す（当たりが符号違いで、探し続けるとき）
            stream.memset_zeros(&mut l.result)?;
        }
        Ok((found[0] != 0).then_some(found[1]))
    }
}

/// `engine` での速度（鍵/秒）。1回 [`MEASURE_LAUNCH`] ほどの起動を [`MEASURE_LAUNCHES`] 回して測る
/// （最初の1回は、繰り返し回数を決めるための短い起動）
fn measure(engine: &impl Engine) -> Result<f64, GpuError> {
    // GPU で探すのは低セキュリティモード（通し番号のシード）だけなので、それで測る
    let needle = Needle {
        bytes: MEASURE_NEEDLE.to_vec(),
        len: MEASURE_NEEDLE.len() as u32,
        fold: true,
        mode: MODE_CONTAINS,
    };
    let mut l = engine.prepare(&[0x5a; 32], SeedSource::Counter, &needle, false)?;
    let start = Instant::now();
    engine.launch(&mut l, 0, 1)?;
    let once = start.elapsed().as_secs_f64().max(1e-6);
    let iters = (MEASURE_LAUNCH.as_secs_f64() / once).clamp(1.0, f64::from(1u32 << 16)) as u32;
    let start = Instant::now();
    for i in 0..MEASURE_LAUNCHES {
        let base = u64::from(i + 1) * engine.keys_per_iter() * u64::from(iters);
        engine.launch(&mut l, base, iters)?;
    }
    let keys = engine.keys_per_iter() * u64::from(iters) * u64::from(MEASURE_LAUNCHES);
    Ok(keys as f64 / start.elapsed().as_secs_f64().max(1e-9))
}

/// `engine` で、`config` の文字列を含む鍵を探す。1回の起動が `target` 前後で終わるよう、1スレッドの繰り返し回数を
/// 調整しながら起動し続け、当たったら（ベンチマークでなければ）その鍵を CPU で作り直して確かめてから返す
/// （x の符号だけが違って含まない鍵なら、探し続ける）。
/// 止める合図か時間切れで、鍵なしで返す。`name` は確かめて合わなかったときに出す GPU の名前
fn search_with<E: Engine>(
    engine: &E,
    target: Duration,
    name: &str,
    config: &SearchConfig,
    on_progress: &mut dyn Progress,
) -> Result<Outcome, GpuError> {
    // シードを作る ChaCha20 の鍵（起動ごとに新しい乱数）
    let key: [u8; 32] = rand::random();
    let needle = Needle::new(config.pattern.as_ref());
    // ベンチマークでは当たっても止まらない
    let mut l = engine.prepare(&key, config.seeds, &needle, !config.bench)?;

    let start = Instant::now();
    let mut tries = 0u64;
    let mut iters = 1u32;
    let mut next_report = start + config.progress_interval;
    loop {
        let launch = Instant::now();
        let found = engine.launch(&mut l, tries, iters)?;
        let took = launch.elapsed();
        tries += engine.keys_per_iter() * u64::from(iters);
        let stats = Stats {
            tries,
            elapsed: start.elapsed(),
        };

        if let Some(index) = found
            && !config.bench
            && let Some(key) = confirm(name, &key, index, config)
        {
            return Ok(Outcome {
                key: Some(key),
                stats,
            });
        }
        if config.cancel.is_cancelled()
            || config
                .time_limit
                .is_some_and(|limit| stats.elapsed >= limit)
        {
            return Ok(Outcome { key: None, stats });
        }
        if Instant::now() >= next_report {
            on_progress.report(stats);
            next_report += config.progress_interval;
        }
        // 1回の起動が target 前後になるよう、繰り返し回数を調整する（急に増やしすぎない）
        let scale = target.as_secs_f64() / took.as_secs_f64().max(1e-6);
        iters = ((f64::from(iters) * scale.clamp(0.5, 4.0)) as u32).clamp(1, 1 << 16);
    }
}

/// 通し番号 `index` の鍵を CPU で作り直し、文字列が含まれればその鍵を返す。GPU は x の符号を計算しないので、
/// 符号だけが違えば含む鍵も知らせてくる。そのときは None（探し続ける）。どちらでもなければ GPU `name` の計算か
/// 判定が間違っているので止める
fn confirm(name: &str, key: &[u8; 32], index: u64, config: &SearchConfig) -> Option<VanityKey> {
    let found = VanityKey::from_seed(seed(key, config.seeds, index));
    let Some(p) = &config.pattern else {
        return Some(found);
    };
    if p.is_match(&found.public_b64()) {
        return Some(found);
    }
    let mut flipped = *found.public_bytes();
    flipped[31] ^= 0x80;
    let mut text = [0u8; PUBLIC_B64_LEN];
    encode_public_b64(&flipped, &mut text);
    // 一致しなければ GPU の計算か判定が間違っている（バグ）ので止める
    assert!(
        p.is_match(&text),
        "{}",
        tf!(
            "内部エラー: GPU（{}）の結果が ed25519-dalek・総当たりと不一致",
            "internal error: the result of the GPU ({}) differs from ed25519-dalek with a naive search",
            name
        )
    );
    None
}

/// [`search_with`] で探し、GPU の操作に失敗したら止める（[`Searcher::run`] の中身）
fn run_with<E: Engine>(
    engine: &E,
    target: Duration,
    name: &str,
    config: &SearchConfig,
    on_progress: &mut dyn Progress,
) -> Outcome {
    search_with(engine, target, name, config, on_progress).unwrap_or_else(|e| {
        panic!(
            "{}",
            tf!("GPU での探索に失敗: {e}", "GPU search failed: {e}")
        )
    })
}

/// 設定ごとに測る周の数（前から1周、後ろから1周）。測っている間に GPU のクロックが上がっていっても、
/// 前の周で先に測ったものが後の周では後になるので、順番による有利・不利が打ち消し合う
const MEASURE_ROUNDS: usize = 2;

/// 設定 `candidates` を全部 `load` で読み込んでから `measure` で測り、一番速いものを返す（ほかは捨てる）。
/// 測る前に最初の設定を1回測って捨て（GPU を温める）、[`MEASURE_ROUNDS`] 周の平均で比べる。
/// 読み込めない・測れない設定は飛ばし、全部だめなら最初のエラーを返す
fn fastest<K>(
    candidates: &[GpuOptions],
    mut load: impl FnMut(GpuOptions) -> Result<K, GpuError>,
    mut measure: impl FnMut(&K) -> Result<f64, GpuError>,
) -> Result<(K, Vec<(GpuOptions, f64)>), GpuError> {
    let mut first_error = None;
    // (設定, 読み込んだもの, 測った速度の合計。測れなければ None)
    let mut loaded = Vec::new();
    for &options in candidates {
        match load(options) {
            Ok(k) => loaded.push((options, k, Some(0.0))),
            Err(e) => {
                first_error.get_or_insert(e);
            }
        }
    }
    if let Some((_, k, _)) = loaded.first() {
        let _ = measure(k);
    }
    let n = loaded.len();
    for round in 0..MEASURE_ROUNDS {
        for step in 0..n {
            let i = if round % 2 == 0 { step } else { n - 1 - step };
            let (_, k, total) = &mut loaded[i];
            if total.is_none() {
                continue;
            }
            match measure(k) {
                Ok(speed) => *total = total.map(|t| t + speed),
                Err(e) => {
                    *total = None;
                    first_error.get_or_insert(e);
                }
            }
        }
    }
    let results: Vec<(GpuOptions, f64)> = loaded
        .iter()
        .filter_map(|(o, _, t)| t.map(|t| (*o, t / MEASURE_ROUNDS as f64)))
        .collect();
    let best = results
        .iter()
        .enumerate()
        .fold(None::<(usize, f64)>, |best, (i, &(_, speed))| match best {
            Some((_, s)) if s >= speed => best,
            _ => Some((i, speed)),
        });
    match best {
        Some((i, _)) => {
            let options = results[i].0;
            let k = loaded
                .into_iter()
                .find(|(o, _, _)| *o == options)
                .map(|(_, k, _)| k)
                .expect("測れた設定は読み込んである");
            Ok((k, results))
        }
        None => Err(first_error.unwrap_or(GpuError::NoSettings)),
    }
}

/// 段に分けて測る: `first` を [`fastest`] で測り、一番速かった設定の体演算の種類で `second` を測り、
/// その一番速い設定から `third` を測って、一番速いものを返す（測った結果は全部の段のもの）。
/// 2 段目・3 段目が全部だめなら、その前の段の一番速いものを使う
fn fastest_staged<K>(
    first: &[GpuOptions],
    second: impl FnOnce(Limbs) -> Vec<GpuOptions>,
    third: impl FnOnce(GpuOptions) -> Vec<GpuOptions>,
    mut load: impl FnMut(GpuOptions) -> Result<K, GpuError>,
    mut measure: impl FnMut(&K) -> Result<f64, GpuError>,
) -> Result<(K, Vec<(GpuOptions, f64)>), GpuError> {
    let best_of = |r: &[(GpuOptions, f64)]| {
        r.iter()
            .fold(None::<(GpuOptions, f64)>, |best, &(o, s)| match best {
                Some((_, b)) if b >= s => best,
                _ => Some((o, s)),
            })
            .expect("測れた設定がある")
            .0
    };
    let (mut k, mut results) = fastest(first, &mut load, &mut measure)?;
    let mut best = best_of(&results);
    if let Ok((k2, more)) = fastest(&second(best.limbs), &mut load, &mut measure) {
        best = best_of(&more);
        k = k2;
        results.extend(more);
    }
    if let Ok((k3, more)) = fastest(&third(best), &mut load, &mut measure) {
        k = k3;
        results.extend(more);
    }
    Ok((k, results))
}

/// GPU で探す [`Searcher`]
pub struct Gpu {
    kernel: Kernel,
    info: GpuInfo,
    chosen: Chosen,
    /// 1回の起動にかける時間の目安
    target: Duration,
    /// 複数の GPU で探すときの番号（表示用）
    ordinal: Option<usize>,
}

/// 使える GPU の数（ドライバが無ければ 0）
pub fn device_count() -> usize {
    // SAFETY: ライブラリを読み込めるか試すだけ
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return 0;
    }
    CudaContext::device_count().map_or(0, |n| n.max(0) as usize)
}

/// `ordinal` 番目の GPU の情報
fn device_info(ctx: &Arc<CudaContext>) -> Result<GpuInfo, GpuError> {
    let ratio = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO)
        .unwrap_or(64);
    Ok(GpuInfo {
        name: ctx.name()?,
        compute: ctx.compute_capability()?,
        sms: ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32,
        memory: ctx.total_mem()?,
        fp64_fast: ratio <= FP64_FAST_RATIO,
    })
}

impl Gpu {
    /// `ordinal` 番目の GPU の情報（カーネルは読み込まない）
    pub fn info_of(ordinal: usize) -> Result<GpuInfo, GpuError> {
        // SAFETY: ライブラリを読み込めるか試すだけ
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(GpuError::DriverNotFound);
        }
        device_info(&CudaContext::new(ordinal)?)
    }

    /// `ordinal` 番目の GPU を使えるようにする（カーネルの読み込みと表の転送をする。
    /// [`GpuSetting::Fastest`] なら、設定を測ってから一番速いものを使い、測り始める前に、測る設定の数を渡して
    /// `on_measure` を呼ぶ）
    pub fn new(
        ordinal: usize,
        setting: GpuSetting,
        on_measure: impl FnOnce(usize),
    ) -> Result<Gpu, GpuError> {
        // SAFETY: ライブラリを読み込めるか試すだけ
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(GpuError::DriverNotFound);
        }
        let ctx = CudaContext::new(ordinal)?;
        let info = device_info(&ctx)?;
        // 画面をつないだ GPU（カーネルの実行時間に制限がある）なら起動を短くする
        let target =
            match ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT) {
                Ok(1) => DISPLAY_LAUNCH,
                _ => TARGET_LAUNCH,
            };
        let stream = ctx.default_stream();
        let mut loaded = Loaded::default();
        let (kernel, chosen) = match setting {
            GpuSetting::Fixed(options) => (
                Kernel::load(&ctx, &stream, &info, options, &mut loaded)?,
                Chosen::Fixed,
            ),
            GpuSetting::Default => (
                Kernel::load(&ctx, &stream, &info, GpuOptions::default(), &mut loaded)?,
                Chosen::Default,
            ),
            GpuSetting::Fastest => {
                on_measure(measured_count(info.fp64_fast));
                let mut rejected = Vec::new();
                let (kernel, results) = fastest_staged(
                    &first_stage(info.fp64_fast),
                    second_stage,
                    third_stage,
                    |options| {
                        let k = Kernel::load(&ctx, &stream, &info, options, &mut loaded);
                        if let Err(GpuError::SelfCheck(o)) = &k {
                            rejected.push(*o);
                        }
                        k
                    },
                    measure,
                )?;
                (kernel, Chosen::Fastest { results, rejected })
            }
        };
        Ok(Gpu {
            kernel,
            info,
            chosen,
            target,
            ordinal: None,
        })
    }

    /// 同じ型の別の GPU で測った設定を使ったことを、表示に出す
    pub fn same_as_another(mut self) -> Gpu {
        self.chosen = Chosen::Same;
        self
    }

    /// 複数の GPU で探すとき、表示に GPU の番号を付ける
    pub fn numbered(mut self, ordinal: usize) -> Gpu {
        self.ordinal = Some(ordinal);
        self
    }

    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// 使っている設定
    pub fn options(&self) -> GpuOptions {
        self.kernel.options
    }
}

impl Searcher for Gpu {
    fn run(&self, config: &SearchConfig, on_progress: &mut dyn Progress) -> Outcome {
        run_with(
            &self.kernel,
            self.target,
            &self.info.name,
            config,
            on_progress,
        )
    }

    fn status(&self, _config: &SearchConfig) -> Vec<(&'static str, String)> {
        let (i, k) = (&self.info, &self.kernel);
        let how = match &self.chosen {
            Chosen::Fastest { results, .. } => tf!(
                "{} 通りを測って一番速いもの",
                "the fastest of {} measured",
                results.len()
            ),
            Chosen::Same => t!(
                "同じ型の GPU で測ったもの",
                "measured on a GPU of the same model"
            )
            .into(),
            Chosen::Fixed => t!("指定", "given").into(),
            Chosen::Default => t!(
                "既定。--check では測らない",
                "default; not measured with --check"
            )
            .into(),
        };
        let name = match self.ordinal {
            Some(n) => format!("#{n} {}", i.name),
            None => i.name.clone(),
        };
        let mut rows = vec![
            ("GPU", name),
            (
                t!("世代", "Compute"),
                format!("sm_{}{}", i.compute.0, i.compute.1),
            ),
            ("SM", i.sms.to_string()),
            (
                t!("メモリ", "Memory"),
                format!("{:.0} GB", i.memory as f64 / 1e9),
            ),
            (
                t!("設定", "Setting"),
                tf!("{}（{how}）", "{} ({how})", k.options),
            ),
            // CPU と同時に探すとき、CPU の「並列」と並ぶので別の名前にする
            (
                t!("グリッド", "Grid"),
                tf!(
                    "{} ブロック × {} スレッド × {} 鍵",
                    "{} blocks × {} threads × {} keys",
                    k.grid,
                    BLOCK,
                    k.options.batch
                ),
            ),
            (
                t!("表", "Table"),
                tf!(
                    "W = {}（{:.1} MB）",
                    "W = {} ({:.1} MB)",
                    k.options.width,
                    k.options.table_bytes() as f64 / 1e6
                ),
            ),
        ];
        // 計算が CPU と合わなかった設定（カーネルのバグか GPU の不具合。使わずに飛ばした）
        if let Chosen::Fastest { rejected, .. } = &self.chosen
            && !rejected.is_empty()
        {
            let list: Vec<String> = rejected.iter().map(|o| o.to_string()).collect();
            rows.push((
                t!("使えない設定", "Rejected"),
                tf!(
                    "{}（GPU の計算が CPU と合わない）",
                    "{} (GPU results differ from the CPU)",
                    list.join(t!("、", "; "))
                ),
            ));
        }
        rows
    }
}

/// 通し番号 `index` のシード（kernel.cu と同じ。ChaCha20 か、`key` の最後の 8 バイトに通し番号を足したもの）
fn seed(key: &[u8; 32], source: SeedSource, index: u64) -> [u8; 32] {
    match source {
        SeedSource::ChaCha20 => chacha20_seed(key, index),
        SeedSource::Counter => counter_seed(key, index),
    }
}

/// ChaCha20 の鍵を 32 ビットの語に（kernel.cu に渡す形）
fn key_words(key: &[u8; 32]) -> [u32; 8] {
    std::array::from_fn(|i| u32::from_le_bytes(*key[4 * i..].first_chunk().expect("32 バイト")))
}

/// 通し番号 `index` のシード（kernel.cu の chacha20_seed と同じ: ChaCha20 のブロック `index / 2` の 64 バイトの、
/// `index` が偶数なら前半・奇数なら後半の 32 バイト）
fn chacha20_seed(key: &[u8; 32], index: u64) -> [u8; 32] {
    let mut rng = ChaCha20Rng::from_seed(*key);
    rng.set_block_pos(index >> 1);
    let mut block = [0u8; 64];
    rng.fill_bytes(&mut block);
    let half = block.as_chunks::<32>().0;
    half[(index & 1) as usize]
}

/// カーネルをこの GPU 向けにコンパイルする。NVRTC がその世代を知らなければ（GPU より NVRTC が古い）、
/// 知っている中で新しい世代の PTX にして、ドライバに変換させる
fn compile(compute: (i32, i32), build: Build) -> Result<cudarc::nvrtc::Ptx, GpuError> {
    let exact = format!("sm_{}{}", compute.0, compute.1);
    let mut errors = Vec::new();
    for arch in [exact.as_str(), "compute_90", "compute_80", PTX_ARCH] {
        match compile_for(arch, build) {
            Ok(ptx) => return Ok(ptx),
            Err(e) => errors.push(format!("{arch}: {}", compile_error(&e))),
        }
    }
    Err(GpuError::Compile(errors.join("\n")))
}

/// NVRTC でカーネルを `arch`（sm_86・compute_75 など）向けの PTX にする
fn compile_for(arch: &str, build: Build) -> Result<cudarc::nvrtc::Ptx, CompileError> {
    let mut options = vec![format!("--gpu-architecture={arch}")];
    options.extend(build.defines());
    compile_ptx_with_opts(
        KERNEL,
        CompileOptions {
            options,
            ..Default::default()
        },
    )
}

/// kernel.cu（改行は LF にそろえる）とコンパイルの設定のハッシュ（埋め込む PTX の1行目に書く）
pub fn kernel_hash(build: Build) -> String {
    use sha2::{Digest, Sha256};
    let source = KERNEL.replace('\r', "");
    let digest = Sha256::digest(format!(
        "{source}\n// {PTX_ARCH} {}",
        build.defines().join(" ")
    ));
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// `build` の埋め込む PTX（`gpu/ptx/` のファイルの中身）を NVRTC で作る（`cargo make ptx`）。
/// 互換性のため、なるべく古い NVRTC で作る（PTX の版はコンパイルした NVRTC の版で決まり、それより古いドライバでは読めない）
pub fn portable_ptx(build: Build) -> Result<String, GpuError> {
    // SAFETY: ライブラリを読み込めるか試すだけ
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(GpuError::NvrtcNotFound);
    }
    let ptx = compile_for(PTX_ARCH, build)
        .map_err(|e| GpuError::Compile(format!("{PTX_ARCH}: {}", compile_error(&e))))?;
    Ok(format!(
        // PTX は ASCII だけ（ptxas は ASCII 以外の文字を受け付けない）
        "// meikey: sha256(kernel.cu, {}) = {} (regenerate: cargo make ptx)\n{}",
        build.defines().join(" "),
        kernel_hash(build),
        ptx.to_src()
    ))
}

fn compile_error(e: &CompileError) -> String {
    match e {
        CompileError::CompileError { log, .. } => log.to_string_lossy().into_owned(),
        other => format!("{other:?}"),
    }
}

/// 体の元 `fe` を、表の形 `layout` の語に（Packed は 32 ビット × 8 語、ほかは 10 リム）
fn fe_words(fe: Fe, layout: Layout, out: &mut [u32]) {
    let bytes = fe.to_bytes();
    let words: [u64; 4] = std::array::from_fn(|i| {
        u64::from_le_bytes(*bytes[8 * i..].first_chunk().expect("32 バイト"))
    });
    // start ビット目から len ビット（len ≤ 26）
    let bits = |start: usize, len: usize| -> u32 {
        let (q, r) = (start / 64, start % 64);
        let mut v = words[q] >> r;
        if r != 0 && q + 1 < 4 {
            v |= words[q + 1] << (64 - r);
        }
        (v & ((1 << len) - 1)) as u32
    };
    match layout {
        Layout::Packed => {
            for (j, w) in bytes.as_chunks::<4>().0.iter().enumerate() {
                out[j] = u32::from_le_bytes(*w);
            }
        }
        // i 番目のリムは ceil(51i / 2) ビット目から、偶数番目 26・奇数番目 25 ビット
        Layout::R25 => {
            for (i, o) in out.iter_mut().take(10).enumerate() {
                *o = bits((51 * i).div_ceil(2), if i % 2 == 0 { 26 } else { 25 });
            }
        }
        // i 番目のリムは 26i ビット目から 26 ビット
        Layout::K26 => {
            for (i, o) in out.iter_mut().take(10).enumerate() {
                *o = bits(26 * i, 26);
            }
        }
    }
}

/// 幅 `width` の表を GPU 用に並べる（1エントリ = y+x, y-x, 2dxy を 0 以上 p 未満にして、形 `layout` で）。
/// 1 行目の 3 つ目は 2dxy ではなく 2xy にする（最初の点を拡張座標にするとき、T = 2xy を掛け算なしで使う）。
/// 行ごとに別のスレッドで作る（W = 20 で約 650 MB あり、1スレッドでは数秒かかる）
fn table_words(width: u32, layout: Layout) -> Vec<u32> {
    let (rows, cols) = rows_cols(width as usize);
    let entry = layout.entry_words();
    // 1 つの値の語数（余白はエントリの最後）
    let per = if layout == Layout::Packed { 8 } else { 10 };
    let mut words = vec![0u32; rows * cols * entry];
    let half = Fe([2, 0, 0, 0, 0]).invert();
    std::thread::scope(|s| {
        for (i, row) in words.chunks_mut(cols * entry).enumerate() {
            s.spawn(move || {
                for_each_entry_in_row(width as usize, i, |k, mut e| {
                    if i == 0 {
                        // (y+x)^2 - (y-x)^2 = 4xy なので、2xy = ((y+x)^2 - (y-x)^2) / 2
                        let [ypx, ymx, _] = e;
                        e[2] = (ypx * ypx - ymx * ymx) * half;
                    }
                    let out = &mut row[k * entry..(k + 1) * entry];
                    for (f, fe) in e.into_iter().enumerate() {
                        fe_words(fe, layout, &mut out[per * f..per * (f + 1)]);
                    }
                });
            });
        }
    });
    words
}

#[cfg(test)]
mod tests {
    //! kernel.cu を CPU 向けにコンパイルしたもの（build.rs）を呼んで、Rust の実装と突き合わせる
    use super::*;
    use crate::property::check;
    #[cfg(meikey_gpu_host)]
    use crate::service::derive::comb::{ROWS, W, signed_digits};
    use crate::service::search::{Both, Parallel};
    use crate::simd_support::available;
    use cudarc::driver::sys::is_culib_present as driver_present;
    use cudarc::nvrtc::sys::is_culib_present as nvrtc_present;
    #[cfg(meikey_gpu_host)]
    use curve25519_dalek::scalar::clamp_integer;
    use num_bigint::BigUint;
    use proptest::prelude::*;
    use rand::random;
    use rstest::{fixture, rstest};
    #[cfg(meikey_gpu_host)]
    use sha2::{Digest, Sha512};
    use similar_asserts::assert_eq;
    use std::array::from_fn;
    use std::thread::scope;
    use std::thread::sleep;

    #[cfg(meikey_gpu_host)]
    unsafe extern "C" {
        fn meikey_test_seed(key: *const u32, index: u64, out: *mut u8);
        fn meikey_test_counter_seed(key: *const u32, index: u64, out: *mut u8);
        fn meikey_test_secret(seed: *const u8, out: *mut u8);
        fn meikey_test_secret_mid(key: *const u32, index: u64, out: *mut u8);
        fn meikey_test_digits(s: *const u8, w: u32, out: *mut i32);
        fn meikey_test_fe_mul(a: *const u32, b: *const u32, out: *mut u32);
        fn meikey_test_fe_sq(a: *const u32, out: *mut u32);
        fn meikey_test_fe_invert_bytes(a: *const u32, out: *mut u8);
        fn meikey_test_contains(
            pk: *const u8,
            needle: *const u8,
            len: u32,
            fold: u32,
            mode: u32,
        ) -> i32;
        fn meikey_test_contains_unsigned(
            pk: *const u8,
            needle: *const u8,
            len: u32,
            fold: u32,
            mode: u32,
        ) -> i32;
        fn meikey_test_tail_b64(pk: *const u8, out: *mut u8);
        fn meikey_test_block() -> u32;
    }

    #[cfg(meikey_gpu_host)]
    type DeriveFn = unsafe extern "C" fn(*const u32, u32, *const u32, u32, u64, *mut u8);
    #[cfg(meikey_gpu_host)]
    type SearchFn = unsafe extern "C" fn(
        u64,
        *const u32,
        u32,
        *const u32,
        u32,
        *const u8,
        u32,
        u32,
        u32,
        u64,
        u32,
        u32,
        *mut u64,
    );

    /// 1つの組（鍵の数・体演算の種類）で CPU 向けにコンパイルしたカーネル
    #[cfg(meikey_gpu_host)]
    struct HostKernel {
        build: Build,
        derive: DeriveFn,
        search: SearchFn,
        /// 32 ビット × 8 語で渡す体演算（kernel.cu の fe_op_words）
        fe_op: unsafe extern "C" fn(u32, *const u32, *const u32, *mut u8),
        batch: unsafe extern "C" fn() -> u32,
        field: unsafe extern "C" fn() -> u32,
        unpacked: unsafe extern "C" fn() -> u32,
        entry_words: unsafe extern "C" fn() -> u32,
        /// 値をブロックの中と同じ組でまとめて逆元にする（kernel.cu の invert_pooled）
        invert_pooled: unsafe extern "C" fn(*const u32, u32, *mut u8),
    }

    // 組ごとの関数の宣言と HOSTS（build.rs が作る）
    #[cfg(meikey_gpu_host)]
    include!(concat!(env!("OUT_DIR"), "/gpu_hosts.rs"));

    /// 幅 `width`・形 `layout` の表（ごとに1回だけ作る）
    #[cfg(meikey_gpu_host)]
    fn table(width: u32, layout: Layout) -> &'static [u32] {
        static TABLES: [std::sync::OnceLock<Vec<u32>>; 63] =
            [const { std::sync::OnceLock::new() }; 63];
        let l = match layout {
            Layout::Packed => 0,
            Layout::R25 => 1,
            Layout::K26 => 2,
        };
        TABLES[3 * width as usize + l].get_or_init(|| table_words(width, layout))
    }

    #[cfg(meikey_gpu_host)]
    fn to_limbs(fe: Fe) -> [u32; 10] {
        let mut out = [0u32; 10];
        for (i, l) in fe.reduce().into_iter().enumerate() {
            out[2 * i] = (l & ((1 << 26) - 1)) as u32;
            out[2 * i + 1] = (l >> 26) as u32;
        }
        out
    }

    #[cfg(meikey_gpu_host)]
    fn fe(bytes: [u8; 32]) -> Fe {
        let mut b = bytes;
        b[31] &= 0x7f;
        Fe::from_bytes(&b)
    }

    /// 文字列を含む鍵か、x の符号だけを変えると含む鍵（GPU が当たりとして知らせる鍵）か
    fn maybe_match(p: &Pattern, pk: &[u8; 32]) -> bool {
        [0u8, 0x80].into_iter().any(|flip| {
            let mut k = *pk;
            k[31] ^= flip;
            let mut text = [0u8; PUBLIC_B64_LEN];
            encode_public_b64(&k, &mut text);
            p.is_match(&text)
        })
    }

    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn builds_match_kernel() {
        assert_eq!(
            HOSTS.iter().map(|h| h.build).collect::<Vec<_>>(),
            BUILDS.to_vec()
        );
        for h in &HOSTS {
            // SAFETY: 引数なしの関数
            let (batch, field, unpacked, entry) =
                unsafe { ((h.batch)(), (h.field)(), (h.unpacked)(), (h.entry_words)()) };
            assert_eq!(batch, h.build.batch);
            assert_eq!(field, h.build.limbs.field());
            assert_eq!(unpacked == 1, h.build.limbs.layout() != Layout::Packed);
            assert_eq!(entry as usize, h.build.limbs.layout().entry_words());
        }
        // meikey_test_* は既定の組（先頭）
        assert_eq!(HOSTS[0].build, GpuOptions::default().build());
        // 1ブロックのスレッド数（カーネルのレジスタの上限と、逆元をまとめる組をこれで決めている）
        // SAFETY: 引数なしの関数
        assert_eq!(unsafe { meikey_test_block() }, BLOCK);
    }

    /// 表の大きさ（W = 12 で約 4.1 MB、16 で約 50 MB、20 で約 650 MB。リムの形の表は 4/3 倍）
    #[rstest]
    #[case::default(GpuOptions::default(), 16 * 32769 * 96)]
    #[case::unpacked(GpuOptions::new(16, 32, Limbs::K26Table), 16 * 32769 * 128)]
    #[case::narrow(GpuOptions::new(12, 16, Limbs::R32), 21 * 2049 * 96)]
    #[case::widest(GpuOptions::new(20, 16, Limbs::F51), 13 * 524289 * 96)]
    fn table_bytes(#[case] options: GpuOptions, #[case] want: usize) {
        assert_eq!(options.table_bytes(), want);
    }

    /// 体演算の種類は、`--gpu-limbs` の値と行き来できる。知らない値はエラー
    #[rstest]
    fn limbs_codes_round_trip() {
        for l in Limbs::ALL {
            assert_eq!(l.code().parse::<Limbs>(), Ok(l));
        }
        assert_eq!(
            "9".parse::<Limbs>(),
            Err("10・10u・8・10k・10ku・5f のどれかが必要".into())
        );
    }

    /// 表の値のリムの形は、field.rs の 51 ビット × 5 リムを分けたもの（25.5 ビット）・26 ビットずつに分けたもの
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn table_layouts_hold_limbs() {
        check(256, any::<[u8; 32]>(), |b| {
            let x = fe(b);
            let mut r25 = [0u32; 10];
            fe_words(x, Layout::R25, &mut r25);
            prop_assert_eq!(r25, to_limbs(x));
            let mut k26 = [0u32; 10];
            fe_words(x, Layout::K26, &mut k26);
            let back = k26
                .iter()
                .enumerate()
                .fold(BigUint::from(0u8), |acc, (i, &l)| {
                    acc + (BigUint::from(l) << (26 * i))
                });
            prop_assert_eq!(back, BigUint::from_bytes_le(&x.to_bytes()));
            prop_assert_eq!(k26.iter().all(|&l| l < 1 << 26), true);
            Ok(())
        });
    }

    /// 通し番号 2i・2i + 1 のシードを並べると、ChaCha20Rng（ストリーム 0）のブロック i の 64 バイトになる
    /// （2 本ぶんで 1 ブロック。重なりも抜けもない）
    #[rstest]
    fn seeds_are_consecutive_chacha20_output() {
        let key: [u8; 32] = from_fn(|i| i as u8);
        let mut rng = ChaCha20Rng::from_seed(key);
        let mut stream = [0u8; 64 * 3];
        rand::Rng::fill_bytes(&mut rng, &mut stream);
        let seeds: Vec<u8> = (0..6).flat_map(|i| chacha20_seed(&key, i)).collect();
        assert_eq!(seeds, stream.to_vec());
    }

    /// シードは rand の ChaCha20Rng（ストリーム 0）のブロック `index / 2` の前半（偶数）か後半（奇数）。
    /// 低セキュリティモードのシードは model::search の counter_seed と同じ
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn seed_matches_chacha20_rng() {
        check(256, (any::<[u8; 32]>(), any::<u64>()), |(key, index)| {
            let mut got = [0u8; 32];
            // SAFETY: key は 8 語、out は 32 バイト
            unsafe { meikey_test_seed(key_words(&key).as_ptr(), index, got.as_mut_ptr()) };
            prop_assert_eq!(got, chacha20_seed(&key, index));
            // SAFETY: 同上
            unsafe { meikey_test_counter_seed(key_words(&key).as_ptr(), index, got.as_mut_ptr()) };
            prop_assert_eq!(got, counter_seed(&key, index));
            Ok(())
        });
    }

    /// 秘密スカラーは clamp(SHA-512(seed)[..32])（sha2 と curve25519-dalek）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn secret_scalar_matches_sha2() {
        check(256, any::<[u8; 32]>(), |s| {
            let mut got = [0u8; 32];
            // SAFETY: どちらも 32 バイト
            unsafe { meikey_test_secret(s.as_ptr(), got.as_mut_ptr()) };
            let h: [u8; 64] = Sha512::digest(s).into();
            let want = clamp_integer(*h.first_chunk().unwrap());
            prop_assert_eq!(got, want);
            Ok(())
        });
    }

    /// SHA-512 の最初の 3 ラウンドを前計算しても、低セキュリティモードのシードの秘密スカラーは同じ
    /// （通し番号の足し算が最後の 8 バイトの中で桁上がりする値も試す）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn secret_mid_matches_sha2() {
        let index = prop_oneof![any::<u64>(), Just(0u64), Just(1u64), Just(u64::MAX)];
        check(256, (any::<[u8; 32]>(), index), |(key, index)| {
            let mut got = [0u8; 32];
            // SAFETY: key は 8 語、out は 32 バイト
            unsafe { meikey_test_secret_mid(key_words(&key).as_ptr(), index, got.as_mut_ptr()) };
            let h: [u8; 64] = Sha512::digest(counter_seed(&key, index)).into();
            prop_assert_eq!(got, clamp_integer(*h.first_chunk().unwrap()));
            Ok(())
        });
    }

    /// 幅 w の桁（comb.rs と同じ作り方を、多倍長整数で書いたもの）
    #[cfg(meikey_gpu_host)]
    fn reference_digits(s: &[u8; 32], w: u32) -> Vec<i32> {
        let mut m = (BigUint::from_bytes_le(s) >> 3u32) & ((BigUint::from(1u8) << 251u32) - 1u32);
        let rows = 251u32.div_ceil(w);
        let mask = (BigUint::from(1u8) << w) - 1u32;
        let (half, mut carry) = (1i64 << (w - 1), 0i64);
        let mut out = Vec::new();
        for i in 0..rows {
            let raw = i64::try_from(&m & &mask).unwrap() + carry;
            m >>= w;
            if i == rows - 1 {
                out.push(raw as i32);
            } else {
                carry = (raw + half) >> w;
                out.push((raw - (carry << w)) as i32);
            }
        }
        out
    }

    /// 桁の分解は、どの幅でも多倍長整数で計算したものと同じで、幅 12 なら comb.rs の桁とも同じ。
    /// 桁を足し戻すと m になる（下の桁は -2^(W-1)〜2^(W-1) - 1、最上位は 0〜2^(W-1)）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn digits_match_reference(#[values(2, 3, 7, 11, 12, 13, 16, 17, 19, 20)] w: u32) {
        check(128, any::<[u8; 32]>().prop_map(clamp_integer), |s| {
            let rows = 251u32.div_ceil(w) as usize;
            let mut got = vec![0i32; rows];
            // SAFETY: s は 32 バイト、out は ceil(251 / w) 個
            unsafe { meikey_test_digits(s.as_ptr(), w, got.as_mut_ptr()) };
            prop_assert_eq!(&got, &reference_digits(&s, w));
            if w as usize == W {
                prop_assert_eq!(got.clone(), signed_digits(&s).map(i32::from).to_vec());
                prop_assert_eq!(got.len(), ROWS);
            }
            let half = 1i32 << (w - 1);
            for (i, d) in got.iter().enumerate() {
                let ok = match i == rows - 1 {
                    true => (0..=half).contains(d),
                    false => (-half..half).contains(d),
                };
                prop_assert_eq!(ok, true, "{} 桁目 {}", i, d);
            }
            Ok(())
        });
    }

    /// 体の乗算・逆元は field.rs と同じ値（25.5 ビット × 10 リム）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn field_matches_fe() {
        check(256, (any::<[u8; 32]>(), any::<[u8; 32]>()), |(a, b)| {
            let (x, y) = (fe(a), fe(b));
            let mut prod = [0u32; 10];
            // SAFETY: どれも 10 リム
            unsafe {
                meikey_test_fe_mul(
                    to_limbs(x).as_ptr(),
                    to_limbs(y).as_ptr(),
                    prod.as_mut_ptr(),
                )
            };
            let back = Fe(from_fn(|i| {
                u64::from(prod[2 * i]) + (u64::from(prod[2 * i + 1]) << 26)
            }));
            prop_assert_eq!(back.to_bytes(), (x * y).to_bytes());
            if x.to_bytes() != [0; 32] {
                let mut inv = [0u8; 32];
                // SAFETY: 10 リムと 32 バイト
                unsafe { meikey_test_fe_invert_bytes(to_limbs(x).as_ptr(), inv.as_mut_ptr()) };
                prop_assert_eq!(inv, x.invert().to_bytes());
            }
            Ok(())
        });
    }

    /// 2 乗の専用の計算（fe_sq）は fe_mul(a, a) とビット単位で同じ。リムは fe_sq の上限
    /// （偶数番目 1.5·2^26 未満、奇数番目 1.5·2^25 未満）まで、上限ちょうど手前の値も含めて試す
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn fe_sq_matches_fe_mul() {
        let limit = |i: usize| {
            if i.is_multiple_of(2) {
                3u32 << 25
            } else {
                3u32 << 24
            }
        };
        let limbs = proptest::collection::vec(any::<u32>(), 10).prop_map(move |v| {
            from_fn::<u32, 10, _>(|i| match v[i] % 8 {
                0 => limit(i) - 1,
                1 => 0,
                _ => v[i] % limit(i),
            })
        });
        check(1024, limbs, |a| {
            let (mut sq, mut mul) = ([0u32; 10], [0u32; 10]);
            // SAFETY: どれも 10 リム
            unsafe {
                meikey_test_fe_sq(a.as_ptr(), sq.as_mut_ptr());
                meikey_test_fe_mul(a.as_ptr(), a.as_ptr(), mul.as_mut_ptr());
            }
            prop_assert_eq!(sq, mul);
            Ok(())
        });
    }

    /// 体演算（32 ビット × 8 語で渡す版）は、どの組でも mod p で正しく、結果は p 未満。8 リムは 2^256 未満の
    /// 任意の値（p 以上・2^255 以上を含む）を受け付けるので、境界の値も混ぜる。op: 0 = 加算、1 = 減算、
    /// 2 = 乗算、3 = 2乗、4 = 逆元（0 の逆元は 0）、5 = 加算の結果どうしの乗算、6 = 減算の結果どうしの乗算、
    /// 7 = 減算の結果の逆元（5〜7 はリムの上限の境界）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn fe_op_words_matches_bigint(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] variant: usize,
        #[values(0, 1, 2, 3, 4, 5, 6, 7)] op: u32,
    ) {
        let h = &HOSTS[variant];
        let p = (BigUint::from(1u8) << 255u32) - 19u32;
        let full = (BigUint::from(1u8) << 256u32) - 1u32;
        // 8 リム以外は 2^255 未満の値だけ（表の値は p 未満）
        let max = if h.build.limbs == Limbs::R32 {
            full.clone()
        } else {
            (BigUint::from(1u8) << 255u32) - 1u32
        };
        let edges: Vec<BigUint> = [
            BigUint::from(0u8),
            BigUint::from(1u8),
            BigUint::from(38u8),
            &p - 1u32,
            p.clone(),
            &p + 1u32,
            &p * 2u32 - 1u32,
            &full - 37u32,
            full.clone(),
            BigUint::from(1u8) << 255u32,
            (BigUint::from(1u8) << 255u32) - 1u32,
            (BigUint::from(1u8) << 204u32) - 1u32,
        ]
        .into_iter()
        .filter(|x| x <= &max)
        .collect();
        let words = |x: &BigUint| -> [u32; 8] {
            let d = x.to_u32_digits();
            from_fn(|i| d.get(i).copied().unwrap_or(0))
        };
        let pick = |raw: [u32; 8], e: Option<usize>| -> BigUint {
            match e {
                Some(i) => edges[i % edges.len()].clone(),
                None => BigUint::from_slice(&raw) % (&max + 1u32),
            }
        };
        let strategy = (
            any::<[u32; 8]>(),
            any::<[u32; 8]>(),
            proptest::option::weighted(0.3, any::<usize>()),
            proptest::option::weighted(0.3, any::<usize>()),
        );
        let inv = |x: &BigUint| (x % &p).modpow(&(&p - 2u32), &p);
        check(512, strategy, |(ra, rb, ea, eb)| {
            let (a, b) = (pick(ra, ea), pick(rb, eb));
            let diff = (&a + &p * 4u32 - &b) % &p;
            let want = match op {
                0 => (&a + &b) % &p,
                1 => diff.clone(),
                2 => (&a * &b) % &p,
                3 => (&a * &a) % &p,
                4 => inv(&a),
                5 => (&a + &b) * (&a + &b) % &p,
                6 => (&p - &diff * &diff % &p) % &p,
                _ => inv(&diff),
            };
            let mut got = [0u8; 32];
            // SAFETY: a・b は 8 語、out は 32 バイト
            unsafe { (h.fe_op)(op, words(&a).as_ptr(), words(&b).as_ptr(), got.as_mut_ptr()) };
            prop_assert_eq!(
                BigUint::from_bytes_le(&got),
                want,
                "{} op {} {:x} {:x}",
                h.build,
                op,
                a,
                b
            );
            Ok(())
        });
    }

    /// 8 リムの GPU 用のインラインアセンブリ（fe_add・fe_sub・fe_mul・fe_sq）を、PTX の命令の意味どおりに解釈して
    /// 実行すると、mod p で正しく、結果は 2^256 未満（CPU 向けのコンパイルではアセンブリを通らないので、ここで確かめる）。
    /// fe_sq は fe_mul(a, a) とビット単位で同じ
    #[rstest]
    fn r32_inline_asm_is_correct() {
        let p = (BigUint::from(1u8) << 255u32) - 19u32;
        let full = (BigUint::from(1u8) << 256u32) - 1u32;
        let asm = |name: &str| ptx_asm::extract(KERNEL, name);
        let (add, sub, mul, sq) = (asm("fe_add"), asm("fe_sub"), asm("fe_mul"), asm("fe_sq"));
        let edges: Vec<BigUint> = vec![
            BigUint::from(0u8),
            BigUint::from(1u8),
            BigUint::from(38u8),
            &p - 1u32,
            p.clone(),
            &p + 1u32,
            &p * 2u32 - 1u32,
            &full - 37u32,
            full.clone(),
            BigUint::from(1u8) << 255u32,
        ];
        let words = |x: &BigUint| -> [u32; 8] {
            let d = x.to_u32_digits();
            from_fn(|i| d.get(i).copied().unwrap_or(0))
        };
        let pick = |raw: [u32; 8], e: Option<usize>| -> BigUint {
            match e {
                Some(i) => edges[i % edges.len()].clone(),
                None => BigUint::from_slice(&raw),
            }
        };
        let strategy = (
            any::<[u32; 8]>(),
            any::<[u32; 8]>(),
            proptest::option::weighted(0.3, any::<usize>()),
            proptest::option::weighted(0.3, any::<usize>()),
        );
        check(4096, strategy, |(ra, rb, ea, eb)| {
            let (a, b) = (pick(ra, ea), pick(rb, eb));
            let (wa, wb) = (words(&a), words(&b));
            let num = |w: [u32; 8]| BigUint::from_slice(&w);
            let ab: Vec<u32> = wa.iter().chain(&wb).copied().collect();
            for (code, want) in [
                (&add, (&a + &b) % &p),
                (&sub, (&a + &p * 4u32 - &b) % &p),
                (&mul, (&a * &b) % &p),
            ] {
                prop_assert_eq!(num(ptx_asm::run(code, &ab)) % &p, want);
            }
            let squared = ptx_asm::run(&sq, &wa);
            prop_assert_eq!(num(squared) % &p, (&a * &a) % &p);
            let aa: Vec<u32> = wa.iter().chain(&wa).copied().collect();
            prop_assert_eq!(squared, ptx_asm::run(&mul, &aa));
            Ok(())
        });
    }

    /// kernel.cu のインラインアセンブリ（8 リムの体演算）を取り出して、PTX の命令の意味どおりに実行する
    mod ptx_asm {
        /// `name` の asm の命令（8 リムの節の、CUDA 向けの関数）
        pub fn extract(kernel: &str, name: &str) -> Vec<String> {
            let section = &kernel[kernel.find("#elif MEIKEY_FIELD == 8").expect("8 リムの節")..];
            let start = section
                .find(&format!("DEV fe {name}("))
                .unwrap_or_else(|| panic!("{name} が無い"));
            let body = &section[start..];
            let body =
                &body[body.find("asm(").expect("asm")..body.find("        :").expect("出力")];
            body.lines()
                .filter_map(|l| {
                    let l = l.trim();
                    let s = l.strip_prefix('"')?.strip_suffix('"')?;
                    let s = s.trim_end_matches("\\n\\t").trim();
                    (!s.is_empty() && s != "{" && s != "}" && !s.starts_with(".reg"))
                        .then(|| s.trim_end_matches(';').to_string())
                })
                .collect()
        }

        /// 入力 %n（n = 出力の数 8 から）を `inputs` にして実行し、出力 %0〜%7 を返す。
        /// 桁上がりのフラグは足し算では桁上がり、引き算では借り
        pub fn run(code: &[String], inputs: &[u32]) -> [u32; 8] {
            let mut regs = std::collections::HashMap::<String, u32>::new();
            for (i, v) in inputs.iter().enumerate() {
                regs.insert(format!("%{}", 8 + i), *v);
            }
            let mut cf = 0u64;
            let val = |regs: &std::collections::HashMap<String, u32>, x: &str| -> u64 {
                match regs.get(x) {
                    Some(v) => u64::from(*v),
                    None => x.parse::<u64>().unwrap_or_else(|_| panic!("{x} が無い")),
                }
            };
            for line in code {
                let (op, args) = line.split_once(char::is_whitespace).expect("命令");
                let a: Vec<&str> = args.split(',').map(str::trim).collect();
                let parts: Vec<&str> = op.split('.').collect();
                let cc = parts.contains(&"cc");
                let carry_in = parts[0].ends_with('c');
                let result = match parts[0] {
                    "mov" => val(&regs, a[1]),
                    "and" => val(&regs, a[1]) & val(&regs, a[2]),
                    "mul" => val(&regs, a[1]) * val(&regs, a[2]),
                    "add" | "addc" => {
                        let s = val(&regs, a[1]) + val(&regs, a[2]) + if carry_in { cf } else { 0 };
                        if cc {
                            cf = s >> 32;
                        }
                        s
                    }
                    "sub" | "subc" => {
                        let s = i128::from(val(&regs, a[1]))
                            - i128::from(val(&regs, a[2]))
                            - if carry_in { i128::from(cf) } else { 0 };
                        if cc {
                            cf = u64::from(s < 0);
                        }
                        s.rem_euclid(1 << 32) as u64
                    }
                    "mad" | "madc" => {
                        let prod = val(&regs, a[1]) * val(&regs, a[2]);
                        let part = match parts.contains(&"hi") {
                            true => prod >> 32,
                            false => prod & 0xffff_ffff,
                        };
                        let s = part + val(&regs, a[3]) + if carry_in { cf } else { 0 };
                        if cc {
                            cf = s >> 32;
                        }
                        s
                    }
                    other => panic!("知らない命令 {other}"),
                };
                regs.insert(a[0].to_string(), (result & 0xffff_ffff) as u32);
            }
            from_fn_out(&regs)
        }

        fn from_fn_out(regs: &std::collections::HashMap<String, u32>) -> [u32; 8] {
            std::array::from_fn(|i| regs[&format!("%{i}")])
        }
    }

    /// まとめて計算した公開鍵は、同じシードから ed25519-dalek で作った鍵と一致する（埋め込む組すべて、
    /// 幅をいくつか）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn derive_batch_matches_dalek(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] variant: usize,
        #[values(SeedSource::ChaCha20, SeedSource::Counter)] source: SeedSource,
        #[values(2, 9, 12)] width: u32,
    ) {
        let h = &HOSTS[variant];
        let (batch, derive) = (h.build.batch, h.derive);
        let table = table(width, h.build.limbs.layout());
        check(4, (any::<[u8; 32]>(), any::<u64>()), |(key, first)| {
            let first = first >> 8; // 足しても桁あふれしないように
            let mut got = vec![0u8; 32 * batch as usize];
            // SAFETY: 表はその幅・形の行数·列数のエントリ、key は 8 語、out は batch·32 バイト
            unsafe {
                derive(
                    table.as_ptr(),
                    width,
                    key_words(&key).as_ptr(),
                    u32::from(source == SeedSource::Counter),
                    first,
                    got.as_mut_ptr(),
                )
            };
            for (j, pk) in got.as_chunks::<32>().0.iter().enumerate() {
                let want = VanityKey::from_seed(seed(&key, source, first + j as u64));
                prop_assert_eq!(pk, want.public_bytes(), "#{}", j);
            }
            Ok(())
        });
    }

    /// x の符号を 0 にした pk で「含まれる可能性がある」が偽なら、符号が 0 でも 1 でも含まれない。
    /// 符号で変わる文字（後ろから 2 文字目）を含む位置に文字列を書いた鍵も混ぜる
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn contains_unsigned_never_misses(
        #[values(false, true)] case_sensitive: bool,
        #[values("[A-Za-z0-9+/]{1,2}", "[A-Za-z0-9+/]{3,6}")] needles: &'static str,
        #[values(Mode::Contains, Mode::Suffix, Mode::Regex)] mode: Mode,
    ) {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD;
        let encode = |pk: &[u8; 32]| {
            let mut b = [0u8; PUBLIC_B64_LEN];
            encode_public_b64(pk, &mut b);
            b
        };
        let strategy = (any::<[u8; 32]>(), needles, 0usize..8, any::<bool>());
        check(2048, strategy, |(mut pk, n, back, plant)| {
            if plant {
                // 符号が 1 の鍵の base64 の、符号で変わる文字を含む位置に文字列を書いて、鍵に戻す
                pk[31] |= 0x80;
                let mut text = encode(&pk);
                let sign_char = text.len() - 2;
                let start = (sign_char + 1)
                    .saturating_sub(back.min(n.len()))
                    .min(text.len() - n.len());
                text[start..start + n.len()].copy_from_slice(n.as_bytes());
                pk = *b64.decode(text).unwrap().last_chunk::<32>().unwrap();
            }
            let p = mode_pattern(&n, case_sensitive, mode);
            let mut unsigned = pk;
            unsigned[31] &= 0x7f;
            let maybe = host_match(&unsigned, &p, true);
            for sign in [0u8, 0x80] {
                let mut k = unsigned;
                k[31] |= sign;
                let text = encode(&k);
                if p.is_match(&text) {
                    prop_assert!(maybe, "見逃した: {}", String::from_utf8_lossy(&text));
                }
            }
            Ok(())
        });
    }

    /// base64 は model::key と同じで、判定は Pattern と同じ（大小文字の区別あり・なし）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn b64_and_match_agree_with_pattern(
        #[values(false, true)] case_sensitive: bool,
        #[values(Mode::Contains, Mode::Suffix, Mode::Regex)] mode: Mode,
    ) {
        let needle = "[A-Za-z0-9+/]{1,3}";
        check(1024, (any::<[u8; 32]>(), needle), |(seed, n)| {
            let key = VanityKey::from_seed(seed);
            let pk = key.public_bytes();
            let mut tail = [0u8; 44];
            // SAFETY: pk は 32 バイト、out は 44 バイト
            unsafe { meikey_test_tail_b64(pk.as_ptr(), tail.as_mut_ptr()) };
            prop_assert_eq!(&tail[..], &key.public_b64()[24..]);

            let p = mode_pattern(&n, case_sensitive, mode);
            prop_assert_eq!(host_match(pk, &p, false), p.is_match(&key.public_b64()));
            Ok(())
        });
    }

    /// 鍵の base64 の一部（位置・長さはいろいろ。大小文字を変えたり1文字変えたりもする）を探すと、
    /// Pattern と同じ結果になる（先頭 2 文字で候補を絞ったあとの、残りを比べる部分も通す）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn match_agrees_with_pattern_for_parts_of_the_key(
        #[values(false, true)] case_sensitive: bool,
        #[values(Mode::Contains, Mode::Suffix, Mode::Regex)] mode: Mode,
    ) {
        let cases = (
            any::<[u8; 32]>(),
            0usize..44,
            1usize..=44,
            any::<u64>(),
            proptest::option::of((0usize..44, 0u8..64)),
        );
        // 正規表現は 1 件ごとに DFA と確率を計算するので、件数を減らす
        let n = if mode == Mode::Regex { 512 } else { 4096 };
        check(n, cases, |(seed, start, len, flips, edit)| {
            let key = VanityKey::from_seed(seed);
            let b64 = key.public_b64();
            let tail = &b64[24..];
            let len = len.min(44 - start);
            let mut n: Vec<u8> = tail[start..start + len].to_vec();
            // 大小文字をビットごとに入れ替える（区別しないなら一致のまま、区別するなら変わる）
            for (i, c) in n.iter_mut().enumerate() {
                if flips >> (i % 64) & 1 == 1 {
                    c.make_ascii_uppercase();
                } else {
                    c.make_ascii_lowercase();
                }
            }
            if let Some((i, v)) = edit {
                let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                n[i % len] = alphabet[v as usize];
            }
            let n = String::from_utf8(n).unwrap();
            let source = match mode {
                Mode::Regex => regex_syntax::escape(&n),
                _ => n.clone(),
            };
            let Ok(p) = Pattern::with_mode(&source, case_sensitive, mode) else {
                return Ok(()); // 長すぎるなど、Pattern が受け付けない文字列
            };
            let got = host_match(key.public_bytes(), &p, false);
            prop_assert_eq!(
                got,
                p.is_match(&b64),
                "{:?} in {:?}",
                n,
                String::from_utf8_lossy(tail)
            );
            Ok(())
        });
    }

    /// 文字列 `n` を探し方 `mode` で探す Pattern（正規表現なら、文字列そのものと、それを少し広げたもの
    /// （1 文字目は大小文字のどちらでも・最後は 1 文字足してもよい）を交互に）
    #[cfg(meikey_gpu_host)]
    fn mode_pattern(n: &str, case_sensitive: bool, mode: Mode) -> Pattern {
        let s = match mode {
            Mode::Regex if n.len().is_multiple_of(2) => regex_syntax::escape(n),
            Mode::Regex => {
                let (first, rest) = n.split_at(1);
                format!(
                    "[{}{}]{}.?",
                    first.to_ascii_lowercase(),
                    first.to_ascii_uppercase(),
                    regex_syntax::escape(rest)
                )
            }
            _ => n.to_string(),
        };
        Pattern::with_mode(&s, case_sensitive, mode).unwrap()
    }

    /// CPU 向けにコンパイルした kernel.cu の pk_match で判定する（`unsigned` なら x の符号のビットは 0 にしてある）
    #[cfg(meikey_gpu_host)]
    fn host_match(pk: &[u8; 32], p: &Pattern, unsigned: bool) -> bool {
        let n = Needle::new(Some(p));
        let bytes = [&n.bytes[..], &[0; 64]].concat();
        let f = match unsigned {
            true => meikey_test_contains_unsigned,
            false => meikey_test_contains,
        };
        // SAFETY: pk は 32 バイト、needle は len バイト（正規表現なら DFA の表）と余白
        unsafe {
            f(
                pk.as_ptr(),
                bytes.as_ptr(),
                n.len,
                u32::from(n.fold),
                n.mode,
            ) != 0
        }
    }

    /// 埋め込んだ PTX は今の kernel.cu とその組から作ったもの（違えば `cargo make ptx` で作り直す）
    #[rstest]
    fn embedded_ptx_matches_kernel_source(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] i: usize,
    ) {
        let (build, ptx) = (BUILDS[i], PTXS[i]);
        let first = ptx.lines().next().unwrap_or_default();
        assert_eq!(
            first.contains(&kernel_hash(build)),
            true,
            "{} が kernel.cu と食い違っている（cargo make ptx で作り直す）: {first}",
            build.ptx_file()
        );
        assert_eq!(
            ptx.contains(".target sm_75")
                && ptx.contains(".entry meikey_search")
                && ptx.contains(".entry meikey_pooled"),
            true
        );
        assert_eq!(embedded_ptx(build), Some(ptx));
    }

    /// 埋め込む組は重ならず、測らないときの既定と、測る設定を全部含む。1 段目は幅 16・32 鍵で体演算を全部
    /// （倍精度の遅い GPU では FP64 を除く）、2 段目はその種類で幅 5 通り × 鍵の数 2 通り、3 段目はブロック数 3 通り
    #[rstest]
    fn builds_cover_measured_settings() {
        for (i, b) in BUILDS.iter().enumerate() {
            assert_eq!(BUILDS[..i].contains(b), false, "{b} が重なっている");
        }
        assert_eq!(BUILDS[0], GpuOptions::default().build());
        let first = first_stage(true);
        assert_eq!(
            first.iter().map(|o| o.limbs).collect::<Vec<_>>(),
            Limbs::ALL.to_vec()
        );
        assert_eq!(
            first_stage(false).iter().any(|o| o.limbs == Limbs::F51),
            false
        );
        for l in Limbs::ALL {
            let second = second_stage(l);
            assert_eq!(second.len(), 10);
            for o in first.iter().chain(&second) {
                assert_eq!(o.in_range(), true, "{o}");
                assert_eq!(embedded_ptx(o.build()).is_some(), true, "{o}");
            }
        }
        assert_eq!(measured_count(true), 19);
        assert_eq!(measured_count(false), 18);
        let o = GpuOptions::new(18, 16, Limbs::R32);
        assert_eq!(third_stage(o), vec![o, o.with_blocks(3), o.with_blocks(2)]);
        assert_eq!(third_stage(o).iter().all(|o| o.in_range()), true);
        assert_eq!(
            GpuOptions::new(12, 32, Limbs::R25)
                .with_blocks(0)
                .in_range(),
            false
        );
        assert_eq!(
            embedded_ptx(Build {
                batch: 8,
                limbs: Limbs::R25
            }),
            None
        );
        assert_eq!(GpuOptions::new(21, 32, Limbs::R25).in_range(), false);
        assert_eq!(GpuOptions::new(12, 65, Limbs::R25).in_range(), false);
    }

    /// GpuError の日本語・英語の表示
    #[rstest]
    #[case::driver(
        GpuError::DriverNotFound,
        "CUDA のドライバが見つからない（NVIDIA の GPU とドライバが必要。libcuda / nvcuda.dll）",
        "CUDA driver not found (an NVIDIA GPU and its driver are needed; libcuda / nvcuda.dll)"
    )]
    #[case::no_device(GpuError::NoDevice, "CUDA の GPU が無い", "no CUDA GPU found")]
    #[case::nvrtc(
        GpuError::NvrtcNotFound,
        "libnvrtc が見つからない",
        "libnvrtc not found"
    )]
    #[case::range(
        GpuError::OutOfRange(GpuOptions::new(21, 32, Limbs::R25)),
        "表の幅は 2〜20、まとめる鍵の数は 1〜64（W = 21・32 鍵・10 リム）",
        "the table width must be 2-20 and the keys per batch 1-64 (W = 21, 32 keys, 10 limbs)"
    )]
    #[case::not_embedded(
        GpuError::NotEmbedded(Build { batch: 8, limbs: Limbs::R32 }),
        "8 鍵・8 リム は埋め込んでいない設定なので NVRTC（libnvrtc）が要る（LD_LIBRARY_PATH / PATH に追加）",
        "8 keys, 8 limbs is not built in, so NVRTC (libnvrtc) is needed (add it to LD_LIBRARY_PATH / PATH)"
    )]
    #[case::self_check(
        GpuError::SelfCheck(GpuOptions::new(16, 16, Limbs::K26Table)),
        "W = 16・16 鍵・10 リム・Karatsuba・展開した表 の GPU の計算が CPU と合わない（この設定は使えない）",
        "the GPU results of W = 16, 16 keys, 10 limbs, Karatsuba, unpacked table differ from the CPU (this setting cannot be used)"
    )]
    #[case::fp64(
        GpuError::SelfCheck(GpuOptions::new(18, 32, Limbs::F51)),
        "W = 18・32 鍵・5 リム・FP64 の GPU の計算が CPU と合わない（この設定は使えない）",
        "the GPU results of W = 18, 32 keys, 5 limbs, FP64 differ from the CPU (this setting cannot be used)"
    )]
    #[case::blocks(
        GpuError::SelfCheck(GpuOptions::new(16, 32, Limbs::R25).with_blocks(3)),
        "W = 16・32 鍵・10 リム・SM に 3 ブロック の GPU の計算が CPU と合わない（この設定は使えない）",
        "the GPU results of W = 16, 32 keys, 10 limbs, 3 blocks per SM differ from the CPU (this setting cannot be used)"
    )]
    #[case::no_settings(GpuError::NoSettings, "測る設定が無い", "no settings to measure")]
    #[case::compile(GpuError::Compile("sm_90: log".into()), "カーネルのコンパイルに失敗:\nsm_90: log", "failed to compile the kernel:\nsm_90: log")]
    fn gpu_error_display(#[case] e: GpuError, #[case] ja: &str, #[case] en: &str) {
        assert_eq!(e.to_string(), ja);
        crate::lang::set(crate::lang::Lang::En);
        assert_eq!(e.to_string(), en);
    }

    /// 測った中で一番速い設定を選ぶ。読み込めない・測れない設定は飛ばし、全部だめなら最初のエラー。
    /// `speeds[i]` は i 番目の設定の速さ（Err なら読み込めない）、`broken` は測ると失敗する設定
    #[rstest]
    #[case::picks_fastest(&[Ok(5.7), Ok(6.1), Ok(7.0), Ok(6.8)], &[], Ok(2))]
    #[case::first_of_ties(&[Ok(3.0), Ok(3.0)], &[], Ok(0))]
    #[case::skips_load_errors(&[Err("a"), Ok(1.0), Err("b"), Ok(2.0)], &[], Ok(3))]
    #[case::skips_measure_errors(&[Ok(1.0), Ok(9.0), Ok(2.0)], &[1], Ok(2))]
    #[case::all_load_errors(&[Err("a"), Err("b")], &[], Err(Some("a")))]
    #[case::all_measure_errors(&[Ok(1.0)], &[0], Err(Some("測れない")))]
    #[case::nothing(&[], &[], Err(None))]
    fn fastest_picks_the_fastest(
        #[case] speeds: &[Result<f64, &str>],
        #[case] broken: &[usize],
        // Err(Some(ログ)) は最初のエラー（Compile）、Err(None) は測る設定が無い
        #[case] want: Result<usize, Option<&str>>,
    ) {
        let candidates: Vec<GpuOptions> = (0..speeds.len() as u32)
            .map(|i| GpuOptions::new(i + 2, 1, Limbs::R25))
            .collect();
        let got = fastest(
            &candidates,
            |o| {
                let i = (o.width - 2) as usize;
                speeds[i]
                    .map(|_| i)
                    .map_err(|e| GpuError::Compile(e.into()))
            },
            |&i| match broken.contains(&i) {
                true => Err(GpuError::Compile("測れない".into())),
                false => Ok(speeds[i].unwrap()),
            },
        );
        match (got, want) {
            (Ok((k, results)), Ok(w)) => {
                assert_eq!(k, w);
                let ok: Vec<(GpuOptions, f64)> = (0..speeds.len())
                    .filter(|i| speeds[*i].is_ok() && !broken.contains(i))
                    .map(|i| (candidates[i], speeds[i].unwrap()))
                    .collect();
                assert_eq!(results, ok, "平均は同じ速さのまま");
            }
            (Err(e), Err(Some(w))) => {
                assert_eq!(
                    matches!(&e, GpuError::Compile(log) if log == w),
                    true,
                    "{e:?}"
                )
            }
            (Err(e), Err(None)) => {
                assert_eq!(matches!(e, GpuError::NoSettings), true, "{e:?}")
            }
            (got, want) => panic!("{:?} ではなく {want:?}", got.map(|(k, _)| k)),
        }
    }

    /// 測るたびに GPU が速くなっていっても（クロックが上がっていく途中）、前から・後ろからの2周の平均で
    /// 比べるので、本当に速い設定を選ぶ（前からの1周だけなら、後で測った遅い設定を選んでしまう）。
    /// 最初の設定を1回測って捨ててから、2周測る
    #[rstest]
    fn fastest_is_not_fooled_by_warming_up() {
        let base = [1.0, 0.97, 0.95];
        let candidates: Vec<GpuOptions> = (0..3)
            .map(|i| GpuOptions::new(i + 2, 1, Limbs::R25))
            .collect();
        let mut calls = Vec::new();
        let (k, results) = fastest(
            &candidates,
            |o| Ok((o.width - 2) as usize),
            |&i| {
                calls.push(i);
                // 測るたびに 5% ずつ速くなる
                Ok(base[i] * (1.0 + 0.05 * calls.len() as f64))
            },
        )
        .unwrap();
        assert_eq!(
            calls,
            vec![0, 0, 1, 2, 2, 1, 0],
            "温めてから、前から1周・後ろから1周"
        );
        assert_eq!(k, 0);
        let forward_only = (0..3).max_by(|&a, &b| {
            let s = |i: usize| base[i] * (1.0 + 0.05 * (i + 2) as f64);
            s(a).total_cmp(&s(b))
        });
        assert_eq!(
            forward_only,
            Some(2),
            "前からの1周だけなら遅い設定を選んでしまう例"
        );
        assert_eq!(results.len(), 3);
    }

    /// 段に分けて測ると、1 段目で一番速かった体演算の種類で 2 段目（幅・鍵の数）を測り、その一番速い設定で
    /// 3 段目（1 SM に載せるブロック数）を測って、3 段目の一番速いものを使う。
    /// 2 段目が全部読み込めなければ、1 段目の一番速いものから 3 段目を測る
    #[rstest]
    fn fastest_staged_measures_the_best_limbs(#[values(false, true)] second_fails: bool) {
        let speed = |o: GpuOptions| -> f64 {
            let limbs = match o.limbs {
                Limbs::K26 => 3.0,
                Limbs::R32 => 2.0,
                _ => 1.0,
            };
            let blocks = if o.blocks == 3 { 0.5 } else { 0.0 };
            limbs + blocks + f64::from(o.width) / 100.0 + f64::from(o.batch) / 1000.0
        };
        let first = first_stage(false);
        let (mut asked, mut third_from) = (None, None);
        let (k, results) = fastest_staged(
            &first,
            |l| {
                asked = Some(l);
                second_stage(l)
            },
            |o| {
                third_from = Some(o);
                third_stage(o)
            },
            |o| match second_fails && o.width != DEFAULT_WIDTH {
                true => Err(GpuError::NoSettings),
                false => Ok(o),
            },
            |o| Ok(speed(*o)),
        )
        .unwrap();
        assert_eq!(asked, Some(Limbs::K26));
        // 2 段目が全部読み込めないときは、幅 16 の 2 つ（32・16 鍵）だけ読み込める
        let best = match second_fails {
            false => GpuOptions::new(20, 32, Limbs::K26),
            true => GpuOptions::new(16, 32, Limbs::K26),
        };
        assert_eq!(third_from, Some(best));
        assert_eq!(k, best.with_blocks(3));
        let second = if second_fails { 2 } else { 10 };
        assert_eq!(results.len(), first.len() + second + MEASURE_BLOCKS.len());
    }

    /// 1 SM に載せたいブロック数は、PTX の各カーネルの .minnctapersm を書き換えて変える
    /// （埋め込んだ PTX は既定の 4 で、探索と確かめる用の 2 つのカーネルにある）
    #[rstest]
    fn min_blocks_rewrites_ptx(#[values(0, 5, 11)] i: usize) {
        let ptx = PTXS[i];
        let count = |text: &str, n: u32| {
            text.lines()
                .filter(|l| l.trim_end() == format!(".minnctapersm {n}"))
                .count()
        };
        assert_eq!(count(ptx, DEFAULT_BLOCKS), 2);
        let changed = with_min_blocks(ptx, 2);
        assert_eq!(
            (count(&changed, 2), count(&changed, DEFAULT_BLOCKS)),
            (2, 0)
        );
        assert_eq!(with_min_blocks(ptx, DEFAULT_BLOCKS), ptx);
        // 改行が CRLF でも書き換える
        let crlf = ptx.replace('\n', "\r\n");
        assert_eq!(count(&with_min_blocks(&crlf, 3), 3), 2);
    }

    // ---- NVRTC・GPU があるときだけ（無ければ飛ばす。MEIKEY_REQUIRE_SIMD に nvrtc・cuda があれば失敗させる） ----

    #[fixture]
    fn nvrtc() -> Option<()> {
        // SAFETY: ライブラリを読み込めるか試すだけ
        let present = unsafe { nvrtc_present() };
        available("nvrtc", present.then_some(()))
    }

    /// 本物の GPU と、それを使うテストを1つずつ動かすためのロック（テストの間持っておく）。
    /// 同時に動かすと、ほかのテストが表を作る（W = 20 で約 650 MB）・設定を全部測るのと CPU・GPU を取り合い、
    /// 時間を確かめるテスト（gpu_with_cpu）が遅れる
    struct OnGpu {
        gpu: Gpu,
        lock: std::sync::MutexGuard<'static, ()>,
    }

    #[fixture]
    fn gpu() -> Option<OnGpu> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        // SAFETY: ライブラリを読み込めるか試すだけ
        let present =
            unsafe { driver_present() } && CudaContext::device_count().is_ok_and(|n| n > 0);
        available("cuda", present.then_some(()))?;
        // ほかのテストが失敗してロックが壊れていても続ける
        let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let gpu = Gpu::new(0, GpuSetting::Fixed(GpuOptions::default()), |_| {})
            .expect("GPU を使えるようにできる");
        Some(OnGpu { gpu, lock })
    }

    /// カーネルは Turing（T4）から Blackwell（RTX PRO 6000）まで、各世代向けに既定の組でコンパイルでき、
    /// Blackwell 向けには埋め込む組すべてでコンパイルできる
    #[rstest]
    #[case::turing((7, 5), 0)]
    #[case::ampere((8, 0), 0)]
    #[case::ampere_consumer((8, 6), 0)]
    #[case::ada((8, 9), 0)]
    #[case::hopper((9, 0), 0)]
    #[case::hopper_fp64((9, 0), 5)]
    fn kernel_compiles_for_each_generation(
        #[case] compute: (i32, i32),
        #[case] i: usize,
        nvrtc: Option<()>,
    ) {
        if nvrtc.is_none() {
            return;
        }
        compiles(compute, BUILDS[i]);
    }

    #[rstest]
    fn kernel_compiles_every_build_for_blackwell(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] i: usize,
        nvrtc: Option<()>,
    ) {
        if nvrtc.is_none() {
            return;
        }
        compiles((12, 0), BUILDS[i]);
    }

    /// `build` を世代 `compute` 向けにコンパイルでき、探索と公開鍵を書き出すカーネルがある
    fn compiles(compute: (i32, i32), build: Build) {
        let ptx = compile(compute, build).unwrap_or_else(|e| panic!("{e}"));
        let src = ptx.to_src();
        let want = format!(".target sm_{}{}", compute.0, compute.1);
        assert_eq!(src.contains(&want), true, "{want} の PTX になっていない");
        assert_eq!(
            src.contains(".entry meikey_search")
                && src.contains(".entry meikey_derive")
                && src.contains(".entry meikey_pooled"),
            true
        );
    }

    /// GPU で計算した公開鍵は、同じシードから ed25519-dalek で作った鍵と一致する（埋め込んだ組すべてで、
    /// 幅はいくつか。通し番号は奇数から始め、シードを ChaCha20 のブロックの後半から取る場合も通す）
    #[rstest]
    fn gpu_derive_matches_dalek(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] i: usize,
        #[values(9, 16, 20)] width: u32,
        gpu: Option<OnGpu>,
    ) {
        let Some(OnGpu { lock: _lock, .. }) = gpu else {
            return;
        };
        let b = BUILDS[i];
        let options = GpuOptions::new(width, b.batch, b.limbs);
        let gpu =
            Gpu::new(0, GpuSetting::Fixed(options), |_| {}).expect("GPU を使えるようにできる");
        let key: [u8; 32] = random();
        let first = (1 << 40) + 1;
        for source in [SeedSource::ChaCha20, SeedSource::Counter] {
            let got = gpu
                .kernel
                .derive_on_gpu(&key, source, first, 8, BLOCK)
                .unwrap();
            for (i, pk) in got.as_chunks::<32>().0.iter().enumerate() {
                let want = VanityKey::from_seed(seed(&key, source, first + i as u64));
                assert_eq!(pk, want.public_bytes(), "{options} #{i}");
            }
        }
    }

    /// 探索のカーネル（本物の GPU）が知らせる当たりは、文字列を含むか x の符号だけが違えば含む鍵で、
    /// 1 回の起動の中でそれより前の通し番号には無い（ブロックの中で逆元をまとめても、各スレッドの鍵は正しい）
    #[rstest]
    fn gpu_search_kernel_reports_first_candidate(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] i: usize,
        gpu: Option<OnGpu>,
    ) {
        let Some(OnGpu { lock: _lock, .. }) = gpu else {
            return;
        };
        let b = BUILDS[i];
        let options = GpuOptions::new(12, b.batch, b.limbs);
        let gpu =
            Gpu::new(0, GpuSetting::Fixed(options), |_| {}).expect("GPU を使えるようにできる");
        let p = Pattern::new("aB", true).unwrap();
        let key: [u8; 32] = random();
        let mut l = gpu
            .kernel
            .prepare(&key, SeedSource::Counter, &Needle::new(Some(&p)), true)
            .unwrap();
        let hit = gpu
            .kernel
            .launch(&mut l, 0, 1)
            .unwrap()
            .expect("当たりがある");
        let pk = |n: u64| *VanityKey::from_seed(seed(&key, SeedSource::Counter, n)).public_bytes();
        assert_eq!(maybe_match(&p, &pk(hit)), true, "{options}");
        // 当たりより前の鍵のうち、同じスレッドの前の鍵には当たりが無い（ほかのスレッドは順序が決まらない）
        let start = hit - hit % u64::from(b.batch);
        assert_eq!((start..hit).any(|n| maybe_match(&p, &pk(n))), false);
    }

    /// 設定を測って選ぶと、1 段目（体演算の種類）・2 段目（その種類の幅・鍵の数）・3 段目（1 SM に載せるブロック数）を
    /// 測って、一番速いものを使う。
    /// その設定でも鍵が見つかる
    #[rstest]
    fn gpu_uses_the_fastest_setting(gpu: Option<OnGpu>) {
        let Some(OnGpu { lock: _lock, .. }) = gpu else {
            return;
        };
        let mut count = None;
        let gpu = Gpu::new(0, GpuSetting::Fastest, |n| count = Some(n))
            .expect("GPU を使えるようにできる");
        assert_eq!(
            count,
            Some(measured_count(gpu.info.fp64_fast)),
            "測る前に知らせる"
        );
        let Chosen::Fastest { results, rejected } = &gpu.chosen else {
            panic!("測っていない");
        };
        assert_eq!(rejected, &Vec::new(), "計算が CPU と合わない設定がある");
        let first = first_stage(gpu.info.fp64_fast);
        let measured: Vec<GpuOptions> = results.iter().map(|r| r.0).collect();
        let second = first.len() + MEASURE_BATCHES.len() * MEASURE_WIDTHS.len();
        assert_eq!(measured[..first.len()].to_vec(), first);
        assert_eq!(
            measured[first.len()..second].to_vec(),
            second_stage(gpu.options().limbs)
        );
        assert_eq!(measured[second..].to_vec(), third_stage(measured[second]));
        let best = results[second..]
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert_eq!(gpu.options(), best.0);
        let p = Pattern::new("Ab1", false).unwrap();
        let config = SearchConfig {
            time_limit: Some(Duration::from_secs(60)),
            ..SearchConfig::new(Some(p.clone()))
        };
        let key = gpu
            .run(&config, &mut |_| {})
            .key
            .expect("3文字ならすぐ見つかる");
        assert_eq!(p.is_match(&key.public_b64()), true);
    }

    /// GPU で探すと、文字列を含む鍵が見つかる（大小文字の区別あり・なし）。ベンチマークは時間で止まる
    #[rstest]
    fn gpu_search_finds_key(#[values(false, true)] case_sensitive: bool, gpu: Option<OnGpu>) {
        let Some(OnGpu { gpu, lock: _lock }) = gpu else {
            return;
        };
        let p = Pattern::new("Ab1", case_sensitive).unwrap();
        let config = SearchConfig {
            time_limit: Some(Duration::from_secs(60)),
            ..SearchConfig::new(Some(p.clone()))
        };
        let out = gpu.run(&config, &mut |_| {});
        let key = out.key.expect("3文字ならすぐ見つかる");
        assert_eq!(p.is_match(&key.public_b64()), true);

        let bench = SearchConfig {
            bench: true,
            time_limit: Some(Duration::from_millis(500)),
            ..config
        };
        let out = gpu.run(&bench, &mut |_| {});
        assert_eq!(out.key.is_none() && out.stats.tries > 0, true);
    }

    /// GPU の探索は外からの合図で止まる（時間制限が無くても）。CPU と同時に探すと、どちらかが見つける
    #[rstest]
    fn gpu_with_cpu(#[values(false, true)] cancel_only: bool, gpu: Option<OnGpu>) {
        let Some(OnGpu { gpu, lock: _lock }) = gpu else {
            return;
        };
        if cancel_only {
            let config = SearchConfig::new(Some(
                Pattern::new(&"z".repeat(Pattern::MAX_LEN), false).unwrap(),
            ));
            let cancel = config.cancel.clone();
            let cancelled = std::sync::OnceLock::new();
            let out = scope(|s| {
                s.spawn(|| {
                    sleep(Duration::from_millis(300));
                    cancelled.set(Instant::now()).unwrap();
                    cancel.cancel();
                });
                gpu.run(&config, &mut |_| {})
            });
            // 合図してから止まるまで（1回の起動は 0.2 秒前後）
            let delay = cancelled.get().expect("止まる前に合図している").elapsed();
            assert_eq!(out.key.is_none() && out.stats.tries > 0, true);
            assert_eq!(
                delay < Duration::from_secs(5),
                true,
                "止まるのが遅い（{delay:?}）"
            );
        } else {
            let p = Pattern::new("Ab1c", false).unwrap();
            let config = SearchConfig {
                threads: 1,
                time_limit: Some(Duration::from_secs(60)),
                ..SearchConfig::new(Some(p.clone()))
            };
            let out = Both(gpu, Parallel).run(&config, &mut |_| {});
            let key = out.key.expect("4文字ならすぐ見つかる");
            assert_eq!(p.is_match(&key.public_b64()), true);
        }
    }

    /// カーネル全体（CPU で真似たもの）: 見つけた通し番号の鍵は文字列を含むか、x の符号だけが違えば含み、
    /// それより前の番号の鍵はどちらでもない（1スレッドなら最初の当たりを返す）。stop_on_hit なら途中で止まる
    /// （埋め込む組すべて）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn search_finds_first_hit(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] variant: usize,
        #[values(SeedSource::ChaCha20, SeedSource::Counter)] source: SeedSource,
    ) {
        let h = &HOSTS[variant];
        let width = 10;
        let (table, search) = (table(width, h.build.limbs.layout()), h.search);
        let key: [u8; 32] = from_fn(|i| i as u8 * 7);
        let p = Pattern::new("ab", false).unwrap();
        let needle = b"ab";
        let mut result = [0u64; 2];
        // SAFETY: 表・鍵・文字列・結果の大きさは kernel.cu の前提どおり
        unsafe {
            search(
                1,
                table.as_ptr(),
                width,
                key_words(&key).as_ptr(),
                u32::from(source == SeedSource::Counter),
                needle.as_ptr(),
                2,
                1,
                MODE_CONTAINS,
                0,
                32,
                1,
                result.as_mut_ptr(),
            );
        }
        assert_eq!(result[0], 1, "512 鍵以上のうちに「ab」はほぼ確実に出る");
        let hit = result[1];
        let pk = |i| *VanityKey::from_seed(seed(&key, source, i)).public_bytes();
        assert_eq!(maybe_match(&p, &pk(hit)), true);
        for i in 0..hit {
            assert_eq!(maybe_match(&p, &pk(i)), false, "{i} が先に当たる");
        }
    }

    /// 末尾・正規表現でも、カーネル全体（CPU で真似たもの）が最初の当たりを返す（体演算の違う組で）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn search_finds_first_hit_with_mode(
        #[values(0, 8, 11)] variant: usize,
        #[values((Mode::Suffix, "a"), (Mode::Suffix, "Q9"), (Mode::Regex, "^[A-P][0-9]"), (Mode::Regex, "x.y|z$"))]
        pattern: (Mode, &str),
    ) {
        let h = &HOSTS[variant];
        let width = 13;
        let (table, search) = (table(width, h.build.limbs.layout()), h.search);
        let key: [u8; 32] = from_fn(|i| (i as u8).wrapping_mul(11));
        let p = Pattern::with_mode(pattern.1, false, pattern.0).unwrap();
        let n = Needle::new(Some(&p));
        let bytes = [&n.bytes[..], &[0; 64]].concat();
        let mut result = [0u64; 2];
        // SAFETY: 表・鍵・文字列・結果の大きさは kernel.cu の前提どおり
        unsafe {
            search(
                1,
                table.as_ptr(),
                width,
                key_words(&key).as_ptr(),
                1,
                bytes.as_ptr(),
                n.len,
                u32::from(n.fold),
                n.mode,
                0,
                1024,
                1,
                result.as_mut_ptr(),
            );
        }
        assert_eq!(
            result[0], 1,
            "1 万鍵以上のうちにほぼ確実に出る（Q9 は 2048 鍵に 1 回）"
        );
        let hit = result[1];
        let maybe = |i| {
            maybe_match(
                &p,
                VanityKey::from_seed(seed(&key, SeedSource::Counter, i)).public_bytes(),
            )
        };
        assert_eq!(maybe(hit), true);
        assert_eq!((0..hit).any(maybe), false, "{hit} より前に当たりがある");
    }

    /// ブロックの中で逆元をまとめる組（ワープの同じレーンのスレッド。端数のブロックでは組の大きさがばらばら）で
    /// 逆元にしても、どの値も field.rs の逆元と同じ（埋め込む組すべて）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn pooled_inverses_match_fe(#[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] variant: usize) {
        let h = &HOSTS[variant];
        let values = proptest::collection::vec(any::<[u8; 32]>(), 1..=128);
        check(16, values, |raw| {
            let xs: Vec<Fe> = raw
                .iter()
                .map(|b| fe(*b))
                .map(|x| match x.to_bytes() == [0; 32] {
                    true => Fe([1, 0, 0, 0, 0]),
                    false => x,
                })
                .collect();
            let words: Vec<u32> = xs
                .iter()
                .flat_map(|x| {
                    x.to_bytes()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|w| u32::from_le_bytes(*w))
                        .collect::<Vec<_>>()
                })
                .collect();
            let mut out = vec![0u8; 32 * xs.len()];
            // SAFETY: words は n·8 語、out は n·32 バイト（n ≤ 128）
            unsafe { (h.invert_pooled)(words.as_ptr(), xs.len() as u32, out.as_mut_ptr()) };
            for (i, (x, got)) in xs.iter().zip(out.as_chunks::<32>().0).enumerate() {
                prop_assert_eq!(
                    got,
                    &x.invert().to_bytes(),
                    "{} #{} / {}",
                    h.build,
                    i,
                    xs.len()
                );
            }
            Ok(())
        });
    }

    // ---- GPU なしで探索の流れを確かめる（CPU 向けにコンパイルしたカーネルと、mockall のモック） ----

    /// CPU 向けにコンパイルしたカーネルを呼ぶ [`Engine`]（GPU の代わり）。`threads` スレッドぶんを順に回す
    #[cfg(meikey_gpu_host)]
    struct HostEngine {
        host: &'static HostKernel,
        width: u32,
        threads: u64,
    }

    /// [`HostEngine`] の起動に渡すもの（[`Launch`] と同じ中身を CPU のメモリに持つ）
    #[cfg(meikey_gpu_host)]
    struct HostLaunch {
        key: [u32; 8],
        counter: u32,
        needle: Vec<u8>,
        len: u32,
        fold: u32,
        mode: u32,
        stop_on_hit: u32,
        result: [u64; 2],
    }

    #[cfg(meikey_gpu_host)]
    impl HostEngine {
        /// 表は先に作っておく（探索の時間を測るテストで、最初の起動が表を作る分だけ遅れないように）
        fn new(variant: usize) -> HostEngine {
            let host = &HOSTS[variant];
            let width = 12;
            table(width, host.build.limbs.layout());
            HostEngine {
                host,
                width,
                threads: 4,
            }
        }
    }

    #[cfg(meikey_gpu_host)]
    impl Engine for HostEngine {
        type Launch = HostLaunch;

        fn keys_per_iter(&self) -> u64 {
            self.threads * u64::from(self.host.build.batch)
        }

        fn prepare(
            &self,
            key: &[u8; 32],
            source: SeedSource,
            needle: &Needle,
            stop_on_hit: bool,
        ) -> Result<HostLaunch, GpuError> {
            Ok(HostLaunch {
                key: key_words(key),
                counter: u32::from(source == SeedSource::Counter),
                needle: [&needle.bytes[..], &[0; 64]].concat(),
                len: needle.len,
                fold: u32::from(needle.fold),
                mode: needle.mode,
                stop_on_hit: u32::from(stop_on_hit),
                result: [0; 2],
            })
        }

        fn launch(
            &self,
            l: &mut HostLaunch,
            base: u64,
            iters: u32,
        ) -> Result<Option<u64>, GpuError> {
            // SAFETY: 表・鍵・文字列・結果の大きさは kernel.cu の前提どおり
            unsafe {
                (self.host.search)(
                    self.threads,
                    table(self.width, self.host.build.limbs.layout()).as_ptr(),
                    self.width,
                    l.key.as_ptr(),
                    l.counter,
                    l.needle.as_ptr(),
                    l.len,
                    l.fold,
                    l.mode,
                    base,
                    iters,
                    l.stop_on_hit,
                    l.result.as_mut_ptr(),
                );
            }
            let found = (l.result[0] != 0).then_some(l.result[1]);
            l.result = [0; 2];
            Ok(found)
        }
    }

    /// [`HostEngine`] で探す [`Searcher`]（[`Gpu`] と同じく [`run_with`] で探す。CPU と同時に探すテスト用）
    #[cfg(meikey_gpu_host)]
    struct HostGpu(HostEngine);

    #[cfg(meikey_gpu_host)]
    impl Searcher for HostGpu {
        fn run(&self, config: &SearchConfig, on_progress: &mut dyn Progress) -> Outcome {
            run_with(
                &self.0,
                Duration::from_millis(20),
                "host",
                config,
                on_progress,
            )
        }

        fn status(&self, _config: &SearchConfig) -> Vec<(&'static str, String)> {
            Vec::new()
        }
    }

    /// 探索の流れ（起動の繰り返し・当たりの確認）を CPU のカーネルで通すと、どの探し方でも一致する鍵が見つかる
    /// （体演算の違う組と、シードの作り方の両方で）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn host_search_finds_key(
        #[values(0, 2, 5, 9)] variant: usize,
        #[values((Mode::Contains, "ab"), (Mode::Suffix, "Q9"), (Mode::Regex, "^[A-P][0-9]"))]
        pattern: (Mode, &str),
        #[values(SeedSource::ChaCha20, SeedSource::Counter)] seeds: SeedSource,
    ) {
        let p = Pattern::with_mode(pattern.1, false, pattern.0).unwrap();
        let config = SearchConfig {
            seeds,
            time_limit: Some(Duration::from_secs(60)),
            ..SearchConfig::new(Some(p.clone()))
        };
        let engine = HostEngine::new(variant);
        let out = search_with(
            &engine,
            Duration::from_millis(20),
            "host",
            &config,
            &mut |_| {},
        )
        .unwrap();
        let key = out.key.expect("短い文字列なので見つかる");
        assert_eq!(p.is_match(&key.public_b64()), true);
        assert_eq!(out.stats.tries % engine.keys_per_iter(), 0);
    }

    /// 符号で変わる文字（後ろから 2 文字目）を含む末尾の文字列を探すと、符号だけが違う鍵も当たりとして知らされる。
    /// それは返さずに探し続け、本当に含む鍵を返す
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn host_search_skips_sign_only_hits() {
        let p = Pattern::with_mode("xA", true, Mode::Suffix).unwrap();
        let config = SearchConfig {
            seeds: SeedSource::Counter,
            time_limit: Some(Duration::from_secs(120)),
            ..SearchConfig::new(Some(p.clone()))
        };
        let engine = HostEngine {
            threads: 32,
            ..HostEngine::new(0)
        };
        let reported = std::sync::Mutex::new(Vec::new());
        struct Recording<'a>(&'a HostEngine, &'a std::sync::Mutex<Vec<u64>>);
        impl Engine for Recording<'_> {
            type Launch = HostLaunch;
            fn keys_per_iter(&self) -> u64 {
                self.0.keys_per_iter()
            }
            fn prepare(
                &self,
                key: &[u8; 32],
                source: SeedSource,
                needle: &Needle,
                stop_on_hit: bool,
            ) -> Result<HostLaunch, GpuError> {
                self.0.prepare(key, source, needle, stop_on_hit)
            }
            fn launch(
                &self,
                l: &mut HostLaunch,
                base: u64,
                iters: u32,
            ) -> Result<Option<u64>, GpuError> {
                let found = self.0.launch(l, base, iters)?;
                if let Some(i) = found {
                    self.1.lock().unwrap().push(i);
                }
                Ok(found)
            }
        }
        // 符号だけが違う当たりが先に出るまで何度か探す（どちらも 1/1024 くらいで出る）
        for _ in 0..32 {
            reported.lock().unwrap().clear();
            let out = search_with(
                &Recording(&engine, &reported),
                Duration::from_millis(20),
                "host",
                &config,
                &mut |_| {},
            )
            .unwrap();
            let key = out.key.expect("2 文字なので見つかる");
            assert_eq!(p.is_match(&key.public_b64()), true);
            if reported.lock().unwrap().len() >= 2 {
                return;
            }
        }
        panic!("符号だけが違う当たりが一度も出ない");
    }

    /// ベンチマークでは当たっても止まらず、時間切れで鍵なしで返る。進捗は途中で何度も知らせる
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn host_bench_runs_until_time_limit() {
        let config = SearchConfig {
            bench: true,
            seeds: SeedSource::Counter,
            time_limit: Some(Duration::from_millis(300)),
            progress_interval: Duration::from_millis(50),
            ..SearchConfig::new(Some(Pattern::new("a", false).unwrap()))
        };
        let mut reports = Vec::new();
        let out = search_with(
            &HostEngine::new(0),
            Duration::from_millis(20),
            "host",
            &config,
            &mut |s: Stats| reports.push(s.tries),
        )
        .unwrap();
        assert_eq!(out.key.is_none(), true);
        assert_eq!(out.stats.elapsed >= Duration::from_millis(300), true);
        assert_eq!(reports.len() >= 2, true, "{reports:?}");
        assert_eq!(reports.windows(2).all(|w| w[0] < w[1]), true, "{reports:?}");
    }

    /// 止める合図が出ていれば、1回起動したところで鍵なしで返る
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn host_search_stops_when_cancelled() {
        let config = SearchConfig {
            seeds: SeedSource::Counter,
            ..SearchConfig::new(Some(
                Pattern::new(&"z".repeat(Pattern::MAX_LEN), false).unwrap(),
            ))
        };
        config.cancel.cancel();
        let engine = HostEngine::new(0);
        let out = search_with(
            &engine,
            Duration::from_millis(20),
            "host",
            &config,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(
            (out.key.is_none(), out.stats.tries),
            (true, engine.keys_per_iter())
        );
    }

    /// 設定の速さも CPU のカーネルで測れる（正の有限の値）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn host_measure_gives_a_speed() {
        let speed = measure(&HostEngine::new(0)).unwrap();
        assert_eq!(speed.is_finite() && speed > 0.0, true, "{speed}");
    }

    /// CPU と同時に探しても（[`Both`]）、どちらかが見つけた一致する鍵が返る
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn host_gpu_with_cpu_finds_key() {
        let p = Pattern::new("ab", false).unwrap();
        let config = SearchConfig {
            threads: 1,
            seeds: SeedSource::Counter,
            time_limit: Some(Duration::from_secs(60)),
            ..SearchConfig::new(Some(p.clone()))
        };
        let out = Both(HostGpu(HostEngine::new(0)), Parallel).run(&config, &mut |_| {});
        let key = out.key.expect("短い文字列なので見つかる");
        assert_eq!(p.is_match(&key.public_b64()), true);
    }

    /// 起動がすぐ終わるなら、繰り返し回数を 4 倍ずつ（上限 2^16）増やし、通し番号は前の起動の続きから始める。
    /// 当たりの通し番号の鍵を返し、試行回数は起動した分の合計
    #[rstest]
    fn search_grows_iterations_and_continues_numbering() {
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut mock = MockEngine::new();
        mock.expect_keys_per_iter().return_const(10u64);
        mock.expect_prepare().times(1).returning(|_, _, _, stop| {
            assert_eq!(stop, true, "ベンチマークでなければ当たりで止める");
            Ok(())
        });
        let seen = calls.clone();
        mock.expect_launch().returning(move |_, base, iters| {
            let mut c = seen.lock().unwrap();
            c.push((base, iters));
            Ok((c.len() == 12).then_some(base + 3))
        });
        let config = SearchConfig::new(None);
        let out = search_with(
            &mock,
            Duration::from_millis(200),
            "mock",
            &config,
            &mut |_| {},
        )
        .unwrap();
        let calls = calls.lock().unwrap().clone();
        let iters: Vec<u32> = calls.iter().map(|c| c.1).collect();
        assert_eq!(
            iters,
            vec![
                1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 65536, 65536, 65536
            ]
        );
        let mut base = 0;
        for &(b, i) in &calls {
            assert_eq!(b, base);
            base += 10 * u64::from(i);
        }
        assert_eq!(out.stats.tries, base);
        assert_eq!(out.key.is_some(), true);
    }

    /// 起動の準備・起動に失敗したらエラーを返し、Searcher としては止まる（CUDA のエラーの文は、表示に
    /// ドライバが要るので、止まるときの表示は別のエラーで確かめる）
    #[rstest]
    #[case::prepare(true)]
    #[case::launch(false)]
    fn search_returns_errors(#[case] in_prepare: bool) {
        let mock = |error: fn() -> GpuError| {
            let mut mock = MockEngine::new();
            mock.expect_keys_per_iter().return_const(10u64);
            match in_prepare {
                true => {
                    mock.expect_prepare()
                        .returning(move |_, _, _, _| Err(error()));
                }
                false => {
                    mock.expect_prepare().returning(|_, _, _, _| Ok(()));
                    mock.expect_launch().returning(move |_, _, _| Err(error()));
                }
            }
            mock
        };
        let config = SearchConfig::new(None);
        let driver = mock(|| {
            GpuError::Driver(DriverError(
                cudarc::driver::sys::CUresult::CUDA_ERROR_LAUNCH_FAILED,
            ))
        });
        let e = search_with(
            &driver,
            Duration::from_millis(200),
            "mock",
            &config,
            &mut |_| {},
        )
        .unwrap_err();
        assert_eq!(matches!(e, GpuError::Driver(_)), true);
        let other = mock(|| GpuError::NoSettings);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_with(
                &other,
                Duration::from_millis(200),
                "mock",
                &config,
                &mut |_| {},
            )
        }));
        let message = *panicked.unwrap_err().downcast::<String>().unwrap();
        assert_eq!(message, "GPU での探索に失敗: 測る設定が無い");
    }

    /// GPU が文字列を含まない鍵を当たりとして返したら（符号を変えても含まない。GPU の計算か判定のバグ）、
    /// その鍵は返さずに止まる
    #[rstest]
    #[should_panic(expected = "内部エラー: GPU（mock）の結果が ed25519-dalek・総当たりと不一致")]
    fn search_rejects_wrong_hits() {
        let mut mock = MockEngine::new();
        mock.expect_keys_per_iter().return_const(10u64);
        mock.expect_prepare().returning(|_, _, _, _| Ok(()));
        mock.expect_launch().returning(|_, _, _| Ok(Some(0)));
        let p = Pattern::new(&"z".repeat(Pattern::MAX_LEN), false).unwrap();
        let config = SearchConfig::new(Some(p));
        let _ = search_with(
            &mock,
            Duration::from_millis(200),
            "mock",
            &config,
            &mut |_| {},
        );
    }

    /// 速さを測るときは、通し番号のシード・当たっても止まらない設定で、短い 1 回のあと同じ回数の起動を 3 回する
    #[rstest]
    fn measure_launches_once_then_three_times() {
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut mock = MockEngine::new();
        mock.expect_keys_per_iter().return_const(100u64);
        mock.expect_prepare()
            .times(1)
            .returning(|_, source, needle, stop| {
                assert_eq!(
                    (source, needle.mode, stop),
                    (SeedSource::Counter, MODE_CONTAINS, false)
                );
                Ok(())
            });
        let seen = calls.clone();
        mock.expect_launch().returning(move |_, base, iters| {
            seen.lock().unwrap().push((base, iters));
            sleep(Duration::from_millis(1));
            Ok(None)
        });
        let speed = measure(&mock).unwrap();
        let calls = calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 4);
        assert_eq!(calls[0], (0, 1));
        let iters = calls[1].1;
        assert_eq!(calls[1..].iter().all(|c| c.1 == iters), true, "{calls:?}");
        assert_eq!(
            calls[1..].iter().map(|c| c.0).collect::<Vec<_>>(),
            (1..=3)
                .map(|i| i * 100 * u64::from(iters))
                .collect::<Vec<_>>()
        );
        assert_eq!(speed.is_finite() && speed > 0.0, true, "{speed}");
    }
}
