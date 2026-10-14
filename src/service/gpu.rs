//! GPU（CUDA）で鍵を探す [`Searcher`]（feature `cuda`、`--gpu`）
//!
//! シードの生成から文字列の判定までを GPU のカーネル（`gpu/kernel.cu`）で行い、CPU は起動と結果の確認だけをする。
//!
//! - カーネルの設定（表の幅 W、逆元をまとめる鍵の数、体演算のリムの数。[`GpuOptions`]）ごとに、compute_75（Turing）向けの
//!   PTX（`gpu/ptx/`）を埋め込んでおく（[`EMBEDDED`]）。起動時にドライバがその GPU 向けに変換する
//!   （変換結果はドライバがキャッシュするので2回目からは速い）。要るのは NVIDIA のドライバ（libcuda / nvcuda.dll）
//!   だけで、CUDA Toolkit は要らない。PTX は `cargo make ptx` で kernel.cu から作る
//! - 速い設定は GPU によって違うので、探す前に埋め込んだ設定を全部この GPU で短く測り、一番速いものを使う
//!   （[`GpuSetting::Fastest`]）。`--gpu-width`・`--gpu-batch`・`--gpu-limbs` で決めることもでき、埋め込んでいない設定や、
//!   ドライバが古くて PTX を読めないときは、NVRTC（libnvrtc）があればその GPU 向けにコンパイルする
//! - 読み込んだカーネルは、まず少しの鍵を計算させて CPU（ed25519-dalek）と比べ、合わなければその設定は使わない
//!   （[`GpuError::SelfCheck`]。測って選ぶときは飛ばして、状態の表示に出す）
//! - CUDA のライブラリは実行時に読み込むので、ビルドに CUDA は要らない
//! - シードは `ChaCha20(鍵 = 起動ごとの乱数, ブロック番号 = 鍵の通し番号 / 2)` の 64 バイトの、通し番号が
//!   偶数なら前半・奇数なら後半の 32 バイト（1 ブロックで 2 本ぶん）。
//!   当たりは通し番号だけを受け取り、CPU で同じシードを作り直して ed25519-dalek で計算し、
//!   文字列が本当に含まれるかを確かめてから返す（CPU 版と同じく、GPU の計算にバグがあっても誤った鍵は返さない）
//! - 櫛形テーブル（comb.rs と同じ作り方、幅 W）は CPU で作り、各値を 32 ビット × 8 語に詰めて GPU に送る
//!   （1 エントリ 96 バイト。カーネルが読むときに体演算の形に戻す）
//! - 1回の起動（カーネル）が `TARGET_LAUNCH`（0.2 秒）前後で終わるように、1スレッドの繰り返し回数を測りながら調整する。
//!   画面をつないだ GPU（Windows のタイムアウト検出の対象）では、画面が固まらないよう `DISPLAY_LAUNCH`（0.05 秒）にする

use crate::lang::{t, tf};
use crate::model::key::VanityKey;
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
use std::sync::Arc;
use std::time::{Duration, Instant};

/// カーネルのソース
const KERNEL: &str = include_str!("gpu/kernel.cu");

/// 埋め込む PTX の世代。これより新しい GPU でもドライバが変換して動かせる（Turing 〜 Blackwell）
pub const PTX_ARCH: &str = "compute_75";

/// 表の幅 W の既定（`--check` のように測らないとき）
pub const DEFAULT_WIDTH: u32 = 12;

/// 1スレッドが1回にまとめて計算する鍵の数（逆元をまとめる数）の既定（`--check` のように測らないとき）
pub const DEFAULT_BATCH: u32 = 32;

/// 体演算のリムの数の既定（`--check` のように測らないとき）
pub const DEFAULT_LIMBS: u32 = 10;

/// 選べる体演算のリムの数（kernel.cu の MEIKEY_LIMBS。10 = 25.5 ビット × 10、8 = 32 ビット × 8）
pub const LIMBS: [u32; 2] = [10, 8];

/// 選べる表の幅 W（kernel.cu の MEIKEY_W の範囲）
pub const WIDTHS: std::ops::RangeInclusive<u32> = 2..=20;

/// 選べる、逆元をまとめる鍵の数（kernel.cu の MEIKEY_BATCH の範囲）
pub const BATCHES: std::ops::RangeInclusive<u32> = 1..=64;

/// GPU のカーネルの設定
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuOptions {
    /// 表の幅 W（大きいほど点の足し算が減り、表は大きくなる。W = 12 で約 4.1 MB、16 で約 50 MB、20 で約 650 MB）
    pub width: u32,
    /// 1スレッドが1回にまとめて計算する鍵の数（大きいほど逆元の分担が減り、1スレッドが使うメモリは増える）
    pub batch: u32,
    /// 体演算のリムの数（10 = 25.5 ビット × 10 リムを 64 ビットの積和で、8 = 32 ビット × 8 リムを桁上がりつきの
    /// 命令で。どちらが速いかは GPU で違う）
    pub limbs: u32,
}

impl Default for GpuOptions {
    fn default() -> Self {
        GpuOptions::new(DEFAULT_WIDTH, DEFAULT_BATCH, DEFAULT_LIMBS)
    }
}

impl GpuOptions {
    pub const fn new(width: u32, batch: u32, limbs: u32) -> Self {
        GpuOptions {
            width,
            batch,
            limbs,
        }
    }

    /// kernel.cu が受け付ける設定か
    fn in_range(&self) -> bool {
        WIDTHS.contains(&self.width) && BATCHES.contains(&self.batch) && LIMBS.contains(&self.limbs)
    }

    /// 表の大きさ（バイト）
    pub fn table_bytes(&self) -> usize {
        let (rows, cols) = rows_cols(self.width as usize);
        rows * cols * ENTRY_WORDS * 4
    }

    /// 埋め込む PTX のファイル名（`gpu/ptx/` の中）
    pub fn ptx_file(&self) -> String {
        format!("w{}_b{}_l{}.ptx", self.width, self.batch, self.limbs)
    }
}

impl fmt::Display for GpuOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&tf!(
            "W = {}・{} 鍵・{} リム",
            "W = {}, {} keys, {} limbs",
            self.width,
            self.batch,
            self.limbs
        ))
    }
}

/// PTX を埋め込んで、探す前に測る設定（表の幅 12・14・16・18・20 × まとめる鍵の数 32・64 × リムの数 10・8）。
/// 先頭は既定の設定（build.rs の VARIANTS も同じ並び）
pub const EMBEDDED: [GpuOptions; 20] = [
    GpuOptions::new(12, 32, 10),
    GpuOptions::new(12, 32, 8),
    GpuOptions::new(12, 64, 10),
    GpuOptions::new(12, 64, 8),
    GpuOptions::new(14, 32, 10),
    GpuOptions::new(14, 32, 8),
    GpuOptions::new(14, 64, 10),
    GpuOptions::new(14, 64, 8),
    GpuOptions::new(16, 32, 10),
    GpuOptions::new(16, 32, 8),
    GpuOptions::new(16, 64, 10),
    GpuOptions::new(16, 64, 8),
    GpuOptions::new(18, 32, 10),
    GpuOptions::new(18, 32, 8),
    GpuOptions::new(18, 64, 10),
    GpuOptions::new(18, 64, 8),
    GpuOptions::new(20, 32, 10),
    GpuOptions::new(20, 32, 8),
    GpuOptions::new(20, 64, 10),
    GpuOptions::new(20, 64, 8),
];

/// [`EMBEDDED`] の設定で kernel.cu を compute_75 向けにコンパイルした PTX（`cargo make ptx` で作る）。
/// 1行目に kernel.cu と設定のハッシュ（[`kernel_hash`]）を書いておき、ソースと食い違っていないかをテストで確かめる
const PTXS: [&str; 20] = [
    include_str!("gpu/ptx/w12_b32_l10.ptx"),
    include_str!("gpu/ptx/w12_b32_l8.ptx"),
    include_str!("gpu/ptx/w12_b64_l10.ptx"),
    include_str!("gpu/ptx/w12_b64_l8.ptx"),
    include_str!("gpu/ptx/w14_b32_l10.ptx"),
    include_str!("gpu/ptx/w14_b32_l8.ptx"),
    include_str!("gpu/ptx/w14_b64_l10.ptx"),
    include_str!("gpu/ptx/w14_b64_l8.ptx"),
    include_str!("gpu/ptx/w16_b32_l10.ptx"),
    include_str!("gpu/ptx/w16_b32_l8.ptx"),
    include_str!("gpu/ptx/w16_b64_l10.ptx"),
    include_str!("gpu/ptx/w16_b64_l8.ptx"),
    include_str!("gpu/ptx/w18_b32_l10.ptx"),
    include_str!("gpu/ptx/w18_b32_l8.ptx"),
    include_str!("gpu/ptx/w18_b64_l10.ptx"),
    include_str!("gpu/ptx/w18_b64_l8.ptx"),
    include_str!("gpu/ptx/w20_b32_l10.ptx"),
    include_str!("gpu/ptx/w20_b32_l8.ptx"),
    include_str!("gpu/ptx/w20_b64_l10.ptx"),
    include_str!("gpu/ptx/w20_b64_l8.ptx"),
];

/// 設定 `options` の埋め込んだ PTX（埋め込んでいない設定なら None）
fn embedded_ptx(options: GpuOptions) -> Option<&'static str> {
    EMBEDDED.iter().position(|&o| o == options).map(|i| PTXS[i])
}

/// カーネルの設定の決め方
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSetting {
    /// 埋め込んだ設定（[`EMBEDDED`]）を全部この GPU で測り、一番速いものを使う
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

/// 表の1エントリの語数（y+x, y-x, 2dxy を 32 ビット × 8 語ずつ。kernel.cu の ENTRY_WORDS）
const ENTRY_WORDS: usize = 24;

/// GPU を使えないときのエラー
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// CUDA のドライバ（libcuda・nvcuda.dll）が無い
    #[error("{}", t!(
        "CUDA のドライバが見つからない（NVIDIA の GPU とドライバが必要。libcuda / nvcuda.dll）",
        "CUDA driver not found (an NVIDIA GPU and its driver are needed; libcuda / nvcuda.dll)"
    ))]
    DriverNotFound,
    /// NVRTC（libnvrtc）が無い（埋め込む PTX を作るとき）
    #[error("{}", t!("libnvrtc が見つからない", "libnvrtc not found"))]
    NvrtcNotFound,
    /// 表の幅・まとめる鍵の数・リムの数が範囲の外
    #[error("{}", tf!(
        "表の幅は {}〜{}、まとめる鍵の数は {}〜{}、リムの数は 10 か 8（{}）",
        "the table width must be {}-{}, the keys per batch {}-{} and the limbs 10 or 8 ({})",
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
    NotEmbedded(GpuOptions),
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
    func: CudaFunction,
    /// 公開鍵を計算して書き出すだけのカーネル（読み込んだときの確認と、テスト用）
    derive: CudaFunction,
    /// 表（幅が同じ設定で共有する。測るときに幅ごとに1回だけ作って送る）
    table: Arc<CudaSlice<u32>>,
    options: GpuOptions,
    /// 起動するブロック数
    grid: u32,
}

/// GPU に送った表（幅ごと）
#[derive(Default)]
struct Tables(Vec<(u32, Arc<CudaSlice<u32>>)>);

impl Tables {
    /// 幅 `width` の表（まだ無ければ作って送る）
    fn get(
        &mut self,
        stream: &Arc<CudaStream>,
        width: u32,
    ) -> Result<Arc<CudaSlice<u32>>, GpuError> {
        if let Some((_, t)) = self.0.iter().find(|(w, _)| *w == width) {
            return Ok(t.clone());
        }
        let table = Arc::new(stream.clone_htod(&table_words(width))?);
        self.0.push((width, table.clone()));
        Ok(table)
    }
}

/// kernel.cu の探し方（MODE_CONTAINS など。model::pattern::Mode の順）
const MODE_CONTAINS: u32 = 0;
const MODE_SUFFIX: u32 = 1;
const MODE_REGEX: u32 = 2;

/// GPU に渡す探す文字列（kernel.cu の pk_match の needle・len・fold・mode）
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
    /// 設定 `options` のカーネルを読み込み、表を送る（`tables` に同じ幅の表があればそれを使う）。
    /// 埋め込んだ PTX があればそれを、無いかドライバが読めなければ NVRTC でこの GPU 向けにコンパイルしたものを使う
    fn load(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        info: &GpuInfo,
        options: GpuOptions,
        tables: &mut Tables,
    ) -> Result<Kernel, GpuError> {
        if !options.in_range() {
            return Err(GpuError::OutOfRange(options));
        }
        let embedded =
            embedded_ptx(options).map(|ptx| ctx.load_module(cudarc::nvrtc::Ptx::from_src(ptx)));
        let module = match embedded {
            Some(Ok(m)) => m,
            // 埋め込んでいない設定か、ドライバが古くて埋め込んだ PTX を読めない: NVRTC でコンパイルする
            other => {
                // SAFETY: ライブラリを読み込めるか試すだけ
                if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
                    return Err(match other {
                        Some(Err(e)) => GpuError::PtxNotLoadable(e),
                        _ => GpuError::NotEmbedded(options),
                    });
                }
                ctx.load_module(compile(info.compute, options)?)?
            }
        };
        let func = module.load_function("meikey_search")?;
        let derive = module.load_function("meikey_derive")?;
        // SM あたりに同時に載るブロック数（レジスタの使用量で決まる）の分だけ起動する
        let per_sm = func
            .occupancy_max_active_blocks_per_multiprocessor(BLOCK, 0, None)
            .unwrap_or(2)
            .max(1);
        let table = tables.get(stream, options.width)?;
        let kernel = Kernel {
            func,
            derive,
            table,
            options,
            grid: info.sms.max(1) * per_sm,
        };
        kernel.self_check(stream)?;
        Ok(kernel)
    }

    /// 1 ブロック分（[`SELF_CHECK_THREADS`] スレッド）の公開鍵を GPU で計算し、そのうち先頭と最後のスレッドの全部と、
    /// ほかのスレッドの1本目を CPU（ed25519-dalek）で計算し直して比べる。シードの作り方の両方で確かめ、
    /// 合わなければ使えない設定（通し番号は奇数から始めて、シードを ChaCha20 のブロックの後半から取る場合も通す）
    fn self_check(&self, stream: &Arc<CudaStream>) -> Result<(), GpuError> {
        let batch = self.options.batch as usize;
        let keys = SELF_CHECK_THREADS as usize * batch;
        for source in [SeedSource::ChaCha20, SeedSource::Counter] {
            let key: [u8; 32] = rand::random();
            let first = (rand::random::<u64>() >> 8) | 1;
            let got = self.derive_on_gpu(stream, &key, source, first, 1, SELF_CHECK_THREADS)?;
            let checked = (0..keys)
                .filter(|i| i % batch == 0 || i / batch == 0 || i / batch == keys / batch - 1);
            for i in checked {
                let want = VanityKey::from_seed(seed(&key, source, first + i as u64));
                if &got.as_chunks::<32>().0[i] != want.public_bytes() {
                    return Err(GpuError::SelfCheck(self.options));
                }
            }
        }
        Ok(())
    }

    /// meikey_derive で、通し番号 `first` から `blocks` × `threads` スレッド分の公開鍵を計算する
    fn derive_on_gpu(
        &self,
        stream: &Arc<CudaStream>,
        key: &[u8; 32],
        source: SeedSource,
        first: u64,
        blocks: u32,
        threads: u32,
    ) -> Result<Vec<u8>, GpuError> {
        let n = (blocks * threads) as usize * self.options.batch as usize;
        let key_dev = stream.clone_htod(&key_words(key))?;
        let counter = u32::from(source == SeedSource::Counter);
        let mut out = stream.alloc_zeros::<u8>(n * 32)?;
        let mut b = stream.launch_builder(&self.derive);
        b.arg(&*self.table)
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

    /// 1回の起動で計算する鍵の数（1スレッドの繰り返し1回あたり）
    fn keys_per_iter(&self) -> u64 {
        u64::from(self.grid) * u64::from(BLOCK) * u64::from(self.options.batch)
    }

    /// 起動に渡すものを GPU に置く。`key` はシードを作る乱数（ChaCha20 の鍵か、通し番号を足す 32 バイト）
    fn prepare(
        &self,
        stream: &Arc<CudaStream>,
        key: &[u8; 32],
        source: SeedSource,
        needle: &Needle,
        stop_on_hit: bool,
    ) -> Result<Launch, GpuError> {
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

    /// 通し番号 `base` から、1スレッドが `iters` 回ずつ計算する起動を1回して、終わるまで待つ。
    /// 当たりがあれば、その通し番号を返す
    fn launch(
        &self,
        stream: &Arc<CudaStream>,
        l: &mut Launch,
        base: u64,
        iters: u32,
    ) -> Result<Option<u64>, GpuError> {
        let mut b = stream.launch_builder(&self.func);
        b.arg(&*self.table)
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
        Ok((found[0] != 0).then_some(found[1]))
    }

    /// この GPU での速度（鍵/秒）。1回 [`MEASURE_LAUNCH`] ほどの起動を [`MEASURE_LAUNCHES`] 回して測る
    /// （最初の1回は、繰り返し回数を決めるための短い起動）
    fn measure(&self, stream: &Arc<CudaStream>) -> Result<f64, GpuError> {
        // GPU で探すのは低セキュリティモード（通し番号のシード）だけなので、それで測る
        let needle = Needle {
            bytes: MEASURE_NEEDLE.to_vec(),
            len: MEASURE_NEEDLE.len() as u32,
            fold: true,
            mode: MODE_CONTAINS,
        };
        let mut l = self.prepare(stream, &[0x5a; 32], SeedSource::Counter, &needle, false)?;
        let start = Instant::now();
        self.launch(stream, &mut l, 0, 1)?;
        let once = start.elapsed().as_secs_f64().max(1e-6);
        let iters = (MEASURE_LAUNCH.as_secs_f64() / once).clamp(1.0, f64::from(1u32 << 16)) as u32;
        let start = Instant::now();
        for i in 0..MEASURE_LAUNCHES {
            let base = u64::from(i + 1) * self.keys_per_iter() * u64::from(iters);
            self.launch(stream, &mut l, base, iters)?;
        }
        let keys = self.keys_per_iter() * u64::from(iters) * u64::from(MEASURE_LAUNCHES);
        Ok(keys as f64 / start.elapsed().as_secs_f64().max(1e-9))
    }
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

/// GPU で探す [`Searcher`]
pub struct Gpu {
    stream: Arc<CudaStream>,
    kernel: Kernel,
    info: GpuInfo,
    chosen: Chosen,
    /// 1回の起動にかける時間の目安
    target: Duration,
}

impl Gpu {
    /// `ordinal` 番目の GPU を使えるようにする（カーネルの読み込みと表の転送をする。
    /// [`GpuSetting::Fastest`] なら、埋め込んだ設定を全部測ってから一番速いものを使い、
    /// 測り始める前に `on_measure` を呼ぶ）
    pub fn new(
        ordinal: usize,
        setting: GpuSetting,
        on_measure: impl FnOnce(),
    ) -> Result<Gpu, GpuError> {
        // SAFETY: ライブラリを読み込めるか試すだけ
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(GpuError::DriverNotFound);
        }
        let ctx = CudaContext::new(ordinal)?;
        let info = GpuInfo {
            name: ctx.name()?,
            compute: ctx.compute_capability()?,
            sms: ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?
                as u32,
            memory: ctx.total_mem()?,
        };
        // 画面をつないだ GPU（カーネルの実行時間に制限がある）なら起動を短くする
        let target =
            match ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT) {
                Ok(1) => DISPLAY_LAUNCH,
                _ => TARGET_LAUNCH,
            };
        let stream = ctx.default_stream();
        let (kernel, chosen) = match setting {
            GpuSetting::Fixed(options) => (
                Kernel::load(&ctx, &stream, &info, options, &mut Tables::default())?,
                Chosen::Fixed,
            ),
            GpuSetting::Default => (
                Kernel::load(
                    &ctx,
                    &stream,
                    &info,
                    GpuOptions::default(),
                    &mut Tables::default(),
                )?,
                Chosen::Default,
            ),
            GpuSetting::Fastest => {
                on_measure();
                let mut rejected = Vec::new();
                let mut tables = Tables::default();
                let (kernel, results) = fastest(
                    &EMBEDDED,
                    |options| {
                        let k = Kernel::load(&ctx, &stream, &info, options, &mut tables);
                        if let Err(GpuError::SelfCheck(o)) = &k {
                            rejected.push(*o);
                        }
                        k
                    },
                    |k| k.measure(&stream),
                )?;
                (kernel, Chosen::Fastest { results, rejected })
            }
        };
        Ok(Gpu {
            stream,
            kernel,
            info,
            chosen,
            target,
        })
    }

    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// 使っている設定
    pub fn options(&self) -> GpuOptions {
        self.kernel.options
    }

    fn search(
        &self,
        config: &SearchConfig,
        on_progress: &mut dyn Progress,
    ) -> Result<Outcome, GpuError> {
        let (stream, kernel) = (&self.stream, &self.kernel);
        // シードを作る ChaCha20 の鍵（起動ごとに新しい乱数）
        let key: [u8; 32] = rand::random();
        let needle = Needle::new(config.pattern.as_ref());
        // ベンチマークでは当たっても止まらない
        let mut l = kernel.prepare(stream, &key, config.seeds, &needle, !config.bench)?;

        let start = Instant::now();
        let mut tries = 0u64;
        let mut iters = 1u32;
        let mut next_report = start + config.progress_interval;
        loop {
            let launch = Instant::now();
            let found = kernel.launch(stream, &mut l, tries, iters)?;
            let took = launch.elapsed();
            tries += kernel.keys_per_iter() * u64::from(iters);
            let stats = Stats {
                tries,
                elapsed: start.elapsed(),
            };

            if let Some(index) = found
                && !config.bench
            {
                let key = self.confirm(&key, index, config);
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
            // 1回の起動が self.target 前後になるよう、繰り返し回数を調整する（急に増やしすぎない）
            let scale = self.target.as_secs_f64() / took.as_secs_f64().max(1e-6);
            iters = ((f64::from(iters) * scale.clamp(0.5, 4.0)) as u32).clamp(1, 1 << 16);
        }
    }

    /// 通し番号 `index` の鍵を CPU で作り直し、文字列が含まれることを確かめる
    fn confirm(&self, key: &[u8; 32], index: u64, config: &SearchConfig) -> VanityKey {
        let found = VanityKey::from_seed(seed(key, config.seeds, index));
        if let Some(p) = &config.pattern {
            // 一致しなければ GPU の計算か判定が間違っている（バグ）ので止める
            assert!(
                p.is_match(&found.public_b64()),
                "{}",
                tf!(
                    "内部エラー: GPU（{}）の結果が ed25519-dalek・総当たりと不一致",
                    "internal error: the result of the GPU ({}) differs from ed25519-dalek with a naive search",
                    self.info.name
                )
            );
        }
        found
    }
}

impl Searcher for Gpu {
    fn run(&self, config: &SearchConfig, on_progress: &mut dyn Progress) -> Outcome {
        self.search(config, on_progress).unwrap_or_else(|e| {
            panic!(
                "{}",
                tf!("GPU での探索に失敗: {e}", "GPU search failed: {e}")
            )
        })
    }

    fn status(&self, _config: &SearchConfig) -> Vec<(&'static str, String)> {
        let (i, k) = (&self.info, &self.kernel);
        let how = match &self.chosen {
            Chosen::Fastest { results, .. } => tf!(
                "{} 通りを測って一番速いもの",
                "the fastest of {} measured",
                results.len()
            ),
            Chosen::Fixed => t!("指定", "given").into(),
            Chosen::Default => t!(
                "既定。--check では測らない",
                "default; not measured with --check"
            )
            .into(),
        };
        let mut rows = vec![
            ("GPU", i.name.clone()),
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

/// 通し番号 `index` のシード（kernel.cu と同じ。ChaCha20 か、`key` の先頭 8 バイトに通し番号を足したもの）
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
fn compile(compute: (i32, i32), options: GpuOptions) -> Result<cudarc::nvrtc::Ptx, GpuError> {
    let exact = format!("sm_{}{}", compute.0, compute.1);
    let mut errors = Vec::new();
    for arch in [exact.as_str(), "compute_90", "compute_80", PTX_ARCH] {
        match compile_for(arch, options) {
            Ok(ptx) => return Ok(ptx),
            Err(e) => errors.push(format!("{arch}: {}", compile_error(&e))),
        }
    }
    Err(GpuError::Compile(errors.join("\n")))
}

/// NVRTC でカーネルを `arch`（sm_86・compute_75 など）向けの PTX にする
fn compile_for(arch: &str, options: GpuOptions) -> Result<cudarc::nvrtc::Ptx, CompileError> {
    compile_ptx_with_opts(
        KERNEL,
        CompileOptions {
            options: vec![
                format!("--gpu-architecture={arch}"),
                format!("-DMEIKEY_W={}", options.width),
                format!("-DMEIKEY_BATCH={}", options.batch),
                format!("-DMEIKEY_LIMBS={}", options.limbs),
            ],
            ..Default::default()
        },
    )
}

/// kernel.cu（改行は LF にそろえる）とコンパイルの設定のハッシュ（埋め込む PTX の1行目に書く）
pub fn kernel_hash(options: GpuOptions) -> String {
    use sha2::{Digest, Sha256};
    let source = KERNEL.replace('\r', "");
    let digest = Sha256::digest(format!(
        "{source}\n// {PTX_ARCH} W={} BATCH={} LIMBS={}",
        options.width, options.batch, options.limbs
    ));
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// 設定 `options` の埋め込む PTX（`gpu/ptx/` のファイルの中身）を NVRTC で作る（`cargo make ptx`）。
/// 互換性のため、なるべく古い NVRTC で作る（PTX の版はコンパイルした NVRTC の版で決まり、それより古いドライバでは読めない）
pub fn portable_ptx(options: GpuOptions) -> Result<String, GpuError> {
    // SAFETY: ライブラリを読み込めるか試すだけ
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(GpuError::NvrtcNotFound);
    }
    let ptx = compile_for(PTX_ARCH, options)
        .map_err(|e| GpuError::Compile(format!("{PTX_ARCH}: {}", compile_error(&e))))?;
    Ok(format!(
        // PTX は ASCII だけ（ptxas は ASCII 以外の文字を受け付けない）
        "// meikey: sha256(kernel.cu, W={}, BATCH={}, LIMBS={}) = {} (regenerate: cargo make ptx)\n{}",
        options.width,
        options.batch,
        options.limbs,
        kernel_hash(options),
        ptx.to_src()
    ))
}

fn compile_error(e: &CompileError) -> String {
    match e {
        CompileError::CompileError { log, .. } => log.to_string_lossy().into_owned(),
        other => format!("{other:?}"),
    }
}

/// 幅 `width` の表を GPU 用（1エントリ = y+x, y-x, 2dxy を 0 以上 p 未満にして 32 ビット × 8 語ずつ）に並べる。
/// 1 行目の 3 つ目は 2dxy ではなく 2xy にする（最初の点を拡張座標にするとき、T = 2xy を掛け算なしで使う）。
/// 行ごとに別のスレッドで作る（W = 20 で約 650 MB あり、1スレッドでは数秒かかる）
fn table_words(width: u32) -> Vec<u32> {
    let (_, cols) = rows_cols(width as usize);
    let mut words = vec![0u32; GpuOptions::new(width, 1, 10).table_bytes() / 4];
    let half = Fe([2, 0, 0, 0, 0]).invert();
    std::thread::scope(|s| {
        for (i, row) in words.chunks_mut(cols * ENTRY_WORDS).enumerate() {
            s.spawn(move || {
                let row = row.as_chunks_mut::<ENTRY_WORDS>().0;
                for_each_entry_in_row(width as usize, i, |k, mut entry| {
                    if i == 0 {
                        // (y+x)^2 - (y-x)^2 = 4xy なので、2xy = ((y+x)^2 - (y-x)^2) / 2
                        let [ypx, ymx, _] = entry;
                        entry[2] = (ypx * ypx - ymx * ymx) * half;
                    }
                    for (f, fe) in entry.into_iter().enumerate() {
                        for (j, w) in fe.to_bytes().as_chunks::<4>().0.iter().enumerate() {
                            row[k][8 * f + j] = u32::from_le_bytes(*w);
                        }
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
    #[cfg(meikey_gpu_host)]
    use crate::property::check;
    #[cfg(meikey_gpu_host)]
    use crate::service::derive::comb::{ROWS, signed_digits};
    #[cfg(meikey_gpu_host)]
    use crate::service::derive::field::Fe;
    use crate::service::search::{Both, Parallel};
    use crate::simd_support::available;
    use cudarc::driver::sys::is_culib_present as driver_present;
    use cudarc::nvrtc::sys::is_culib_present as nvrtc_present;
    #[cfg(meikey_gpu_host)]
    use curve25519_dalek::scalar::clamp_integer;
    #[cfg(meikey_gpu_host)]
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
        fn meikey_test_digits(s: *const u8, out: *mut i32);
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
    type DeriveFn = unsafe extern "C" fn(*const u32, *const u32, u32, u64, *mut u8);
    #[cfg(meikey_gpu_host)]
    type SearchFn = unsafe extern "C" fn(
        u64,
        *const u32,
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

    /// 1つの設定で CPU 向けにコンパイルしたカーネル
    #[cfg(meikey_gpu_host)]
    struct HostKernel {
        options: GpuOptions,
        derive: DeriveFn,
        search: SearchFn,
        /// 32 ビット × 8 語で渡す体演算（kernel.cu の fe_op_words）
        fe_op: unsafe extern "C" fn(u32, *const u32, *const u32, *mut u8),
        batch: unsafe extern "C" fn() -> u32,
        width: unsafe extern "C" fn() -> u32,
        limbs: unsafe extern "C" fn() -> u32,
    }

    // 設定ごとの関数の宣言と HOSTS（build.rs が作る）
    #[cfg(meikey_gpu_host)]
    include!(concat!(env!("OUT_DIR"), "/gpu_hosts.rs"));

    /// 幅 `width` の表（幅ごとに1回だけ作る）
    #[cfg(meikey_gpu_host)]
    fn table(width: u32) -> &'static [u32] {
        static TABLES: [std::sync::OnceLock<Vec<u32>>; 21] =
            [const { std::sync::OnceLock::new() }; 21];
        TABLES[width as usize].get_or_init(|| table_words(width))
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

    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn options_match_kernel() {
        assert_eq!(
            HOSTS.iter().map(|h| h.options).collect::<Vec<_>>(),
            EMBEDDED.to_vec()
        );
        for h in &HOSTS {
            // SAFETY: 引数なしの関数
            let got = unsafe { GpuOptions::new((h.width)(), (h.batch)(), (h.limbs)()) };
            assert_eq!(got, h.options);
        }
        // meikey_test_* は既定の設定（先頭）
        assert_eq!(HOSTS[0].options, GpuOptions::default());
        // 1ブロックのスレッド数（カーネルのレジスタの上限をこれで決めている）
        // SAFETY: 引数なしの関数
        assert_eq!(unsafe { meikey_test_block() }, BLOCK);
        // 既定の幅は CPU の表（comb::W）と同じ（下の digits_match_comb で comb の桁と比べるため）
        assert_eq!(DEFAULT_WIDTH as usize, crate::service::derive::comb::W);
    }

    /// 表の大きさ（W = 12 で約 4.1 MB、16 で約 50 MB、20 で約 650 MB）
    #[rstest]
    #[case::default(GpuOptions::default(), 21 * 2049 * 96)]
    #[case::wide(GpuOptions::new(16, 32, 8), 16 * 32769 * 96)]
    #[case::widest(GpuOptions::new(20, 64, 10), 13 * 524289 * 96)]
    fn table_bytes(#[case] options: GpuOptions, #[case] want: usize) {
        assert_eq!(options.table_bytes(), want);
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

    /// シードは rand の ChaCha20Rng（ストリーム 0）のブロック `index / 2` の前半（偶数）か後半（奇数）
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

    /// 桁の分解は comb.rs と同じ
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn digits_match_comb() {
        check(256, any::<[u8; 32]>().prop_map(clamp_integer), |s| {
            let mut got = [0i32; ROWS];
            // SAFETY: s は 32 バイト、out は ROWS 個
            unsafe { meikey_test_digits(s.as_ptr(), got.as_mut_ptr()) };
            prop_assert_eq!(got, signed_digits(&s).map(i32::from));
            Ok(())
        });
    }

    /// 体の乗算・逆元は field.rs と同じ値
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

    /// 体演算（32 ビット × 8 語で渡す版）は、どの設定でも mod p で正しく、結果は p 未満。8 リムは 2^256 未満の
    /// 任意の値（p 以上・2^255 以上を含む）を受け付けるので、境界の値も混ぜる。op: 0 = 加算、1 = 減算、
    /// 2 = 乗算、3 = 2乗、4 = 逆元（0 の逆元は 0）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn fe_op_words_matches_bigint(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)]
        variant: usize,
        #[values(0, 1, 2, 3, 4)] op: u32,
    ) {
        use num_bigint::BigUint;
        let h = &HOSTS[variant];
        let p = (BigUint::from(1u8) << 255u32) - 19u32;
        let full = (BigUint::from(1u8) << 256u32) - 1u32;
        // 10 リムは 2^255 未満の値だけ（表の値は p 未満）
        let max = if h.options.limbs == 8 {
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
        check(512, strategy, |(ra, rb, ea, eb)| {
            let (a, b) = (pick(ra, ea), pick(rb, eb));
            let want = match op {
                0 => (&a + &b) % &p,
                1 => (&a + &p * 4u32 - &b) % &p,
                2 => (&a * &b) % &p,
                3 => (&a * &a) % &p,
                _ => (&a % &p).modpow(&(&p - 2u32), &p),
            };
            let mut got = [0u8; 32];
            // SAFETY: a・b は 8 語、out は 32 バイト
            unsafe { (h.fe_op)(op, words(&a).as_ptr(), words(&b).as_ptr(), got.as_mut_ptr()) };
            prop_assert_eq!(
                BigUint::from_bytes_le(&got),
                want,
                "{} op {} {:x} {:x}",
                h.options,
                op,
                a,
                b
            );
            Ok(())
        });
    }

    /// まとめて計算した公開鍵は、同じシードから ed25519-dalek で作った鍵と一致する（埋め込む設定すべて）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn derive_batch_matches_dalek(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)]
        variant: usize,
        #[values(SeedSource::ChaCha20, SeedSource::Counter)] source: SeedSource,
    ) {
        let h = &HOSTS[variant];
        let (table, batch, derive) = (table(h.options.width), h.options.batch, h.derive);
        check(16, (any::<[u8; 32]>(), any::<u64>()), |(key, first)| {
            let first = first >> 8; // 足しても桁あふれしないように
            let mut got = vec![0u8; 32 * batch as usize];
            // SAFETY: 表はその設定の行数·列数のエントリ、key は 8 語、out は batch·32 バイト
            unsafe {
                derive(
                    table.as_ptr(),
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
            let mut b = [0u8; crate::model::key::PUBLIC_B64_LEN];
            crate::model::key::encode_public_b64(pk, &mut b);
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
        check(4096, cases, |(seed, start, len, flips, edit)| {
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

    /// 埋め込んだ PTX は今の kernel.cu とその設定から作ったもの（違えば `cargo make ptx` で作り直す）
    #[rstest]
    fn embedded_ptx_matches_kernel_source(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)] i: usize,
    ) {
        let (options, ptx) = (EMBEDDED[i], PTXS[i]);
        let first = ptx.lines().next().unwrap_or_default();
        assert_eq!(
            first.contains(&kernel_hash(options)),
            true,
            "{} が kernel.cu と食い違っている（cargo make ptx で作り直す）: {first}",
            options.ptx_file()
        );
        assert_eq!(
            ptx.contains(".target sm_75") && ptx.contains(".entry meikey_search"),
            true
        );
        assert_eq!(embedded_ptx(options), Some(ptx));
    }

    /// 埋め込んだ設定はどれも選べる範囲にあり、重ならず、測らないときの既定も含む
    #[rstest]
    fn embedded_options_are_valid() {
        for (i, o) in EMBEDDED.iter().enumerate() {
            assert_eq!(o.in_range(), true, "{o}");
            assert_eq!(EMBEDDED[..i].contains(o), false, "{o} が重なっている");
        }
        assert_eq!(EMBEDDED.contains(&GpuOptions::default()), true);
        assert_eq!(embedded_ptx(GpuOptions::new(13, 32, 10)), None);
        assert_eq!(GpuOptions::new(12, 32, 9).in_range(), false);
    }

    /// 測った中で一番速い設定を選ぶ。読み込めない・測れない設定は飛ばし、全部だめなら最初のエラー。
    /// `speeds[i]` は i 番目の設定の速さ（Err なら読み込めない）、`broken` は測ると失敗する設定
    /// GpuError の日本語・英語の表示
    #[rstest]
    #[case::driver(
        GpuError::DriverNotFound,
        "CUDA のドライバが見つからない（NVIDIA の GPU とドライバが必要。libcuda / nvcuda.dll）",
        "CUDA driver not found (an NVIDIA GPU and its driver are needed; libcuda / nvcuda.dll)"
    )]
    #[case::nvrtc(
        GpuError::NvrtcNotFound,
        "libnvrtc が見つからない",
        "libnvrtc not found"
    )]
    #[case::range(
        GpuError::OutOfRange(GpuOptions::new(21, 32, 10)),
        "表の幅は 2〜20、まとめる鍵の数は 1〜64、リムの数は 10 か 8（W = 21・32 鍵・10 リム）",
        "the table width must be 2-20, the keys per batch 1-64 and the limbs 10 or 8 (W = 21, 32 keys, 10 limbs)"
    )]
    #[case::not_embedded(
        GpuError::NotEmbedded(GpuOptions::new(13, 8, 8)),
        "W = 13・8 鍵・8 リム は埋め込んでいない設定なので NVRTC（libnvrtc）が要る（LD_LIBRARY_PATH / PATH に追加）",
        "W = 13, 8 keys, 8 limbs is not built in, so NVRTC (libnvrtc) is needed (add it to LD_LIBRARY_PATH / PATH)"
    )]
    #[case::self_check(
        GpuError::SelfCheck(GpuOptions::new(16, 64, 8)),
        "W = 16・64 鍵・8 リム の GPU の計算が CPU と合わない（この設定は使えない）",
        "the GPU results of W = 16, 64 keys, 8 limbs differ from the CPU (this setting cannot be used)"
    )]
    #[case::no_settings(GpuError::NoSettings, "測る設定が無い", "no settings to measure")]
    #[case::compile(GpuError::Compile("sm_90: log".into()), "カーネルのコンパイルに失敗:\nsm_90: log", "failed to compile the kernel:\nsm_90: log")]
    fn gpu_error_display(#[case] e: GpuError, #[case] ja: &str, #[case] en: &str) {
        assert_eq!(e.to_string(), ja);
        crate::lang::set(crate::lang::Lang::En);
        assert_eq!(e.to_string(), en);
    }

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
            .map(|i| GpuOptions::new(i + 2, 1, 10))
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
        let candidates: Vec<GpuOptions> = (0..3).map(|i| GpuOptions::new(i + 2, 1, 10)).collect();
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
        let gpu = Gpu::new(0, GpuSetting::Fixed(GpuOptions::default()), || {})
            .expect("GPU を使えるようにできる");
        Some(OnGpu { gpu, lock })
    }

    /// カーネルは Turing（T4）から Blackwell（RTX PRO 6000）まで、各世代向けに、選べる設定の端の値でも
    /// コンパイルできる
    #[rstest]
    fn kernel_compiles_for_each_generation(
        #[values((7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (12, 0))] compute: (i32, i32),
        #[values(
            GpuOptions::default(),
            GpuOptions::new(*WIDTHS.start(), *BATCHES.start(), 10),
            GpuOptions::new(*WIDTHS.end(), *BATCHES.end(), 8)
        )]
        options: GpuOptions,
        nvrtc: Option<()>,
    ) {
        if nvrtc.is_none() {
            return;
        }
        let ptx = compile(compute, options).unwrap_or_else(|e| panic!("{e}"));
        let src = ptx.to_src();
        let want = format!(".target sm_{}{}", compute.0, compute.1);
        assert_eq!(src.contains(&want), true, "{want} の PTX になっていない");
        assert_eq!(
            src.contains(".entry meikey_search") && src.contains(".entry meikey_derive"),
            true
        );
    }

    /// GPU で計算した公開鍵は、同じシードから ed25519-dalek で作った鍵と一致する（埋め込んだ設定すべて。
    /// 通し番号は奇数から始め、シードを ChaCha20 のブロックの後半から取る場合も通す）
    #[rstest]
    fn gpu_derive_matches_dalek(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)] i: usize,
        gpu: Option<OnGpu>,
    ) {
        let Some(OnGpu { lock: _lock, .. }) = gpu else {
            return;
        };
        let gpu =
            Gpu::new(0, GpuSetting::Fixed(EMBEDDED[i]), || {}).expect("GPU を使えるようにできる");
        let key: [u8; 32] = random();
        let first = (1 << 40) + 1;
        let got = gpu
            .kernel
            .derive_on_gpu(&gpu.stream, &key, SeedSource::ChaCha20, first, 8, BLOCK)
            .unwrap();
        for (i, pk) in got.as_chunks::<32>().0.iter().enumerate() {
            let want = VanityKey::from_seed(seed(&key, SeedSource::ChaCha20, first + i as u64));
            assert_eq!(pk, want.public_bytes(), "#{i}");
        }
    }

    /// 設定を測って選ぶと、埋め込んだ設定を全部測って、その中で一番速いものを使う。その設定でも鍵が見つかる
    #[rstest]
    fn gpu_uses_the_fastest_setting(gpu: Option<OnGpu>) {
        let Some(OnGpu { lock: _lock, .. }) = gpu else {
            return;
        };
        let mut notified = false;
        let gpu =
            Gpu::new(0, GpuSetting::Fastest, || notified = true).expect("GPU を使えるようにできる");
        assert_eq!(notified, true, "測る前に知らせる");
        let Chosen::Fastest { results, rejected } = &gpu.chosen else {
            panic!("測っていない");
        };
        assert_eq!(rejected, &Vec::new(), "計算が CPU と合わない設定がある");
        assert_eq!(
            results.iter().map(|r| r.0).collect::<Vec<_>>(),
            EMBEDDED.to_vec()
        );
        let best = results.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
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

    /// カーネル全体（CPU で真似たもの）: 見つけた通し番号のシードの鍵は文字列を含み、
    /// それより前の番号の鍵は含まない（1スレッドなら最初の当たりを返す）。stop_on_hit なら途中で止まる
    /// （埋め込む設定すべて）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn search_finds_first_hit(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)]
        variant: usize,
        #[values(SeedSource::ChaCha20, SeedSource::Counter)] source: SeedSource,
    ) {
        let h = &HOSTS[variant];
        let (table, search) = (table(h.options.width), h.search);
        let key: [u8; 32] = from_fn(|i| i as u8 * 7);
        let p = Pattern::new("ab", false).unwrap();
        let needle = b"ab";
        let mut result = [0u64; 2];
        // SAFETY: 表・鍵・文字列・結果の大きさは kernel.cu の前提どおり
        unsafe {
            search(
                1,
                table.as_ptr(),
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
        assert_eq!(
            p.is_match(&VanityKey::from_seed(seed(&key, source, hit)).public_b64()),
            true
        );
        for i in 0..hit {
            assert_eq!(
                p.is_match(&VanityKey::from_seed(seed(&key, source, i)).public_b64()),
                false,
                "{i} が先に当たる"
            );
        }
    }

    /// 末尾・正規表現でも、カーネル全体（CPU で真似たもの）が最初の当たりを返す（幅・リムの違う設定で）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn search_finds_first_hit_with_mode(
        #[values(0, 7, 19)] variant: usize,
        #[values((Mode::Suffix, "a"), (Mode::Suffix, "Q9"), (Mode::Regex, "^[A-P][0-9]"), (Mode::Regex, "x.y|z$"))]
        pattern: (Mode, &str),
    ) {
        let h = &HOSTS[variant];
        let (table, search) = (table(h.options.width), h.search);
        let key: [u8; 32] = from_fn(|i| i as u8 * 11);
        let p = Pattern::with_mode(pattern.1, false, pattern.0).unwrap();
        let n = Needle::new(Some(&p));
        let bytes = [&n.bytes[..], &[0; 64]].concat();
        let mut result = [0u64; 2];
        // SAFETY: 表・鍵・文字列・結果の大きさは kernel.cu の前提どおり
        unsafe {
            search(
                1,
                table.as_ptr(),
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
            "3 万鍵以上のうちにほぼ確実に出る（Q9 は 2048 鍵に 1 回）"
        );
        let hit = result[1];
        let matches =
            |i| p.is_match(&VanityKey::from_seed(seed(&key, SeedSource::Counter, i)).public_b64());
        assert_eq!(matches(hit), true);
        assert_eq!((0..hit).any(matches), false, "{hit} より前に当たりがある");
    }
}
