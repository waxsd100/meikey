//! GPU（CUDA）で鍵を探す [`Searcher`]（feature `cuda`、`--gpu`）
//!
//! シードの生成から文字列の判定までを GPU のカーネル（`gpu/kernel.cu`）で行い、CPU は起動と結果の確認だけをする。
//!
//! - カーネルの設定（表の幅 W と、逆元をまとめる鍵の数。[`GpuOptions`]）ごとに、compute_75（Turing）向けの
//!   PTX（`gpu/ptx/`）を埋め込んでおく（[`EMBEDDED`]）。起動時にドライバがその GPU 向けに変換する
//!   （変換結果はドライバがキャッシュするので2回目からは速い）。要るのは NVIDIA のドライバ（libcuda / nvcuda.dll）
//!   だけで、CUDA Toolkit は要らない。PTX は `cargo make ptx` で kernel.cu から作る
//! - 速い設定は GPU によって違うので、探す前に埋め込んだ設定を全部この GPU で短く測り、一番速いものを使う
//!   （[`GpuSetting::Fastest`]）。`--gpu-width`・`--gpu-batch` で決めることもでき、埋め込んでいない設定や、
//!   ドライバが古くて PTX を読めないときは、NVRTC（libnvrtc）があればその GPU 向けにコンパイルする
//! - CUDA のライブラリは実行時に読み込むので、ビルドに CUDA は要らない
//! - シードは `ChaCha20(鍵 = 起動ごとの乱数, ブロック番号 = 鍵の通し番号)` の先頭 32 バイト。
//!   当たりは通し番号だけを受け取り、CPU で同じシードを作り直して ed25519-dalek で計算し、
//!   文字列が本当に含まれるかを確かめてから返す（CPU 版と同じく、GPU の計算にバグがあっても誤った鍵は返さない）
//! - 櫛形テーブル（comb.rs と同じ作り方、幅 W）は CPU で作って 25.5 ビット × 10 リムにして GPU に送る
//! - 1回の起動（カーネル）が `TARGET_LAUNCH`（0.2 秒）前後で終わるように、1スレッドの繰り返し回数を測りながら調整する。
//!   画面をつないだ GPU（Windows のタイムアウト検出の対象）では、画面が固まらないよう `DISPLAY_LAUNCH`（0.05 秒）にする

use crate::model::key::VanityKey;
use crate::model::search::{Outcome, SearchConfig, Stats};
use crate::service::derive::comb::{entries_with_width, rows_cols};
use crate::service::search::{Progress, Searcher};
#[cfg(test)]
use cudarc::driver::CudaModule;
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
pub const DEFAULT_BATCH: u32 = 16;

/// 選べる表の幅 W（kernel.cu の MEIKEY_W の範囲）
pub const WIDTHS: std::ops::RangeInclusive<u32> = 2..=16;

/// 選べる、逆元をまとめる鍵の数（kernel.cu の MEIKEY_BATCH の範囲）
pub const BATCHES: std::ops::RangeInclusive<u32> = 1..=64;

/// GPU のカーネルの設定
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuOptions {
    /// 表の幅 W（大きいほど点の足し算が減り、表は大きくなる。W = 12 で約 5.8 MB、16 で約 67 MB）
    pub width: u32,
    /// 1スレッドが1回にまとめて計算する鍵の数（大きいほど逆元の分担が減り、1スレッドが使うメモリは増える）
    pub batch: u32,
}

impl Default for GpuOptions {
    fn default() -> Self {
        GpuOptions {
            width: DEFAULT_WIDTH,
            batch: DEFAULT_BATCH,
        }
    }
}

impl GpuOptions {
    pub const fn new(width: u32, batch: u32) -> Self {
        GpuOptions { width, batch }
    }

    /// 表の大きさ（バイト）
    pub fn table_bytes(&self) -> usize {
        let (rows, cols) = rows_cols(self.width as usize);
        rows * cols * ENTRY_WORDS * 4
    }

    /// 埋め込む PTX のファイル名（`gpu/ptx/` の中）
    pub fn ptx_file(&self) -> String {
        format!("w{}_b{}.ptx", self.width, self.batch)
    }
}

impl fmt::Display for GpuOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "W = {}・{} 鍵", self.width, self.batch)
    }
}

/// PTX を埋め込んで、探す前に測る設定（表の幅 12・14・16 × まとめる鍵の数 16・32）
pub const EMBEDDED: [GpuOptions; 6] = [
    GpuOptions::new(12, 16),
    GpuOptions::new(12, 32),
    GpuOptions::new(14, 16),
    GpuOptions::new(14, 32),
    GpuOptions::new(16, 16),
    GpuOptions::new(16, 32),
];

/// [`EMBEDDED`] の設定で kernel.cu を compute_75 向けにコンパイルした PTX（`cargo make ptx` で作る）。
/// 1行目に kernel.cu と設定のハッシュ（[`kernel_hash`]）を書いておき、ソースと食い違っていないかをテストで確かめる
const PTXS: [&str; 6] = [
    include_str!("gpu/ptx/w12_b16.ptx"),
    include_str!("gpu/ptx/w12_b32.ptx"),
    include_str!("gpu/ptx/w14_b16.ptx"),
    include_str!("gpu/ptx/w14_b32.ptx"),
    include_str!("gpu/ptx/w16_b16.ptx"),
    include_str!("gpu/ptx/w16_b32.ptx"),
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
    /// 測って一番速かった。測った設定と速度（鍵/秒）
    Fastest(Vec<(GpuOptions, f64)>),
    Fixed,
    Default,
}

/// 1ブロックのスレッド数
const BLOCK: u32 = 128;

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

/// 表の1エントリの語数（y+x, y-x, 2dxy の 10 リムずつと未使用の 2 語。kernel.cu の ENTRY_WORDS）
const ENTRY_WORDS: usize = 32;

/// GPU を使えないときのエラー
#[derive(Debug)]
pub struct GpuError(String);

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        GpuError(format!("CUDA: {e}"))
    }
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
        write!(
            f,
            "{}（sm_{}{}・SM {}・{:.0} GB）",
            self.name,
            self.compute.0,
            self.compute.1,
            self.sms,
            self.memory as f64 / 1e9
        )
    }
}

/// 1つの設定で読み込んだカーネルと、GPU に送った表
struct Kernel {
    /// テストで別のカーネル（meikey_derive）を読むため
    #[cfg(test)]
    module: Arc<CudaModule>,
    func: CudaFunction,
    table: CudaSlice<u32>,
    options: GpuOptions,
    /// 起動するブロック数
    grid: u32,
}

/// 探索の1回の起動に渡すもの（カーネルの引数のうち、起動ごとに変わらないもの）
struct Launch {
    key: CudaSlice<u32>,
    needle: CudaSlice<u8>,
    needle_len: u32,
    fold: u32,
    stop_on_hit: u32,
    result: CudaSlice<u64>,
    sink: CudaSlice<u32>,
}

impl Kernel {
    /// 設定 `options` のカーネルを読み込み、表を送る。埋め込んだ PTX があればそれを、無いかドライバが読めなければ
    /// NVRTC でこの GPU 向けにコンパイルしたものを使う
    fn load(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        info: &GpuInfo,
        options: GpuOptions,
    ) -> Result<Kernel, GpuError> {
        if !WIDTHS.contains(&options.width) || !BATCHES.contains(&options.batch) {
            return Err(GpuError(format!(
                "表の幅は {}〜{}、まとめる鍵の数は {}〜{}（{options}）",
                WIDTHS.start(),
                WIDTHS.end(),
                BATCHES.start(),
                BATCHES.end()
            )));
        }
        let embedded =
            embedded_ptx(options).map(|ptx| ctx.load_module(cudarc::nvrtc::Ptx::from_src(ptx)));
        let module = match embedded {
            Some(Ok(m)) => m,
            // 埋め込んでいない設定か、ドライバが古くて埋め込んだ PTX を読めない: NVRTC でコンパイルする
            other => {
                // SAFETY: ライブラリを読み込めるか試すだけ
                if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
                    return Err(GpuError(match other {
                        Some(Err(e)) => format!(
                            "PTX を読めない（{e}。ドライバを更新するか、NVRTC を入れて LD_LIBRARY_PATH / PATH に追加）"
                        ),
                        _ => format!(
                            "{options} は埋め込んでいない設定なので NVRTC（libnvrtc）が要る（LD_LIBRARY_PATH / PATH に追加）"
                        ),
                    }));
                }
                ctx.load_module(compile(info.compute, options)?)?
            }
        };
        let func = module.load_function("meikey_search")?;
        // SM あたりに同時に載るブロック数（レジスタの使用量で決まる）の分だけ起動する
        let per_sm = func
            .occupancy_max_active_blocks_per_multiprocessor(BLOCK, 0, None)
            .unwrap_or(2)
            .max(1);
        let table = stream.clone_htod(&table_words(options.width))?;
        Ok(Kernel {
            #[cfg(test)]
            module,
            func,
            table,
            options,
            grid: info.sms.max(1) * per_sm,
        })
    }

    /// 1回の起動で計算する鍵の数（1スレッドの繰り返し1回あたり）
    fn keys_per_iter(&self) -> u64 {
        u64::from(self.grid) * u64::from(BLOCK) * u64::from(self.options.batch)
    }

    /// 起動に渡すものを GPU に置く。`key` はシードを作る ChaCha20 の鍵、`needle` は探す文字列
    /// （fold なら小文字にしたもの）
    fn prepare(
        &self,
        stream: &Arc<CudaStream>,
        key: &[u8; 32],
        needle: &[u8],
        fold: bool,
        stop_on_hit: bool,
    ) -> Result<Launch, GpuError> {
        let key_words: Vec<u32> = key
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        Ok(Launch {
            key: stream.clone_htod(&key_words)?,
            needle: stream.clone_htod(&[needle, &[0; 64]].concat())?,
            needle_len: needle.len() as u32,
            fold: u32::from(fold),
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
        b.arg(&self.table)
            .arg(&l.key)
            .arg(&l.needle)
            .arg(&l.needle_len)
            .arg(&l.fold)
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
        let mut l = self.prepare(stream, &[0x5a; 32], MEASURE_NEEDLE, true, false)?;
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

/// 設定 `candidates` を順に `measure` で読み込んで測り、一番速いものを返す。ほかのものは測ったらすぐ捨てる
/// （GPU のメモリを空けるため）。読み込めない設定は飛ばし、全部だめなら最初のエラーを返す
fn fastest<K>(
    candidates: &[GpuOptions],
    mut measure: impl FnMut(GpuOptions) -> Result<(K, f64), GpuError>,
) -> Result<(K, Vec<(GpuOptions, f64)>), GpuError> {
    let mut best: Option<(K, f64)> = None;
    let mut results = Vec::new();
    let mut first_error = None;
    for &options in candidates {
        match measure(options) {
            Ok((k, speed)) => {
                results.push((options, speed));
                if best.as_ref().is_none_or(|(_, s)| speed > *s) {
                    best = Some((k, speed));
                }
            }
            Err(e) => {
                first_error.get_or_insert(e);
            }
        }
    }
    match best {
        Some((k, _)) => Ok((k, results)),
        None => Err(first_error.unwrap_or_else(|| GpuError("測る設定が無い".into()))),
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
    /// [`GpuSetting::Fastest`] なら、埋め込んだ設定を全部測ってから一番速いものを使う）
    pub fn new(ordinal: usize, setting: GpuSetting) -> Result<Gpu, GpuError> {
        // SAFETY: ライブラリを読み込めるか試すだけ
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(GpuError(
                "CUDA のドライバが見つからない（NVIDIA の GPU とドライバが必要。libcuda / nvcuda.dll）"
                    .into(),
            ));
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
            GpuSetting::Fixed(options) => {
                (Kernel::load(&ctx, &stream, &info, options)?, Chosen::Fixed)
            }
            GpuSetting::Default => (
                Kernel::load(&ctx, &stream, &info, GpuOptions::default())?,
                Chosen::Default,
            ),
            GpuSetting::Fastest => {
                let (kernel, results) = fastest(&EMBEDDED, |options| {
                    let k = Kernel::load(&ctx, &stream, &info, options)?;
                    let speed = k.measure(&stream)?;
                    Ok((k, speed))
                })?;
                (kernel, Chosen::Fastest(results))
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
        // 探す文字列（大小文字を区別しないなら、GPU 側で base64 を小文字にして比べるので小文字に）
        let (needle, fold) = match &config.pattern {
            Some(p) if p.case_sensitive() => (p.as_str().as_bytes().to_vec(), false),
            Some(p) => (p.as_str().to_ascii_lowercase().into_bytes(), true),
            // ベンチマークで判定しないとき: 長さ 0 は一致しない
            None => (Vec::new(), false),
        };
        // ベンチマークでは当たっても止まらない
        let mut l = kernel.prepare(stream, &key, &needle, fold, !config.bench)?;

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
        let found = VanityKey::from_seed(seed(key, index));
        if let Some(p) = &config.pattern {
            // 一致しなければ GPU の計算か判定が間違っている（バグ）ので止める
            assert!(
                p.is_match(&found.public_b64()),
                "内部エラー: GPU（{}）の結果が ed25519-dalek・総当たりと不一致",
                self.info.name
            );
        }
        found
    }
}

impl Searcher for Gpu {
    fn run(&self, config: &SearchConfig, on_progress: &mut dyn Progress) -> Outcome {
        self.search(config, on_progress)
            .unwrap_or_else(|e| panic!("GPU での探索に失敗: {e}"))
    }

    fn status(&self, _config: &SearchConfig) -> Vec<(&'static str, String)> {
        let (i, k) = (&self.info, &self.kernel);
        let how = match &self.chosen {
            Chosen::Fastest(results) => format!("{} 通りを測って一番速いもの", results.len()),
            Chosen::Fixed => "指定".into(),
            Chosen::Default => "既定。--check では測らない".into(),
        };
        vec![
            ("GPU", i.name.clone()),
            ("世代", format!("sm_{}{}", i.compute.0, i.compute.1)),
            ("SM", i.sms.to_string()),
            ("メモリ", format!("{:.0} GB", i.memory as f64 / 1e9)),
            ("設定", format!("{}（{how}）", k.options)),
            // CPU と同時に探すとき、CPU の「並列」と並ぶので別の名前にする
            (
                "グリッド",
                format!(
                    "{} ブロック × {} スレッド × {} 鍵",
                    k.grid, BLOCK, k.options.batch
                ),
            ),
            (
                "表",
                format!(
                    "W = {}（{:.1} MB）",
                    k.options.width,
                    k.options.table_bytes() as f64 / 1e6
                ),
            ),
        ]
    }
}

/// 通し番号 `index` のシード（kernel.cu の chacha20_seed と同じ: ChaCha20 のブロック `index` の先頭 32 バイト）
fn seed(key: &[u8; 32], index: u64) -> [u8; 32] {
    let mut rng = ChaCha20Rng::from_seed(*key);
    rng.set_block_pos(index);
    let mut s = [0u8; 32];
    rng.fill_bytes(&mut s);
    s
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
    Err(GpuError(format!(
        "カーネルのコンパイルに失敗:\n{}",
        errors.join("\n")
    )))
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
        "{source}\n// {PTX_ARCH} W={} BATCH={}",
        options.width, options.batch
    ));
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// 設定 `options` の埋め込む PTX（`gpu/ptx/` のファイルの中身）を NVRTC で作る（`cargo make ptx`）。
/// 互換性のため、なるべく古い NVRTC で作る（PTX の版はコンパイルした NVRTC の版で決まり、それより古いドライバでは読めない）
pub fn portable_ptx(options: GpuOptions) -> Result<String, GpuError> {
    // SAFETY: ライブラリを読み込めるか試すだけ
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(GpuError("libnvrtc が見つからない".into()));
    }
    let ptx = compile_for(PTX_ARCH, options).map_err(|e| GpuError(compile_error(&e)))?;
    Ok(format!(
        // PTX は ASCII だけ（ptxas は ASCII 以外の文字を受け付けない）
        "// meikey: sha256(kernel.cu, W={}, BATCH={}) = {} (regenerate: cargo make ptx)\n{}",
        options.width,
        options.batch,
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

/// 幅 `width` の表を GPU 用（1エントリ = 25.5 ビット × 10 リムを3つ + 未使用 2 語）に並べる
fn table_words(width: u32) -> Vec<u32> {
    let entries = entries_with_width(width as usize);
    let mut words = vec![0u32; entries.len() * ENTRY_WORDS];
    for (e, entry) in words
        .as_chunks_mut::<ENTRY_WORDS>()
        .0
        .iter_mut()
        .zip(entries)
    {
        for (f, fe) in entry.into_iter().enumerate() {
            for (i, l) in fe.reduce().into_iter().enumerate() {
                e[10 * f + 2 * i] = (l & ((1 << 26) - 1)) as u32;
                e[10 * f + 2 * i + 1] = (l >> 26) as u32;
            }
        }
    }
    words
}

#[cfg(test)]
mod tests {
    //! kernel.cu を CPU 向けにコンパイルしたもの（build.rs）を呼んで、Rust の実装と突き合わせる
    use super::*;
    use crate::model::pattern::Pattern;
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
    #[cfg(meikey_gpu_host)]
    use std::sync::LazyLock;
    use std::thread::scope;
    use std::thread::sleep;

    #[cfg(meikey_gpu_host)]
    unsafe extern "C" {
        fn meikey_test_seed(key: *const u32, counter: u64, out: *mut u8);
        fn meikey_test_secret(seed: *const u8, out: *mut u8);
        fn meikey_test_digits(s: *const u8, out: *mut i32);
        fn meikey_test_fe_mul(a: *const u32, b: *const u32, out: *mut u32);
        fn meikey_test_fe_sq(a: *const u32, out: *mut u32);
        fn meikey_test_fe_invert_bytes(a: *const u32, out: *mut u8);
        fn meikey_test_derive_batch(table: *const u32, key: *const u32, first: u64, out: *mut u8);
        fn meikey_test_contains(pk: *const u8, needle: *const u8, len: u32, fold: u32) -> i32;
        fn meikey_test_tail_b64(pk: *const u8, out: *mut u8);
        fn meikey_test_search(
            nthreads: u64,
            table: *const u32,
            key: *const u32,
            needle: *const u8,
            needle_len: u32,
            fold: u32,
            base: u64,
            iters: u32,
            stop_on_hit: u32,
            result: *mut u64,
        );
        fn meikey_test_batch() -> u32;
        fn meikey_test_width() -> u32;
        // host_wide.cpp（表の幅 16、まとめる鍵の数 32）
        fn meikey_wide_test_derive_batch(
            table: *const u32,
            key: *const u32,
            first: u64,
            out: *mut u8,
        );
        fn meikey_wide_test_search(
            nthreads: u64,
            table: *const u32,
            key: *const u32,
            needle: *const u8,
            needle_len: u32,
            fold: u32,
            base: u64,
            iters: u32,
            stop_on_hit: u32,
            result: *mut u64,
        );
        fn meikey_wide_test_batch() -> u32;
        fn meikey_wide_test_width() -> u32;
    }

    /// host_wide.cpp の設定
    #[cfg(meikey_gpu_host)]
    const WIDE: GpuOptions = GpuOptions {
        width: 16,
        batch: 32,
    };

    #[cfg(meikey_gpu_host)]
    static TABLE: LazyLock<Vec<u32>> = LazyLock::new(|| table_words(DEFAULT_WIDTH));

    #[cfg(meikey_gpu_host)]
    static WIDE_TABLE: LazyLock<Vec<u32>> = LazyLock::new(|| table_words(WIDE.width));

    fn key_words(key: &[u8; 32]) -> [u32; 8] {
        from_fn(|i| u32::from_le_bytes(*key[4 * i..].first_chunk().unwrap()))
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
        // SAFETY: 引数なしの関数
        let (batch, width, wide_batch, wide_width) = unsafe {
            (
                meikey_test_batch(),
                meikey_test_width(),
                meikey_wide_test_batch(),
                meikey_wide_test_width(),
            )
        };
        assert_eq!(
            GpuOptions { width, batch },
            GpuOptions::default(),
            "host.cpp は既定の設定"
        );
        assert_eq!(
            GpuOptions {
                width: wide_width,
                batch: wide_batch
            },
            WIDE
        );
        // 既定の幅は CPU の表（comb::W）と同じ（下の digits_match_comb で comb の桁と比べるため）
        assert_eq!(DEFAULT_WIDTH as usize, crate::service::derive::comb::W);
    }

    /// 表の大きさ（W = 12 で約 5.8 MB、16 で約 67 MB）
    #[rstest]
    #[case::default(GpuOptions::default(), 22 * 2049 * 128)]
    #[case::wide(GpuOptions { width: 16, batch: 32 }, 16 * 32769 * 128)]
    fn table_bytes(#[case] options: GpuOptions, #[case] want: usize) {
        assert_eq!(options.table_bytes(), want);
    }

    /// シードは rand の ChaCha20Rng（ストリーム 0）のブロック `index` の先頭 32 バイト
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn seed_matches_chacha20_rng() {
        check(256, (any::<[u8; 32]>(), any::<u64>()), |(key, index)| {
            let mut got = [0u8; 32];
            // SAFETY: key は 8 語、out は 32 バイト
            unsafe { meikey_test_seed(key_words(&key).as_ptr(), index, got.as_mut_ptr()) };
            prop_assert_eq!(got, seed(&key, index));
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
        check(256, any::<[u8; 32]>(), |mut s| {
            s[31] &= 0x7f;
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

    /// まとめて計算した公開鍵は、同じシードから ed25519-dalek で作った鍵と一致する
    /// （既定の設定と、host_wide.cpp の表の幅 16・まとめる鍵の数 32）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn derive_batch_matches_dalek(#[values(false, true)] wide: bool) {
        let (table, batch, derive): (
            &[u32],
            u32,
            unsafe extern "C" fn(*const u32, *const u32, u64, *mut u8),
        ) = if wide {
            (&WIDE_TABLE, WIDE.batch, meikey_wide_test_derive_batch)
        } else {
            (&TABLE, DEFAULT_BATCH, meikey_test_derive_batch)
        };
        check(16, (any::<[u8; 32]>(), any::<u64>()), |(key, first)| {
            let first = first >> 8; // 足しても桁あふれしないように
            let mut got = vec![0u8; 32 * batch as usize];
            // SAFETY: 表はその設定の行数·列数のエントリ、key は 8 語、out は batch·32 バイト
            unsafe {
                derive(
                    table.as_ptr(),
                    key_words(&key).as_ptr(),
                    first,
                    got.as_mut_ptr(),
                )
            };
            for (j, pk) in got.as_chunks::<32>().0.iter().enumerate() {
                let want = VanityKey::from_seed(seed(&key, first + j as u64));
                prop_assert_eq!(pk, want.public_bytes(), "#{}", j);
            }
            Ok(())
        });
    }

    /// base64 は model::key と同じで、判定は Pattern と同じ（大小文字の区別あり・なし）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn b64_and_match_agree_with_pattern(#[values(false, true)] case_sensitive: bool) {
        let needle = "[A-Za-z0-9+/]{1,3}";
        check(1024, (any::<[u8; 32]>(), needle), |(seed, n)| {
            let key = VanityKey::from_seed(seed);
            let pk = key.public_bytes();
            let mut tail = [0u8; 44];
            // SAFETY: pk は 32 バイト、out は 44 バイト
            unsafe { meikey_test_tail_b64(pk.as_ptr(), tail.as_mut_ptr()) };
            prop_assert_eq!(&tail[..], &key.public_b64()[24..]);

            let p = Pattern::new(&n, case_sensitive).unwrap();
            let bytes = if case_sensitive {
                n.clone()
            } else {
                n.to_ascii_lowercase()
            };
            // SAFETY: pk は 32 バイト、needle は len バイト
            let got = unsafe {
                meikey_test_contains(
                    pk.as_ptr(),
                    bytes.as_ptr(),
                    bytes.len() as u32,
                    u32::from(!case_sensitive),
                )
            } != 0;
            prop_assert_eq!(got, p.is_match(&key.public_b64()));
            Ok(())
        });
    }

    /// 鍵の base64 の一部（位置・長さはいろいろ。大小文字を変えたり1文字変えたりもする）を探すと、
    /// Pattern と同じ結果になる（先頭 2 文字で候補を絞ったあとの、残りを比べる部分も通す）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn match_agrees_with_pattern_for_parts_of_the_key(#[values(false, true)] case_sensitive: bool) {
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
            let Ok(p) = Pattern::new(&n, case_sensitive) else {
                return Ok(()); // 長すぎるなど、Pattern が受け付けない文字列
            };
            let bytes = if case_sensitive {
                n.clone()
            } else {
                n.to_ascii_lowercase()
            };
            // SAFETY: pk は 32 バイト、needle は len バイト
            let got = unsafe {
                meikey_test_contains(
                    key.public_bytes().as_ptr(),
                    bytes.as_ptr(),
                    bytes.len() as u32,
                    u32::from(!case_sensitive),
                )
            } != 0;
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

    /// 埋め込んだ PTX は今の kernel.cu とその設定から作ったもの（違えば `cargo make ptx` で作り直す）
    #[rstest]
    fn embedded_ptx_matches_kernel_source(#[values(0, 1, 2, 3, 4, 5)] i: usize) {
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
            assert_eq!(
                WIDTHS.contains(&o.width) && BATCHES.contains(&o.batch),
                true,
                "{o}"
            );
            assert_eq!(EMBEDDED[..i].contains(o), false, "{o} が重なっている");
        }
        assert_eq!(EMBEDDED.contains(&GpuOptions::default()), true);
        assert_eq!(embedded_ptx(GpuOptions::new(13, 16)), None);
    }

    /// 測った中で一番速い設定を選び、ほかは捨てる。読み込めない設定は飛ばし、全部だめなら最初のエラー
    #[rstest]
    #[case::picks_fastest(&[Ok(5.7), Ok(6.1), Ok(7.0), Ok(6.8)], Ok(2))]
    #[case::first_of_ties(&[Ok(3.0), Ok(3.0)], Ok(0))]
    #[case::skips_errors(&[Err("a"), Ok(1.0), Err("b"), Ok(2.0)], Ok(3))]
    #[case::all_errors(&[Err("a"), Err("b")], Err("a"))]
    #[case::nothing(&[], Err("測る設定が無い"))]
    fn fastest_picks_the_fastest(
        #[case] speeds: &[Result<f64, &str>],
        #[case] want: Result<usize, &str>,
    ) {
        let candidates: Vec<GpuOptions> = (0..speeds.len() as u32)
            .map(|i| GpuOptions::new(i + 2, 1))
            .collect();
        let mut measured = Vec::new();
        let got = fastest(&candidates, |o| {
            let i = (o.width - 2) as usize;
            measured.push(i);
            speeds[i].map(|s| (i, s)).map_err(|e| GpuError(e.into()))
        });
        assert_eq!(
            measured,
            (0..speeds.len()).collect::<Vec<_>>(),
            "全部を順に測る"
        );
        match (got, want) {
            (Ok((k, results)), Ok(w)) => {
                assert_eq!(k, w);
                let ok: Vec<f64> = speeds.iter().filter_map(|s| s.ok()).collect();
                assert_eq!(results.iter().map(|r| r.1).collect::<Vec<_>>(), ok);
            }
            (Err(e), Err(w)) => assert_eq!(e.to_string(), w),
            (got, want) => panic!("{:?} ではなく {want:?}", got.map(|(k, _)| k)),
        }
    }

    // ---- NVRTC・GPU があるときだけ（無ければ飛ばす。MEIKEY_REQUIRE_SIMD に nvrtc・cuda があれば失敗させる） ----

    #[fixture]
    fn nvrtc() -> Option<()> {
        // SAFETY: ライブラリを読み込めるか試すだけ
        let present = unsafe { nvrtc_present() };
        available("nvrtc", present.then_some(()))
    }

    #[fixture]
    fn gpu() -> Option<Gpu> {
        // SAFETY: ライブラリを読み込めるか試すだけ
        let present =
            unsafe { driver_present() } && CudaContext::device_count().is_ok_and(|n| n > 0);
        available("cuda", present.then_some(()))?;
        Some(
            Gpu::new(0, GpuSetting::Fixed(GpuOptions::default()))
                .expect("GPU を使えるようにできる"),
        )
    }

    /// カーネルは Turing（T4）から Blackwell（RTX PRO 6000）まで、各世代向けに、選べる設定の端の値でも
    /// コンパイルできる
    #[rstest]
    fn kernel_compiles_for_each_generation(
        #[values((7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (12, 0))] compute: (i32, i32),
        #[values(
            GpuOptions::default(),
            GpuOptions { width: *WIDTHS.start(), batch: *BATCHES.start() },
            GpuOptions { width: *WIDTHS.end(), batch: *BATCHES.end() }
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

    /// GPU で計算した公開鍵は、同じシードから ed25519-dalek で作った鍵と一致する
    #[rstest]
    fn gpu_derive_matches_dalek(gpu: Option<Gpu>) {
        let Some(gpu) = gpu else {
            return;
        };
        let f = gpu.kernel.module.load_function("meikey_derive").unwrap();
        let key: [u8; 32] = random();
        let first = 1 << 40;
        let (blocks, threads) = (8u32, BLOCK);
        let n = (blocks * threads) as usize * gpu.kernel.options.batch as usize;
        let stream = &gpu.stream;
        let key_dev = stream.clone_htod(&key_words(&key)).unwrap();
        let mut out = stream.alloc_zeros::<u8>(n * 32).unwrap();
        let mut b = stream.launch_builder(&f);
        b.arg(&gpu.kernel.table)
            .arg(&key_dev)
            .arg(&first)
            .arg(&mut out);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: 引数は kernel.cu の meikey_derive と同じ並びで、out は全スレッドの鍵の分ある
        unsafe { b.launch(cfg).unwrap() };
        let got = stream.clone_dtoh(&out).unwrap();
        for (i, pk) in got.as_chunks::<32>().0.iter().enumerate() {
            let want = VanityKey::from_seed(seed(&key, first + i as u64));
            assert_eq!(pk, want.public_bytes(), "#{i}");
        }
    }

    /// 設定を測って選ぶと、埋め込んだ設定を全部測って、その中で一番速いものを使う。その設定でも鍵が見つかる
    #[rstest]
    fn gpu_uses_the_fastest_setting(gpu: Option<Gpu>) {
        if gpu.is_none() {
            return;
        }
        let gpu = Gpu::new(0, GpuSetting::Fastest).expect("GPU を使えるようにできる");
        let Chosen::Fastest(results) = &gpu.chosen else {
            panic!("測っていない");
        };
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
    fn gpu_search_finds_key(#[values(false, true)] case_sensitive: bool, gpu: Option<Gpu>) {
        let Some(gpu) = gpu else {
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
    fn gpu_with_cpu(#[values(false, true)] cancel_only: bool, gpu: Option<Gpu>) {
        let Some(gpu) = gpu else {
            return;
        };
        if cancel_only {
            let config = SearchConfig::new(Some(
                Pattern::new(&"z".repeat(Pattern::MAX_LEN), false).unwrap(),
            ));
            let cancel = config.cancel.clone();
            let started = Instant::now();
            let out = scope(|s| {
                s.spawn(|| {
                    sleep(Duration::from_millis(300));
                    cancel.cancel();
                });
                gpu.run(&config, &mut |_| {})
            });
            assert_eq!(out.key.is_none() && out.stats.tries > 0, true);
            assert_eq!(
                started.elapsed() < Duration::from_secs(5),
                true,
                "止まるのが遅い"
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
    /// （既定の設定と、host_wide.cpp の表の幅 16・まとめる鍵の数 32）
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn search_finds_first_hit(#[values(false, true)] wide: bool) {
        type Search = unsafe extern "C" fn(
            u64,
            *const u32,
            *const u32,
            *const u8,
            u32,
            u32,
            u64,
            u32,
            u32,
            *mut u64,
        );
        let (table, search): (&[u32], Search) = if wide {
            (&WIDE_TABLE, meikey_wide_test_search)
        } else {
            (&TABLE, meikey_test_search)
        };
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
                needle.as_ptr(),
                2,
                1,
                0,
                32,
                1,
                result.as_mut_ptr(),
            );
        }
        assert_eq!(result[0], 1, "512 鍵以上のうちに「ab」はほぼ確実に出る");
        let hit = result[1];
        assert_eq!(
            p.is_match(&VanityKey::from_seed(seed(&key, hit)).public_b64()),
            true
        );
        for i in 0..hit {
            assert_eq!(
                p.is_match(&VanityKey::from_seed(seed(&key, i)).public_b64()),
                false,
                "{i} が先に当たる"
            );
        }
    }
}
