//! GPU（CUDA）で鍵を探す [`Searcher`]（feature `cuda`、`--gpu`）
//!
//! シードの生成から文字列の判定までを GPU のカーネル（`gpu/kernel.cu`）で行い、CPU は起動と結果の確認だけをする。
//!
//! - カーネルは compute_75（Turing）向けの PTX（`gpu/kernel.ptx`）を埋め込んでおき、起動時にドライバが
//!   その GPU 向けに変換する（変換結果はドライバがキャッシュするので2回目からは速い）。要るのは NVIDIA の
//!   ドライバ（libcuda / nvcuda.dll）だけで、CUDA Toolkit は要らない。ドライバが古くて PTX を読めないときだけ、
//!   NVRTC（libnvrtc）があればその GPU 向けにコンパイルし直す。PTX は `cargo make ptx` で kernel.cu から作る
//! - CUDA のライブラリは実行時に読み込むので、ビルドに CUDA は要らない
//! - シードは `ChaCha20(鍵 = 起動ごとの乱数, ブロック番号 = 鍵の通し番号)` の先頭 32 バイト。
//!   当たりは通し番号だけを受け取り、CPU で同じシードを作り直して ed25519-dalek で計算し、
//!   文字列が本当に含まれるかを確かめてから返す（CPU 版と同じく、GPU の計算にバグがあっても誤った鍵は返さない）
//! - 櫛形テーブル（`comb::Table`、W = 12）は CPU で作って 25.5 ビット × 10 リムにして GPU に送る
//! - 1回の起動（カーネル）が `TARGET_LAUNCH`（0.2 秒）前後で終わるように、1スレッドの繰り返し回数を測りながら調整する。
//!   画面をつないだ GPU（Windows のタイムアウト検出の対象）では、画面が固まらないよう `DISPLAY_LAUNCH`（0.05 秒）にする

use crate::model::key::VanityKey;
use crate::model::search::{Outcome, SearchConfig, Stats};
use crate::service::derive::comb::{COLS, ROWS, Table, W};
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

/// kernel.cu を compute_75 向けにコンパイルした PTX（`cargo make ptx` で作る）。1行目に kernel.cu の
/// ハッシュ（[`kernel_hash`]）を書いておき、ソースと食い違っていないかをテストで確かめる
const PTX: &str = include_str!("gpu/kernel.ptx");

/// 埋め込む PTX の世代。これより新しい GPU でもドライバが変換して動かせる（Turing 〜 Blackwell）
pub const PTX_ARCH: &str = "compute_75";

/// 1スレッドが1回にまとめて計算する鍵の数（逆元をまとめる数）
const BATCH: u64 = 8;

/// 1ブロックのスレッド数
const BLOCK: u32 = 128;

/// 1回の起動にかける時間の目安（長いほど起動の手間は減るが、当たりや時間切れに気づくのが遅れる）
const TARGET_LAUNCH: Duration = Duration::from_millis(200);

/// 画面をつないだ GPU での目安。Windows は GPU が約 2 秒応答しないとドライバをリセットし、
/// 起動が長いと画面もその間固まるので短くする
const DISPLAY_LAUNCH: Duration = Duration::from_millis(50);

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

/// GPU で探す [`Searcher`]
pub struct Gpu {
    stream: Arc<CudaStream>,
    /// テストで別のカーネル（meikey_derive）を読むため
    #[cfg(test)]
    module: Arc<CudaModule>,
    func: CudaFunction,
    table: CudaSlice<u32>,
    info: GpuInfo,
    /// 起動するブロック数
    grid: u32,
    /// 1回の起動にかける時間の目安
    target: Duration,
}

impl Gpu {
    /// `ordinal` 番目の GPU を使えるようにする（カーネルのコンパイルと表の転送をする）
    pub fn new(ordinal: usize) -> Result<Gpu, GpuError> {
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
        let module = match ctx.load_module(cudarc::nvrtc::Ptx::from_src(PTX)) {
            Ok(m) => m,
            // ドライバが古くて埋め込んだ PTX を読めない: NVRTC があればこの GPU 向けにコンパイルし直す
            Err(embedded) => {
                // SAFETY: ライブラリを読み込めるか試すだけ
                if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
                    return Err(GpuError(format!(
                        "PTX を読めない（{embedded}。ドライバを更新するか、NVRTC を入れて LD_LIBRARY_PATH / PATH に追加）"
                    )));
                }
                ctx.load_module(compile(info.compute)?)?
            }
        };
        let func = module.load_function("meikey_search")?;
        // 画面をつないだ GPU（カーネルの実行時間に制限がある）なら起動を短くする
        let target =
            match ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT) {
                Ok(1) => DISPLAY_LAUNCH,
                _ => TARGET_LAUNCH,
            };
        // SM あたりに同時に載るブロック数（レジスタの使用量で決まる）の分だけ起動する
        let per_sm = func
            .occupancy_max_active_blocks_per_multiprocessor(BLOCK, 0, None)
            .unwrap_or(2)
            .max(1);
        let grid = info.sms.max(1) * per_sm;
        let stream = ctx.default_stream();
        let table = stream.clone_htod(&table_words())?;
        Ok(Gpu {
            stream,
            #[cfg(test)]
            module,
            func,
            table,
            info,
            grid,
            target,
        })
    }

    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// 1回の起動で計算する鍵の数（1スレッドの繰り返し1回あたり）
    fn keys_per_iter(&self) -> u64 {
        u64::from(self.grid) * u64::from(BLOCK) * BATCH
    }

    fn search(
        &self,
        config: &SearchConfig,
        on_progress: &mut dyn Progress,
    ) -> Result<Outcome, GpuError> {
        let stream = &self.stream;
        // シードを作る ChaCha20 の鍵（起動ごとに新しい乱数）
        let key: [u8; 32] = rand::random();
        let key_words: Vec<u32> = key
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        let key_dev = stream.clone_htod(&key_words)?;
        // 探す文字列（大小文字を区別しないなら、GPU 側で base64 を小文字にして比べるので小文字に）
        let (needle, fold) = match &config.pattern {
            Some(p) if p.case_sensitive() => (p.as_str().as_bytes().to_vec(), 0u32),
            Some(p) => (p.as_str().to_ascii_lowercase().into_bytes(), 1u32),
            // ベンチマークで判定しないとき: 長さ 0 は一致しない
            None => (Vec::new(), 0u32),
        };
        let needle_len = needle.len() as u32;
        let needle_dev = stream.clone_htod(&[needle, vec![0; 64]].concat())?;
        let mut result = stream.alloc_zeros::<u64>(2)?;
        let nthreads = (self.grid * BLOCK) as usize;
        let mut sink = stream.alloc_zeros::<u32>(nthreads)?;
        // ベンチマークでは当たっても止まらない
        let stop_on_hit = u32::from(!config.bench);

        let start = Instant::now();
        let mut tries = 0u64;
        let mut iters = 1u32;
        let mut next_report = start + config.progress_interval;
        loop {
            let launch = Instant::now();
            let base = tries;
            let mut b = stream.launch_builder(&self.func);
            b.arg(&self.table)
                .arg(&key_dev)
                .arg(&needle_dev)
                .arg(&needle_len)
                .arg(&fold)
                .arg(&base)
                .arg(&iters)
                .arg(&stop_on_hit)
                .arg(&mut result)
                .arg(&mut sink);
            let cfg = LaunchConfig {
                grid_dim: (self.grid, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: 引数の型と並びは kernel.cu の meikey_search と同じ。sink はスレッド数ぶん、
            // result は 2 語、needle は文字列 + 余白 64 バイト確保してある
            unsafe { b.launch(cfg)? };
            let found = stream.clone_dtoh(&result)?; // 終わるまで待つ
            let took = launch.elapsed();
            tries += self.keys_per_iter() * u64::from(iters);
            let stats = Stats {
                tries,
                elapsed: start.elapsed(),
            };

            if stop_on_hit == 1 && found[0] != 0 {
                let key = self.confirm(&key, found[1], config);
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
        let i = &self.info;
        vec![
            ("GPU", i.name.clone()),
            ("世代", format!("sm_{}{}", i.compute.0, i.compute.1)),
            ("SM", i.sms.to_string()),
            ("メモリ", format!("{:.0} GB", i.memory as f64 / 1e9)),
            // CPU と同時に探すとき、CPU の「並列」と並ぶので別の名前にする
            (
                "グリッド",
                format!("{} ブロック × {} スレッド × {} 鍵", self.grid, BLOCK, BATCH),
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
fn compile(compute: (i32, i32)) -> Result<cudarc::nvrtc::Ptx, GpuError> {
    let exact = format!("sm_{}{}", compute.0, compute.1);
    let mut errors = Vec::new();
    for arch in [exact.as_str(), "compute_90", "compute_80", PTX_ARCH] {
        match compile_for(arch) {
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
fn compile_for(arch: &str) -> Result<cudarc::nvrtc::Ptx, CompileError> {
    compile_ptx_with_opts(
        KERNEL,
        CompileOptions {
            options: vec![
                format!("--gpu-architecture={arch}"),
                format!("-DMEIKEY_W={W}"),
                format!("-DMEIKEY_BATCH={BATCH}"),
            ],
            ..Default::default()
        },
    )
}

/// kernel.cu（改行は LF にそろえる）とコンパイルの設定のハッシュ（埋め込む PTX の1行目に書く）
pub fn kernel_hash() -> String {
    use sha2::{Digest, Sha256};
    let source = KERNEL.replace('\r', "");
    let digest = Sha256::digest(format!("{source}\n// {PTX_ARCH} W={W} BATCH={BATCH}"));
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// 埋め込む PTX（kernel.ptx の中身）を NVRTC で作る（`cargo make ptx`）。互換性のため、なるべく古い NVRTC で作る
/// （PTX の版はコンパイルした NVRTC の版で決まり、それより古いドライバでは読めない）
pub fn portable_ptx() -> Result<String, GpuError> {
    // SAFETY: ライブラリを読み込めるか試すだけ
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(GpuError("libnvrtc が見つからない".into()));
    }
    let ptx = compile_for(PTX_ARCH).map_err(|e| GpuError(compile_error(&e)))?;
    Ok(format!(
        // PTX は ASCII だけ（ptxas は ASCII 以外の文字を受け付けない）
        "// meikey: sha256(kernel.cu) = {} (regenerate: cargo make ptx)\n{}",
        kernel_hash(),
        ptx.to_src()
    ))
}

fn compile_error(e: &CompileError) -> String {
    match e {
        CompileError::CompileError { log, .. } => log.to_string_lossy().into_owned(),
        other => format!("{other:?}"),
    }
}

/// [`Table`] を GPU 用（1エントリ = 25.5 ビット × 10 リムを3つ + 未使用 2 語）に並べる
fn table_words() -> Vec<u32> {
    let table = Table::get();
    let mut words = vec![0u32; ROWS * COLS * ENTRY_WORDS];
    for (n, e) in words
        .as_chunks_mut::<ENTRY_WORDS>()
        .0
        .iter_mut()
        .enumerate()
    {
        for (f, fe) in table.entry(n / COLS, n % COLS).into_iter().enumerate() {
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
    use crate::service::derive::comb::signed_digits;
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
        fn meikey_test_digits(s: *const u8, out: *mut i16);
        fn meikey_test_fe_mul(a: *const u32, b: *const u32, out: *mut u32);
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
    }

    #[cfg(meikey_gpu_host)]
    static TABLE: LazyLock<Vec<u32>> = LazyLock::new(table_words);

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
    fn batch_matches_kernel() {
        // SAFETY: 引数なしの関数
        assert_eq!(u64::from(unsafe { meikey_test_batch() }), BATCH);
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
            let mut got = [0i16; ROWS];
            // SAFETY: s は 32 バイト、out は ROWS 個
            unsafe { meikey_test_digits(s.as_ptr(), got.as_mut_ptr()) };
            prop_assert_eq!(got, signed_digits(&s));
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

    /// まとめて計算した公開鍵は、同じシードから ed25519-dalek で作った鍵と一致する
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn derive_batch_matches_dalek() {
        check(16, (any::<[u8; 32]>(), any::<u64>()), |(key, first)| {
            let first = first >> 8; // 足しても桁あふれしないように
            let mut got = vec![0u8; 32 * BATCH as usize];
            // SAFETY: 表は ROWS·COLS エントリ、key は 8 語、out は BATCH·32 バイト
            unsafe {
                meikey_test_derive_batch(
                    TABLE.as_ptr(),
                    key_words(&key).as_ptr(),
                    first,
                    got.as_mut_ptr(),
                );
            }
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

    /// 埋め込んだ PTX は今の kernel.cu から作ったもの（違えば `cargo make ptx` で作り直す）
    #[rstest]
    fn embedded_ptx_matches_kernel_source() {
        let first = PTX.lines().next().unwrap_or_default();
        assert_eq!(
            first.contains(&kernel_hash()),
            true,
            "kernel.ptx が kernel.cu と食い違っている（cargo make ptx で作り直す）: {first}"
        );
        assert_eq!(
            PTX.contains(".target sm_75") && PTX.contains(".entry meikey_search"),
            true
        );
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
        Some(Gpu::new(0).expect("GPU を使えるようにできる"))
    }

    /// カーネルは Turing（T4）から Blackwell（RTX PRO 6000）まで、各世代向けにコンパイルできる
    #[rstest]
    fn kernel_compiles_for_each_generation(
        #[values((7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (12, 0))] compute: (i32, i32),
        nvrtc: Option<()>,
    ) {
        if nvrtc.is_none() {
            return;
        }
        let ptx = compile(compute).unwrap_or_else(|e| panic!("{e}"));
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
        let f = gpu.module.load_function("meikey_derive").unwrap();
        let key: [u8; 32] = random();
        let first = 1 << 40;
        let (blocks, threads) = (8u32, BLOCK);
        let n = (blocks * threads) as usize * BATCH as usize;
        let stream = &gpu.stream;
        let key_dev = stream.clone_htod(&key_words(&key)).unwrap();
        let mut out = stream.alloc_zeros::<u8>(n * 32).unwrap();
        let mut b = stream.launch_builder(&f);
        b.arg(&gpu.table).arg(&key_dev).arg(&first).arg(&mut out);
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
    #[cfg(meikey_gpu_host)]
    #[rstest]
    fn search_finds_first_hit() {
        let key: [u8; 32] = from_fn(|i| i as u8 * 7);
        let p = Pattern::new("ab", false).unwrap();
        let needle = b"ab";
        let mut result = [0u64; 2];
        // SAFETY: 表・鍵・文字列・結果の大きさは kernel.cu の前提どおり
        unsafe {
            meikey_test_search(
                1,
                TABLE.as_ptr(),
                key_words(&key).as_ptr(),
                needle.as_ptr(),
                2,
                1,
                0,
                64,
                1,
                result.as_mut_ptr(),
            );
        }
        assert_eq!(result[0], 1, "512 鍵のうちに「ab」はほぼ確実に出る");
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
