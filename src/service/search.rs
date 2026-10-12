//! 全コアで鍵を生成して、当たりを探す（設定と結果の型は [`crate::model::search`]）

use crate::model::backend::Simd;
use crate::model::key::{self, PUBLIC_B64_LEN, VanityKey};
use crate::model::search::{Outcome, SearchConfig, Stats};
use crate::service::derive::{BATCH, Deriver, PublicKeys, Seeds};
use crate::service::matcher::Matcher;
use rand::Rng;
use rand::rngs::ChaCha20Rng;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::thread;
use std::time::{Duration, Instant};

/// 進捗確認・停止確認の間隔（当たり・時間切れに気づくまでの最大の遅れ）
const TICK: Duration = Duration::from_millis(10);

/// 進捗の受け取り先（`FnMut(Stats)` のクロージャならそのまま使える）。
/// mockall はクロージャ（`dyn FnMut`）の引数をモックにできないので、この trait で包む
pub trait Progress {
    fn report(&mut self, stats: Stats);
}

impl<F: FnMut(Stats)> Progress for F {
    fn report(&mut self, stats: Stats) {
        self(stats)
    }
}

/// 鍵の探索。controller はこの trait 越しに探索するので、テストでは mockall の
/// `MockSearcher` に差し替えて、鍵を計算せずに好きな結果（見つかった・見つからない）を返せる
#[cfg_attr(test, mockall::automock)]
pub trait Searcher {
    /// 探索を実行する。`on_progress` は呼び出し元スレッドで定期的に呼ばれる。
    fn run(&self, config: &SearchConfig, on_progress: &mut dyn Progress) -> Outcome;

    /// 探索の状態（表示用の「ラベル・値」の行。例: `("SIMD", "AVX2")`、`("GPU", "NVIDIA H100 80GB HBM3")`）
    fn status(&self, config: &SearchConfig) -> Vec<(&'static str, String)>;
}

/// 全コアで総当たりする [`Searcher`]（本番で使うもの。中身は [`run`]）
#[derive(Debug, Clone, Copy, Default)]
pub struct Parallel;

impl Searcher for Parallel {
    fn run(&self, config: &SearchConfig, on_progress: &mut dyn Progress) -> Outcome {
        run(config, |s| on_progress.report(s))
    }

    fn status(&self, config: &SearchConfig) -> Vec<(&'static str, String)> {
        let marks: Vec<String> = Simd::ALL
            .into_iter()
            .filter(|&s| s != Simd::Off)
            .map(|s| format!("{} {}", s.name(), if s.supported() { "✓" } else { "✗" }))
            .collect();
        let b = config.backend;
        vec![
            ("CPU", marks.join("  ")),
            ("SIMD", b.simd.effective().name().into()),
            ("倍算", b.mul.name().into()),
            ("逆元", b.compress.name()),
            ("検索", config.find.name().into()),
            ("並列", format!("{} スレッド", config.threads)),
        ]
    }
}

/// 探索を実行する。`on_progress` は呼び出し元スレッドで定期的に呼ばれる。
pub fn run(config: &SearchConfig, mut on_progress: impl FnMut(Stats)) -> Outcome {
    // 表（最適化ビルドで数十ミリ秒、テスト用のビルドでは数百ミリ秒）は計測の前に作っておく。
    // 時間制限や速度に、表を作る時間を含めないため
    config.backend.prepare();
    let stop = AtomicBool::new(false);
    let tries = AtomicU64::new(0);
    let found = OnceLock::new();
    let start = Instant::now();
    let stats = || Stats {
        tries: tries.load(Relaxed),
        elapsed: start.elapsed(),
    };

    thread::scope(|s| {
        for _ in 0..config.threads.max(1) {
            s.spawn(|| worker(config, &stop, &tries, &found));
        }

        let mut next_report = start + config.progress_interval;
        while !stop.load(Relaxed) {
            thread::sleep(TICK);
            if config
                .time_limit
                .is_some_and(|limit| start.elapsed() >= limit)
            {
                stop.store(true, Relaxed);
            } else if Instant::now() >= next_report {
                on_progress(stats());
                next_report += config.progress_interval;
            }
        }
    });

    Outcome {
        key: found.into_inner().map(VanityKey::from_seed),
        stats: stats(),
    }
}

/// スレッドがパニックで終わるときも、ほかのスレッドと呼び出し元に止まるよう伝える
/// （伝えないと、残りのスレッドが時間制限まで、制限が無ければ永久に回り続ける）
struct StopOnPanic<'a>(&'a AtomicBool);

impl Drop for StopOnPanic<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            self.0.store(true, Relaxed);
        }
    }
}

fn worker(config: &SearchConfig, stop: &AtomicBool, tries: &AtomicU64, found: &OnceLock<[u8; 32]>) {
    let _guard = StopOnPanic(stop);
    let mut rng: ChaCha20Rng = rand::make_rng();
    let mut deriver = Deriver::new(config.backend);
    let judge = config
        .pattern
        .as_ref()
        .map(|p| (p, Matcher::new(p, config.find)));
    let mut seeds: Seeds = [[0u8; 32]; BATCH];
    let mut publics: PublicKeys = [[0u8; 32]; BATCH];
    let mut b64 = [0u8; PUBLIC_B64_LEN];

    while !stop.load(Relaxed) {
        seeds.iter_mut().for_each(|s| rng.fill_bytes(s));
        deriver.public_keys(&seeds, &mut publics);

        for (i, (seed, public)) in seeds.iter().zip(&publics).enumerate() {
            key::encode_public_b64(public, &mut b64);
            let Some((pattern, matcher)) = &judge else {
                std::hint::black_box(&b64); // 判定せずに回す
                continue;
            };
            let hit = matcher.is_match(&b64);
            if config.bench {
                std::hint::black_box(hit); // 判定込みで測る: 当たっても止まらない
                continue;
            }
            if hit {
                // 書き出す鍵は ed25519-dalek で計算し直すので、そちらでも一致するか確かめる
                // （判定は総当たりの Pattern::is_match）。一致しなければ高速版の計算か検索が
                // 間違っている（バグ）ので止める
                assert!(
                    pattern.is_match(&VanityKey::from_seed(*seed).public_b64()),
                    "内部エラー: {}・{} の結果が ed25519-dalek・総当たりと不一致",
                    config.backend,
                    config.find
                );
                let _ = found.set(*seed); // 同時に当たったら先着を採用
                stop.store(true, Relaxed);
                tries.fetch_add(i as u64 + 1, Relaxed);
                return;
            }
        }
        tries.fetch_add(BATCH as u64, Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::backend::Backend;
    use crate::model::pattern::{Find, Pattern};
    use crate::property::check;
    use proptest::prelude::*;
    use rstest::rstest;

    /// 見つからない探索の種類
    #[derive(Debug, Clone, Copy)]
    enum Endless {
        /// 判定しない
        NoPattern,
        /// 42文字の z（現実には見つからない）
        Impossible,
        /// `--bench`: 毎回のように当たる "a" でも、判定するだけで止まらない
        BenchWithHits,
    }

    impl Endless {
        fn config(self) -> SearchConfig {
            match self {
                Endless::NoPattern => SearchConfig::new(None),
                Endless::Impossible => SearchConfig::new(Some(
                    Pattern::new(&"z".repeat(Pattern::MAX_LEN), false).unwrap(),
                )),
                Endless::BenchWithHits => SearchConfig {
                    bench: true,
                    ..SearchConfig::new(Some(Pattern::new("a", false).unwrap()))
                },
            }
        }
    }

    /// 短い文字列はすぐ見つかり、見つかった鍵は本当に一致している。
    /// 期待試行回数は多くても100回ほどなので、10秒で見つからなければ公開鍵の計算か検索が壊れている
    #[rstest]
    fn finds_matching_key(
        #[values("a", "Wa", "+/")] pattern: &str,
        #[values(1, 4)] threads: usize,
        #[values(Backend::FAST, Backend::BASELINE)] backend: Backend,
        #[values(Find::Memmem, Find::Naive)] find: Find,
    ) {
        let cfg = SearchConfig {
            threads,
            backend,
            find,
            time_limit: Some(Duration::from_secs(10)),
            ..SearchConfig::new(Some(Pattern::new(pattern, false).unwrap()))
        };
        let out = run(&cfg, |_| {});

        let key = out.key.expect("短い文字列ならすぐ見つかる");
        assert!(cfg.pattern.unwrap().is_match(&key.public_b64()));
        assert!(out.stats.tries >= 1);
    }

    /// 見つからない探索（--bench で当たりを無視する場合も）は、時間制限が来たらすぐ止まる
    #[rstest]
    fn stops_at_time_limit(
        #[values(Endless::NoPattern, Endless::Impossible, Endless::BenchWithHits)] endless: Endless,
        #[values(Backend::FAST, Backend::BASELINE)] backend: Backend,
        #[values(1, 2)] threads: usize,
    ) {
        let limit = Duration::from_millis(60);
        let cfg = SearchConfig {
            threads,
            backend,
            time_limit: Some(limit),
            ..endless.config()
        };
        let out = run(&cfg, |_| {});

        assert!(out.key.is_none());
        assert!(out.stats.tries > 0, "時間内に鍵を作っている");
        assert!(out.stats.elapsed >= limit);
        assert!(
            out.stats.elapsed < limit + Duration::from_secs(2),
            "止まるのが遅い: {:?}",
            out.stats.elapsed
        );
    }

    /// 進捗は間隔ごとに呼ばれる（早すぎず、多すぎず）。間隔より先に時間制限が来たら呼ばれない。
    /// 進捗の仕組みは公開鍵の計算方法によらないので、ベンチマーク（判定なし）の高速版で測る
    #[rstest]
    #[case::every_40ms(40, 180, 4)]
    #[case::every_60ms(60, 250, 4)]
    #[case::none_before_first_interval(10_000, 60, 0)]
    fn progress_timing(#[case] interval_ms: u64, #[case] limit_ms: u64, #[case] expected: usize) {
        let interval = Duration::from_millis(interval_ms);
        let cfg = SearchConfig {
            threads: 1,
            time_limit: Some(Duration::from_millis(limit_ms)),
            progress_interval: interval,
            ..SearchConfig::new(None)
        };
        let mut at = Vec::new();
        run(&cfg, |s| at.push(s.elapsed));

        assert!(
            (expected.saturating_sub(1)..=expected).contains(&at.len()),
            "{} 回（期待 {expected} 回前後）: {at:?}",
            at.len()
        );
        for (k, t) in at.iter().enumerate() {
            assert!(
                *t >= interval * (k as u32 + 1),
                "{k} 回目が早すぎる: {at:?}"
            );
        }
    }

    /// 本番の [`Parallel`]（trait 越し）も、進捗を [`Progress`] に渡し、時間制限で止まる
    #[rstest]
    fn parallel_searcher_reports_progress() {
        let cfg = SearchConfig {
            threads: 1,
            time_limit: Some(Duration::from_millis(150)),
            progress_interval: Duration::from_millis(40),
            ..SearchConfig::new(None)
        };
        let mut reported = Vec::new();
        let out = Parallel.run(&cfg, &mut |s: Stats| reported.push(s));

        assert!(out.key.is_none());
        assert!(!reported.is_empty(), "進捗が届いていない");
        assert!(reported.iter().all(|s| s.elapsed <= out.stats.elapsed));
    }

    /// 1つのスレッドがパニックしたら、ほかのスレッドも呼び出し元もすぐ止まる。
    /// パニックしなければガードは何もしない
    #[rstest]
    fn stop_on_panic_guard(#[values(true, false)] panics: bool, #[values(2, 4)] threads: usize) {
        let stop = AtomicBool::new(false);
        let started = Instant::now();
        let give_up = Duration::from_millis(50); // パニックしない場合に他のスレッドが待つ時間
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            thread::scope(|s| {
                s.spawn(|| {
                    let _guard = StopOnPanic(&stop);
                    assert!(!panics, "わざとパニック");
                });
                for _ in 1..threads {
                    s.spawn(|| {
                        while !stop.load(Relaxed) && started.elapsed() < give_up {
                            thread::sleep(Duration::from_millis(1));
                        }
                    });
                }
            });
        }));
        assert_eq!(result.is_err(), panics, "パニックは呼び出し元に伝わる");
        assert_eq!(
            stop.load(Relaxed),
            panics,
            "停止フラグはパニックのときだけ立つ"
        );
        assert!(started.elapsed() < Duration::from_secs(5), "すぐ止まる");
    }

    /// どの1文字でも、見つかった鍵は ed25519-dalek で計算し直しても一致する
    #[rstest]
    fn found_key_really_matches(
        #[values(Backend::FAST, Backend::BASELINE)] backend: Backend,
        #[values(Find::Memmem, Find::Naive)] find: Find,
        #[values(false, true)] case_sensitive: bool,
    ) {
        check(2, "[A-Za-z0-9+/]", |c| {
            let cfg = SearchConfig {
                threads: 2,
                backend,
                find,
                time_limit: Some(Duration::from_secs(10)),
                ..SearchConfig::new(Some(Pattern::new(&c, case_sensitive).unwrap()))
            };
            let key = run(&cfg, |_| {}).key.expect("1文字ならすぐ見つかる");
            let recomputed = VanityKey::from_seed(*key.seed()).public_b64();
            prop_assert!(cfg.pattern.unwrap().is_match(&recomputed));
            Ok(())
        });
    }
}
