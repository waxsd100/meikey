//! 探索の設定と結果（探索そのものは [`crate::service::search`]）

use crate::model::backend::Backend;
use crate::model::key::VanityKey;
use crate::model::pattern::{Find, Pattern};
use std::num::NonZero;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// `None` なら判定せずに鍵を作り続ける
    pub pattern: Option<Pattern>,
    pub threads: usize,
    /// 公開鍵の計算方法
    pub backend: Backend,
    /// 公開鍵の中から文字列を探す方法
    pub find: Find,
    /// `true` なら当たっても止まらず、判定だけして回し続ける（判定込みの速度を測る `--bench` 用）
    pub bench: bool,
    /// この時間が経ったら打ち切る
    pub time_limit: Option<Duration>,
    /// 進捗コールバックを呼ぶ間隔
    pub progress_interval: Duration,
    /// 外から探索を止める合図（CPU と GPU で同時に探すとき、片方が見つけたらもう片方を止める）
    pub cancel: Cancel,
    /// シードの作り方
    pub seeds: SeedSource,
}

/// シードの作り方
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeedSource {
    /// 鍵ごとに ChaCha20 の出力から（既定）
    #[default]
    ChaCha20,
    /// OS の乱数で作った 32 バイト（CPU はスレッドごと、GPU は起動ごと）の先頭 8 バイトに、通し番号を足したもの
    /// （低セキュリティモード。乱数を作る手間が省ける）
    Counter,
}

/// `base` の先頭 8 バイト（リトルエンディアン）に `index` を足したシード（[`SeedSource::Counter`]）
pub fn counter_seed(base: &[u8; 32], index: u64) -> [u8; 32] {
    let mut s = *base;
    let head = u64::from_le_bytes(*base.first_chunk().expect("32 バイト"));
    s[..8].copy_from_slice(&head.wrapping_add(index).to_le_bytes());
    s
}

/// 探索を止める合図。clone したものどうしで共有する
#[derive(Debug, Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// 止める（何度呼んでもよい）
    pub fn cancel(&self) {
        self.0.store(true, Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Relaxed)
    }
}

impl SearchConfig {
    pub fn new(pattern: Option<Pattern>) -> Self {
        Self {
            pattern,
            threads: thread::available_parallelism().map_or(1, NonZero::get),
            backend: Backend::default(),
            find: Find::default(),
            bench: false,
            time_limit: None,
            progress_interval: Duration::from_secs(5),
            cancel: Cancel::default(),
            seeds: SeedSource::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub tries: u64,
    pub elapsed: Duration,
}

impl Stats {
    /// 平均の生成速度（鍵/秒）
    pub fn rate(&self) -> f64 {
        self.tries as f64 / self.elapsed.as_secs_f64().max(f64::EPSILON)
    }
}

#[derive(Debug)]
pub struct Outcome {
    /// 見つかった鍵（打ち切り・ベンチマークなら `None`）
    pub key: Option<VanityKey>,
    pub stats: Stats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use similar_asserts::assert_eq;

    /// 速度は 試行回数 / 経過秒。経過時間 0 でも 0 除算せず有限
    #[rstest]
    #[case::per_second(1000, 1.0, Some(1000.0))]
    #[case::half_second(150, 0.5, Some(300.0))]
    #[case::zero_elapsed_no_tries(0, 0.0, None)]
    #[case::zero_elapsed(1, 0.0, None)]
    #[case::zero_elapsed_max_tries(u64::MAX, 0.0, None)]
    fn rate(#[case] tries: u64, #[case] secs: f64, #[case] want: Option<f64>) {
        let r = Stats {
            tries,
            elapsed: Duration::from_secs_f64(secs),
        }
        .rate();
        assert_eq!(r.is_finite(), true);
        if let Some(want) = want {
            assert_eq!((r - want).abs() < 1e-9, true, "{r}");
        }
    }

    /// 通し番号のシードは、先頭 8 バイト（リトルエンディアン）に足すだけで、残りの 24 バイトは元のまま。
    /// 桁あふれは 2^64 で回る
    #[rstest]
    #[case::zero(0, 0, 0)]
    #[case::add(5, 3, 8)]
    #[case::wraps(u64::MAX, 2, 1)]
    fn counter_seed_adds_to_head(#[case] head: u64, #[case] index: u64, #[case] want: u64) {
        let mut base = [0xabu8; 32];
        base[..8].copy_from_slice(&head.to_le_bytes());
        let s = counter_seed(&base, index);
        assert_eq!(u64::from_le_bytes(*s.first_chunk().unwrap()), want);
        assert_eq!(&s[8..], &base[8..]);
    }

    /// 止める合図は clone したものどうしで共有し、新しく作ったものには届かない
    #[rstest]
    fn cancel_is_shared_by_clones(#[values(1, 2)] times: usize) {
        let a = Cancel::default();
        let b = a.clone();
        let other = Cancel::default();
        assert_eq!((a.is_cancelled(), b.is_cancelled()), (false, false));
        for _ in 0..times {
            b.cancel();
        }
        assert_eq!(
            (a.is_cancelled(), b.is_cancelled(), other.is_cancelled()),
            (true, true, false)
        );
    }
}
