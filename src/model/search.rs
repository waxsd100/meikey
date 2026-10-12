//! 探索の設定と結果（探索そのものは [`crate::service::search`]）

use crate::model::backend::Backend;
use crate::model::key::VanityKey;
use crate::model::pattern::{Find, Pattern};
use std::num::NonZero;
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
        assert!(r.is_finite());
        if let Some(want) = want {
            assert!((r - want).abs() < 1e-9, "{r}");
        }
    }
}
