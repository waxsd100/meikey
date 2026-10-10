//! proptest を `#[rstest]` のテストから回すための補助（テスト専用。src/ のユニットテストと IT で共有）
//!
//! `proptest!` マクロは中で `#[test]` を使うので、テストをすべて `#[rstest]` にするために、
//! proptest のテストランナーを直接呼ぶ。動きは `proptest!` と同じで、失敗すると反例を最小化して
//! パニックし、その反例は `proptest-regressions/` に保存されて、次回から必ず先に試される。
//!
//! 試行回数はテストごとに決め、環境変数 `MEIKEY_PROPTEST_SCALE` で全体を何倍にもできる。
//! 例: `MEIKEY_PROPTEST_SCALE=100`（`cargo make proptest-deep` はこれを使う）、
//! `MEIKEY_PROPTEST_SCALE=0.1`（ミューテーションテスト用に減らす）

#![allow(dead_code)] // 読み込む側によって使う関数が違う

use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestCaseResult, TestRunner};

/// `strategy` が作るランダムな値で `test` を `cases` 回（× `MEIKEY_PROPTEST_SCALE`）試す。
/// `test` の中では `prop_assert!` などで確かめ、最後に `Ok(())` を返す
#[track_caller]
pub fn check<S: Strategy>(cases: u32, strategy: S, test: impl Fn(S::Value) -> TestCaseResult) {
    let config = Config {
        cases: scaled(cases),
        // 反例の保存先（proptest-regressions/ の下の、呼び出したファイルと同じ名前）を決める
        source_file: Some(std::panic::Location::caller().file()),
        ..Config::default()
    };
    let mut runner = TestRunner::new(config);
    if let Err(e) = runner.run(&strategy, test) {
        panic!("{e}\n{runner}");
    }
}

/// 試行回数に `MEIKEY_PROPTEST_SCALE` を掛ける（0.1 のような小数も使える。最低 1 回）
fn scaled(cases: u32) -> u32 {
    let scale: f64 = std::env::var("MEIKEY_PROPTEST_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|s: &f64| s.is_finite() && *s > 0.0)
        .unwrap_or(1.0);
    ((f64::from(cases) * scale).round() as u32).max(1)
}
