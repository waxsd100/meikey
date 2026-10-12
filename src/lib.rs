//! meikey — 公開鍵に好きな文字列を刻む ed25519 バニティ鍵ジェネレーター
//!
//! - [`model`] — データの型（探す文字列・鍵・計算方法・探索の設定と結果）と、その型だけで完結する処理
//! - [`controller`] — CLI（引数を読んで service と repository を呼ぶ）と、件数・時間の表示
//! - [`service`] — 並列探索・文字列の判定・シードから公開鍵の計算（AVX2・AVX-512 を含む）
//! - [`repository`] — 鍵を OpenSSH 形式のファイルに保存する

pub mod controller;
pub mod model;
pub mod repository;
pub mod service;

pub use model::backend::{Backend, Compress, Mul, Simd};
pub use model::key::VanityKey;
pub use model::pattern::{Find, Pattern, PatternError};
pub use model::search::{Outcome, SearchConfig, Stats};
pub use service::matcher::Matcher;

// ---- テスト専用（tests/support/ の補助と独立実装を IT と共有する。テストは各ファイルの `mod tests`） ----
#[cfg(test)]
#[path = "../tests/support/property.rs"]
mod property;
#[cfg(test)]
#[path = "../tests/support/rfc8032.rs"]
mod rfc8032;
#[cfg(all(test, target_arch = "x86_64"))]
#[path = "../tests/support/simd.rs"]
mod simd_support;
#[cfg(test)]
#[path = "../tests/support/ssh_format.rs"]
mod ssh_format;
