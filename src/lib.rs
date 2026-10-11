//! meikey — 公開鍵に好きな文字列を刻む ed25519 バニティ鍵ジェネレーター
//!
//! - [`key`] — シードから鍵を作り、OpenSSH 形式に書き出す
//! - [`pattern`] — 探す文字列と、公開鍵に含まれるかの判定
//! - [`search`] — 全コアで鍵を生成して当たりを探す
//! - [`format`] — 件数・時間の表示
//! - [`derive`] — シードから公開鍵をまとめて計算する（高速化ありと高速化前の計算を切り替えられる）

mod comb;
pub mod derive;
mod field;
pub mod format;
pub mod key;
pub mod pattern;
pub mod search;

pub use derive::{Backend, Compress, Mul};
pub use key::VanityKey;
pub use pattern::{Pattern, PatternError};
pub use search::{Outcome, SearchConfig, Stats};

// ---- テスト専用（tests/support/ の補助と独立実装を IT と共有する。テストは各ファイルの `mod tests`） ----
#[cfg(test)]
#[path = "../tests/support/property.rs"]
mod property;
#[cfg(test)]
#[path = "../tests/support/rfc8032.rs"]
mod rfc8032;
#[cfg(test)]
#[path = "../tests/support/ssh_format.rs"]
mod ssh_format;
