//! IT（統合テスト）: ビルドした meikey 本体を起動したり、OpenSSH などの外部ツールと突き合わせたりする
//!
//! IT は1つのバイナリにまとめている。`tests/*.rs` をファイルごとに分けると、ファイルの数だけ
//! テストバイナリのリンクが走り、`cargo test` もバイナリを1つずつ順に実行するため、
//! あるファイルの遅いテストを待つ間ほかのファイルのテストが CPU を使えない。
//!
//! どのモジュールも、ファイルの前半がヘルパーで、テストは同じファイルの末尾の `mod tests` にある。
//!
//! - [`cli`] — CLI の振る舞い（終了コード・エラー・ファイル保護・出力）
//! - [`keys_work`] — 作った鍵が本当に SSH 鍵として使えるか（読める・署名できる・OpenSSH と同じバイト列）
//! - [`encrypted`] — パスフレーズを付けた鍵が仕様どおりに暗号化され、そのパスフレーズでだけ開けるか
//! - [`agent`] — ssh-agent に入れて使えるか（Unix だけ）
//! - [`sshd`] — 実際に sshd にログインできるか（Unix で Docker が使えるときだけ）
//! - [`interactive`] — 引数なしで端末から起動したときの対話（Unix だけ）
//! - [`authorized_keys`] — オプション付き・ほかの行と並べた authorized_keys で使えるか
//! - [`rfc8032`] — 参照実装そのものが RFC 8032 どおりか
//! - [`ssh_format`] — SSH の形式の独立実装そのものが仕様どおりか
//! - [`common`] — 共通のヘルパー（meikey の起動、鍵ファイル、tests/support/ の独立実装）

#[cfg(unix)]
mod agent;
mod authorized_keys;
mod cli;
mod common;
mod encrypted;
#[cfg(unix)]
mod interactive;
mod keys_work;
mod rfc8032;
mod ssh_format;
#[cfg(unix)]
mod sshd;
