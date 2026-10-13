//! CLI のエラー（実行の失敗と、引数の値の誤り）

use super::interactive::AskError;
use crate::lang::{t, tf};
use crate::model::key::InvalidComment;
use crate::model::pattern::PatternError;
use crate::repository::key_file::{KeyFileError, pub_path};
#[cfg(feature = "cuda")]
use crate::service::gpu::GpuError;
use std::path::PathBuf;

/// 実行に失敗した理由（表示して、終了コード 1 で終わる）
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    /// 探す文字列が使えない
    #[error(transparent)]
    Pattern(#[from] PatternError),
    /// コメントに使えない文字がある
    #[error(transparent)]
    Comment(#[from] InvalidComment),
    /// 出力先に鍵のファイル（秘密鍵か `.pub`）が既にあり、--force を付けていない
    #[error("{}", tf!(
        "既に存在: {}（--force で上書き）",
        "already exists: {} (--force to overwrite)",
        .0.display()
    ))]
    OutputExists(PathBuf),
    /// 出力先のディレクトリに書けない（探す前に確かめる）
    #[error("{}", tf!("保存できない: {}", "cannot save: {}", .0))]
    NotWritable(#[source] KeyFileError),
    /// 出力先に鍵のファイルがあるか調べられない
    #[error(transparent)]
    KeyFile(#[from] KeyFileError),
    /// 鍵が見つからずに探索が終わった（時間切れなど）
    #[error("{}", t!("鍵が見つからなかった", "no key was found"))]
    NotFound,
    /// 見つけた鍵を出力先に保存できなかったので、代わりに `private`（公開鍵は `<private>.pub`）に保存した
    #[error("{}", tf!(
        "{}。代わりに {} に保存した（公開鍵は {}）",
        "{}. Saved it to {} instead (public key: {})",
        .source,
        .private.display(),
        pub_path(.private).display()
    ))]
    SavedElsewhere {
        #[source]
        source: KeyFileError,
        private: PathBuf,
    },
    /// 見つけた鍵を、出力先にも代わりの場所（今のディレクトリ・一時ディレクトリ）にも保存できなかった
    #[error("{}", tf!(
        "{}。代わりの場所（今のディレクトリ・一時ディレクトリ）にも保存できなかった",
        "{}. Could not save it elsewhere either (the current directory, the temporary directory)",
        .0
    ))]
    NotSaved(#[source] KeyFileError),
    /// GPU を使えない
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// 対話の答えから組み立てた引数を読めない
    #[error(transparent)]
    Args(#[from] clap::Error),
    /// 対話で質問できない（端末の読み書きなど）
    #[error(transparent)]
    Ask(#[from] AskError),
}

/// 引数の値の誤り（clap が「'--threads <THREADS>' の値 '0' が正しくない: …」の「…」に使う。
/// 対話では入れ直してもらう理由に使う）
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ValueError {
    /// `lo` 以上 `hi` 以下の整数でない
    #[error("{}", tf!("{}〜{} の整数が必要", "must be an integer from {} to {}", .lo, .hi))]
    NotInRange { lo: u32, hi: u32 },
    /// 秒数として読めない
    #[error("{}", t!("秒数ではない", "not a number of seconds"))]
    NotSeconds,
    /// 0 以下の秒数（か、Duration に入らないほど大きい秒数）
    #[error("{}", t!("0 より大きい秒数が必要", "must be greater than 0 seconds"))]
    NotPositive,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::{Lang, set};
    use rstest::rstest;
    use similar_asserts::assert_eq;
    use std::io;

    /// 鍵のファイルの読み書きのエラー（テスト用）
    fn io_error(path: &str, message: &str) -> KeyFileError {
        KeyFileError::Io {
            path: path.into(),
            source: io::Error::other(message.to_string()),
        }
    }

    /// 日本語・英語の表示（ほかのエラーを包むものは、その説明も含める）
    #[rstest]
    #[case::exists(
        CliError::OutputExists("k".into()),
        "既に存在: k（--force で上書き）",
        "already exists: k (--force to overwrite)"
    )]
    #[case::not_writable(
        CliError::NotWritable(io_error("d", "x")),
        "保存できない: d: x",
        "cannot save: d: x"
    )]
    #[case::key_file(CliError::KeyFile(io_error("k", "x")), "k: x", "k: x")]
    #[case::not_found(CliError::NotFound, "鍵が見つからなかった", "no key was found")]
    #[case::saved_elsewhere(
        CliError::SavedElsewhere { source: io_error("o/k", "x"), private: "k-1".into() },
        "o/k: x。代わりに k-1 に保存した（公開鍵は k-1.pub）",
        "o/k: x. Saved it to k-1 instead (public key: k-1.pub)"
    )]
    #[case::not_saved(
        CliError::NotSaved(io_error("o/k", "x")),
        "o/k: x。代わりの場所（今のディレクトリ・一時ディレクトリ）にも保存できなかった",
        "o/k: x. Could not save it elsewhere either (the current directory, the temporary directory)"
    )]
    #[case::pattern(
        CliError::Pattern(PatternError::Empty),
        "文字列が空",
        "the text is empty"
    )]
    #[case::comment(
        CliError::Comment(InvalidComment('\n')),
        "コメントに使えない文字: '\\n'（改行・NUL は不可）",
        "invalid character in the comment: '\\n' (no line breaks or NUL)"
    )]
    fn cli_error_display(#[case] e: CliError, #[case] ja: &str, #[case] en: &str) {
        assert_eq!(e.to_string(), ja);
        set(Lang::En);
        assert_eq!(e.to_string(), en);
    }

    /// 包んだエラーは source でたどれる
    #[rstest]
    #[case::not_writable(CliError::NotWritable(io_error("d", "x")))]
    #[case::saved_elsewhere(CliError::SavedElsewhere { source: io_error("d", "x"), private: "k".into() })]
    #[case::not_saved(CliError::NotSaved(io_error("d", "x")))]
    fn wrapped_errors_have_a_source(#[case] e: CliError) {
        let source = std::error::Error::source(&e).map(|s| s.to_string());
        assert_eq!(source.as_deref(), Some("d: x"));
    }

    #[rstest]
    #[case::range(ValueError::NotInRange { lo: 2, hi: 16 }, "2〜16 の整数が必要", "must be an integer from 2 to 16")]
    #[case::seconds(ValueError::NotSeconds, "秒数ではない", "not a number of seconds")]
    #[case::positive(
        ValueError::NotPositive,
        "0 より大きい秒数が必要",
        "must be greater than 0 seconds"
    )]
    fn value_error_display(#[case] e: ValueError, #[case] ja: &str, #[case] en: &str) {
        assert_eq!(e.to_string(), ja);
        set(Lang::En);
        assert_eq!(e.to_string(), en);
    }
}
