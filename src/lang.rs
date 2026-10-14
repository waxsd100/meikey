//! 表示する言語: ロケールが日本語なら日本語、それ以外は英語。ロケールは、環境変数（LANGUAGE・LC_ALL・
//! LC_MESSAGES・LANG の順）にあればそれ、無ければ OS の設定（Windows では表示言語）
//!
//! 表示する文はどれも、その場で日本語と英語を並べて書き、[`t!`]（そのまま）か [`tf!`]（`format!` する）で
//! 今の言語のほうを選ぶ。
//!
//! テストでは OS のロケールを見ず、スレッドごとに日本語から始まる（[`set`] で切り替える）。
//! テストはスレッドごとに並列に走るので、ほかのテストの言語に影響しない

/// 表示する言語
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Ja,
    En,
}

impl Lang {
    /// ロケールの名前（`ja-JP`・`ja_JP.UTF-8`・`en-US`・`C` など）から。日本語（`ja` で始まる）でなければ英語
    pub fn from_locale(locale: &str) -> Lang {
        let language = locale.split(['-', '_', '.', '@']).next().unwrap_or("");
        match language.eq_ignore_ascii_case("ja") {
            true => Lang::Ja,
            false => Lang::En,
        }
    }
}

/// ロケールの環境変数（POSIX と同じ LANGUAGE・LC_ALL・LC_MESSAGES・LANG の順で、空でない最初のもの）からの言語。
/// LANGUAGE は「:」区切りの最初のもの。Unix では OS の設定もこの環境変数なので同じになり、Windows でも
/// 環境変数で選べる（Git Bash などが LANG を設定する）
pub fn from_env(get: impl Fn(&str) -> Option<String>) -> Option<Lang> {
    ["LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"]
        .into_iter()
        .filter_map(&get)
        .find_map(|v| v.split(':').find(|p| !p.is_empty()).map(Lang::from_locale))
}

/// 今の言語（最初に1回だけ決める。ロケールがわからなければ英語）
#[cfg(not(test))]
pub fn current() -> Lang {
    static LANG: std::sync::OnceLock<Lang> = std::sync::OnceLock::new();
    *LANG.get_or_init(|| {
        from_env(|k| std::env::var(k).ok())
            .unwrap_or_else(|| sys_locale::get_locale().map_or(Lang::En, |l| Lang::from_locale(&l)))
    })
}

#[cfg(test)]
thread_local! {
    static LANG: std::cell::Cell<Lang> = const { std::cell::Cell::new(Lang::Ja) };
}

/// 今の言語（テストではこのスレッドの言語。最初は日本語）
#[cfg(test)]
pub fn current() -> Lang {
    LANG.get()
}

/// このスレッドの言語を変える（テスト用）
#[cfg(test)]
pub fn set(lang: Lang) {
    LANG.set(lang);
}

/// 今の言語の文（`&'static str`）。`t!("日本語", "English")`
macro_rules! t {
    ($ja:literal, $en:literal $(,)?) => {
        match $crate::lang::current() {
            $crate::lang::Lang::Ja => $ja,
            $crate::lang::Lang::En => $en,
        }
    };
}

/// 今の言語の文を `format!` する（`String`）。`tf!("{n} 鍵", "{n} keys")`、引数も `format!` と同じに渡せる
macro_rules! tf {
    ($ja:literal, $en:literal $(, $arg:expr)* $(,)?) => {
        match $crate::lang::current() {
            $crate::lang::Lang::Ja => format!($ja $(, $arg)*),
            $crate::lang::Lang::En => format!($en $(, $arg)*),
        }
    };
}

pub(crate) use {t, tf};

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use similar_asserts::assert_eq;

    /// 環境変数は LANGUAGE・LC_ALL・LC_MESSAGES・LANG の順に、空でない最初のものを使う
    #[rstest]
    #[case::none(&[], None)]
    #[case::lang(&[("LANG", "ja_JP.UTF-8")], Some(Lang::Ja))]
    #[case::lc_all_first(&[("LC_ALL", "en_US.UTF-8"), ("LANG", "ja_JP.UTF-8")], Some(Lang::En))]
    #[case::language_first(&[("LANGUAGE", "ja:en"), ("LC_ALL", "C")], Some(Lang::Ja))]
    #[case::empty_is_skipped(&[("LC_ALL", ""), ("LANG", "ja_JP.UTF-8")], Some(Lang::Ja))]
    #[case::lc_messages(&[("LC_MESSAGES", "ja_JP"), ("LANG", "C")], Some(Lang::Ja))]
    #[case::c(&[("LC_ALL", "C")], Some(Lang::En))]
    fn from_env_order(#[case] vars: &[(&str, &str)], #[case] want: Option<Lang>) {
        let get = |k: &str| {
            vars.iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        };
        assert_eq!(from_env(get), want);
    }

    /// 言語が日本語のロケールだけ日本語。地域・文字コード・修飾子が付いていても、大文字でもよい
    #[rstest]
    #[case::bcp47("ja-JP", Lang::Ja)]
    #[case::posix("ja_JP.UTF-8", Lang::Ja)]
    #[case::bare("ja", Lang::Ja)]
    #[case::upper("JA-jp", Lang::Ja)]
    #[case::modifier("ja_JP@euro", Lang::Ja)]
    #[case::english("en-US", Lang::En)]
    #[case::c("C", Lang::En)]
    #[case::posix_c("POSIX", Lang::En)]
    #[case::c_utf8("C.UTF-8", Lang::En)]
    #[case::other("zh-Hans-CN", Lang::En)]
    #[case::starts_with_ja("jam", Lang::En)]
    #[case::empty("", Lang::En)]
    fn from_locale(#[case] locale: &str, #[case] want: Lang) {
        assert_eq!(Lang::from_locale(locale), want);
    }

    /// t!・tf! は今の言語のほうを選ぶ（テストのスレッドは日本語から始まる）
    #[rstest]
    fn picks_current_language() {
        let n = 3;
        assert_eq!(current(), Lang::Ja);
        assert_eq!(t!("鍵", "key"), "鍵");
        assert_eq!(tf!("{n} 鍵", "{n} keys"), "3 鍵");
        set(Lang::En);
        assert_eq!(t!("鍵", "key"), "key");
        assert_eq!(tf!("{} 鍵", "{} keys", n + 1), "4 keys");
    }
}
