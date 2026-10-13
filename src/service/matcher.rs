//! 探索ループで、公開鍵の base64 に文字列が含まれるかを速く判定する
//!
//! 結果は [`Pattern::find`] と同じ（最初に一致した位置）。探す方法は [`Find`] で選ぶ。

use crate::model::key::{PUBLIC_B64_LEN, PublicB64};
use crate::model::pattern::{Find, Pattern, SEARCH_START};
use memchr::memmem;
use std::ops::Range;

/// スレッドごとに1つ作る判定器
#[derive(Debug, Clone)]
pub struct Matcher(Inner);

// memmem::Finder の大きさはターゲットで変わり、警告が出ない環境もあるので expect ではなく allow
#[allow(
    clippy::large_enum_variant,
    reason = "スレッドに1つしか作らないので大きさは問題にならない（Box にすると判定のたびに間接参照が増える）"
)]
#[derive(Debug, Clone)]
enum Inner {
    Naive(Pattern),
    Memmem {
        finder: memmem::Finder<'static>,
        /// 公開鍵を小文字にそろえてから探す（大小文字を区別しないとき）
        fold: bool,
    },
}

impl Matcher {
    pub fn new(pattern: &Pattern, find: Find) -> Matcher {
        Matcher(match find {
            Find::Naive => Inner::Naive(pattern.clone()),
            Find::Memmem => {
                // 大小文字を区別しないなら、探す文字列も公開鍵も小文字にそろえてから探す
                let fold = !pattern.case_sensitive();
                let needle: Vec<u8> = if fold {
                    pattern.needle().iter().map(|&c| fold_case(c)).collect()
                } else {
                    pattern.needle().to_vec()
                };
                Inner::Memmem {
                    finder: memmem::Finder::new(&needle).into_owned(),
                    fold,
                }
            }
        })
    }

    /// 公開鍵の base64 の中で一致した位置
    #[inline]
    pub fn find(&self, b64: &PublicB64) -> Option<Range<usize>> {
        match &self.0 {
            Inner::Naive(p) => p.find(b64),
            Inner::Memmem { finder, fold } => {
                let hay: &[u8; PUBLIC_B64_LEN - SEARCH_START] =
                    b64[SEARCH_START..].as_array().expect("長さは定数");
                let folded;
                let hay = if *fold {
                    folded = hay.map(fold_case);
                    &folded
                } else {
                    hay
                };
                let i = SEARCH_START + finder.find(hay)?;
                Some(i..i + finder.needle().len())
            }
        }
    }

    #[inline]
    pub fn is_match(&self, b64: &PublicB64) -> bool {
        self.find(b64).is_some()
    }
}

/// base64 の文字を小文字にそろえる。base64 の文字表では、英大文字だけが 0x20 のビットを持たず、
/// 英小文字・数字・`+`・`/` は持っているので、OR 1回で `to_ascii_lowercase` と同じになる
/// （分岐がないので、43文字ぶんがまとめて SIMD 命令になる）
#[inline(always)]
fn fold_case(c: u8) -> u8 {
    c | 0x20
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::pattern::tests::{ALPHABET, PREFIX_ONLY, SAMPLE, b64_with};
    use crate::property::check;
    use proptest::prelude::*;
    use rstest::rstest;
    use similar_asserts::assert_eq;

    fn matcher(s: &str, cs: bool, how: Find) -> Matcher {
        Matcher::new(&Pattern::new(s, cs).unwrap(), how)
    }

    #[rstest]
    #[case::insensitive(SAMPLE, "wako", false, Some(56..60))] // "WaKo"
    #[case::insensitive_upper(SAMPLE, "WAKO", false, Some(56..60))]
    #[case::sensitive(SAMPLE, "WaKo", true, Some(56..60))]
    #[case::sensitive_miss(SAMPLE, "wako", true, None)]
    #[case::at_end(SAMPLE, "Gy9", true, Some(65..68))]
    #[case::at_start(SAMPLE, "IQK2", true, Some(25..29))]
    #[case::miss(SAMPLE, "zzzz", false, None)]
    // 固定部分の中にある文字列には一致させない（どの鍵でも同じなので意味がない）
    #[case::prefix_aaaac3(PREFIX_ONLY, "AAAAC3", true, None)]
    #[case::prefix_middle(PREFIX_ONLY, "NzaC1lZDI1", true, None)]
    #[case::prefix_nte5(PREFIX_ONLY, "NTE5", true, None)]
    #[case::right_after_prefix(PREFIX_ONLY, "0000", true, Some(25..29))]
    fn find(
        #[case] b64: &PublicB64,
        #[case] s: &str,
        #[case] cs: bool,
        #[case] want: Option<Range<usize>>,
        #[values(Find::Memmem, Find::Naive)] how: Find,
    ) {
        let m = matcher(s, cs, how);
        assert_eq!(m.find(b64), want);
        assert_eq!(m.is_match(b64), want.is_some());
    }

    /// base64 の64文字すべてで、OR 0x20 は to_ascii_lowercase と同じ
    #[rstest]
    fn fold_case_is_lowercase_for_base64() {
        for c in ALPHABET.bytes() {
            assert_eq!(fold_case(c), c.to_ascii_lowercase(), "{}", c as char);
        }
        assert_eq!(ALPHABET.len(), 64);
    }

    /// どちらの判定器も [`Pattern::find`]（単純な文字列検索と同じことを model のテストで確かめている）と
    /// 同じ結果になる。`planted` なら探す文字列を必ずどこかに埋め込む。長い文字列は埋め込まないとまず一致しない
    #[rstest]
    fn matches_pattern_find(
        #[values("[A-Za-z0-9+/]{1,6}", "[A-Za-z0-9+/]{7,42}")] needles: &'static str,
        #[values(false, true)] cs: bool,
        #[values(false, true)] planted: bool,
        #[values(Find::Memmem, Find::Naive)] how: Find,
    ) {
        let strategy = ("[A-Za-z0-9+/]{43}", needles, any::<usize>());
        check(1024, strategy, |(rest, needle, pos)| {
            let b64 = b64_with(&needle, planted.then_some(pos), &rest);
            let p = Pattern::new(&needle, cs).unwrap();
            let m = Matcher::new(&p, how);
            prop_assert_eq!(m.find(&b64), p.find(&b64));
            prop_assert_eq!(m.is_match(&b64), p.is_match(&b64));
            if planted {
                prop_assert!(m.is_match(&b64), "埋め込んだのに見つからない");
            }
            Ok(())
        });
    }
}
