//! 探す文字列と、公開鍵に含まれるかの判定（探索ループで速く判定するものは [`crate::service::matcher`]）

use crate::model::key::{FIXED_PREFIX, PUBLIC_B64_LEN, PublicB64};
use std::fmt;
use std::ops::Range;

/// 判定に使う範囲の先頭（固定部分の直後）
pub(crate) const SEARCH_START: usize = FIXED_PREFIX.len();

/// 固定部分の直後の1文字は A〜P しか出ないので、自由に変わるのはその次から
const FREE_LEN: usize = PUBLIC_B64_LEN - SEARCH_START - 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternError {
    Empty,
    InvalidChar(char),
    TooLong { len: usize, max: usize },
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "文字列が空"),
            Self::InvalidChar(c) => {
                write!(f, "使えない文字: '{c}'（使える文字: A-Z a-z 0-9 + /）")
            }
            Self::TooLong { len, max } => {
                write!(f, "文字列が長すぎる: {len} 文字（最大 {max} 文字）")
            }
        }
    }
}

impl std::error::Error for PatternError {}

/// 公開鍵に含めたい文字列
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern {
    needle: Box<[u8]>,
    case_sensitive: bool,
}

impl Pattern {
    /// 指定できる最大文字数
    pub const MAX_LEN: usize = FREE_LEN;

    pub fn new(s: &str, case_sensitive: bool) -> Result<Self, PatternError> {
        (!s.is_empty()).ok_or(PatternError::Empty)?;
        if let Some(c) = s.chars().find(|&c| !is_base64_char(c)) {
            return Err(PatternError::InvalidChar(c));
        }
        (s.len() <= Self::MAX_LEN).ok_or(PatternError::TooLong {
            len: s.len(),
            max: Self::MAX_LEN,
        })?;
        Ok(Self {
            needle: s.as_bytes().into(),
            case_sensitive,
        })
    }

    /// 探す文字列のバイト列（検証済みの ASCII）
    pub(crate) fn needle(&self) -> &[u8] {
        &self.needle
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.needle[..]).expect("検証済みの ASCII")
    }

    pub fn case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    /// 公開鍵の base64 の中で一致した位置
    pub fn find(&self, b64: &PublicB64) -> Option<Range<usize>> {
        let hit = b64[SEARCH_START..]
            .windows(self.needle.len())
            .find(|w| self.eq(w))?;
        b64.subslice_range(hit).map(Into::into)
    }

    pub fn is_match(&self, b64: &PublicB64) -> bool {
        self.find(b64).is_some()
    }

    fn eq(&self, w: &[u8]) -> bool {
        if self.case_sensitive {
            w == &self.needle[..]
        } else {
            w.eq_ignore_ascii_case(&self.needle[..])
        }
    }

    /// 1回の試行で当たる確率
    pub fn probability(&self) -> f64 {
        let n = self.needle.len() as i32;
        // 大小文字を区別しないなら、英字1文字につき当たりが2通りになる
        let variants = if self.case_sensitive {
            0
        } else {
            self.needle
                .iter()
                .filter(|c| c.is_ascii_alphabetic())
                .count() as i32
        };
        let positions = (FREE_LEN + 1 - self.needle.len()) as f64;
        positions * 2f64.powi(variants) / 64f64.powi(n)
    }

    /// 当たりが出るまでの期待試行回数
    pub fn expected_tries(&self) -> f64 {
        self.probability().recip()
    }
}

/// 公開鍵の中から文字列を探す方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Find {
    /// memchr の `memmem`（SIMD を使う部分文字列検索。ripgrep と同じもの）
    #[default]
    Memmem,
    /// 1文字ずつずらして比べる（高速化前）
    Naive,
}

impl Find {
    pub fn name(self) -> &'static str {
        match self {
            Find::Memmem => "memmem",
            Find::Naive => "総当たり",
        }
    }
}

impl fmt::Display for Find {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "検索={}", self.name())
    }
}

fn is_base64_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '+' || c == '/'
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use rstest::rstest;

    /// 実際に生成した鍵
    pub(crate) const SAMPLE: &PublicB64 =
        b"AAAAC3NzaC1lZDI1NTE5AAAAIIQK2/0eXLDku+cWkIDsqHQOwFFTEDDoWaKofXuV/Gy9";

    fn pat(s: &str, cs: bool) -> Pattern {
        Pattern::new(s, cs).unwrap()
    }

    const MSG_EMPTY: &str = "文字列が空";

    /// 受け付けるか（受け付けるなら大小文字の設定もそのまま保つ）、拒否するならそのエラーとメッセージ
    #[rstest]
    #[case::lowercase("wakokara", None)]
    #[case::mixed_case("WaKoKaRa", None)]
    #[case::digits("gack951", None)]
    #[case::symbols("a+b/c", None)]
    #[case::one_char("a", None)]
    #[case::max_minus_1(&"a".repeat(41), None)]
    #[case::max_42(&"a".repeat(42), None)]
    #[case::empty("", Some((PatternError::Empty, MSG_EMPTY)))]
    #[case::hyphen("wako-kara", Some((PatternError::InvalidChar('-'), "使えない文字: '-'（使える文字: A-Z a-z 0-9 + /）")))]
    #[case::space("wako kara", Some((PatternError::InvalidChar(' '), "使えない文字: ' '（使える文字: A-Z a-z 0-9 + /）")))]
    #[case::padding("wako=", Some((PatternError::InvalidChar('='), "使えない文字: '='（使える文字: A-Z a-z 0-9 + /）")))]
    #[case::japanese("わこから", Some((PatternError::InvalidChar('わ'), "使えない文字: 'わ'（使える文字: A-Z a-z 0-9 + /）")))]
    #[case::too_long(&"x".repeat(43), Some((PatternError::TooLong { len: 43, max: 42 }, "文字列が長すぎる: 43 文字（最大 42 文字）")))]
    fn new_validates(
        #[case] s: &str,
        #[case] error: Option<(PatternError, &str)>,
        #[values(false, true)] cs: bool,
    ) {
        assert_eq!(Pattern::MAX_LEN, 42);
        match (Pattern::new(s, cs), error) {
            (Ok(p), None) => {
                assert_eq!(p.as_str(), s);
                assert_eq!(p.case_sensitive(), cs);
            }
            (Err(e), Some((want, message))) => {
                assert_eq!(e, want);
                assert_eq!(e.to_string(), message);
            }
            (got, want) => panic!("{s:?}: {got:?}（期待 {want:?}）"),
        }
    }

    /// 固定部分（先頭25文字）だけで、後ろは全部 '0' の公開鍵
    pub(crate) const PREFIX_ONLY: &PublicB64 =
        b"AAAAC3NzaC1lZDI1NTE5AAAAI0000000000000000000000000000000000000000000";

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
    ) {
        let p = pat(s, cs);
        assert_eq!(p.find(b64), want);
        assert_eq!(p.is_match(b64), want.is_some());
    }

    /// --bench の表示。既定は memmem
    #[rstest]
    #[case::memmem(Find::Memmem, "検索=memmem")]
    #[case::naive(Find::Naive, "検索=総当たり")]
    fn find_display(#[case] how: Find, #[case] want: &str) {
        assert_eq!(how.to_string(), want);
        assert_eq!(Find::default(), Find::Memmem);
    }

    /// 大小文字を区別しないときの期待試行回数と、区別したときに何倍になるか（英字1文字につき2倍）
    #[rstest]
    // 7文字なら入る位置は 42-7+1 = 36 か所
    #[case::seven_letters("abcdefg", 64f64.powi(7) / (2f64.powi(7) * 36.0), 128.0)]
    #[case::letters_and_digits("gack951", 64f64.powi(7) / (2f64.powi(4) * 36.0), 16.0)]
    #[case::eight_letters("wakokara", 64f64.powi(8) / (2f64.powi(8) * 35.0), 256.0)]
    #[case::one_letter("W", 64.0 / (2.0 * 42.0), 2.0)]
    #[case::digits_only("2026", 64f64.powi(4) / 39.0, 1.0)]
    #[case::symbols_only("+/+/", 64f64.powi(4) / 39.0, 1.0)]
    fn expected_tries(#[case] s: &str, #[case] insensitive: f64, #[case] ratio: f64) {
        let (ci, cs) = (
            pat(s, false).expected_tries(),
            pat(s, true).expected_tries(),
        );
        assert!(
            (ci / insensitive - 1.0).abs() < 1e-12,
            "{ci}（期待 {insensitive}）"
        );
        assert!((cs / ci - ratio).abs() < 1e-9, "{}", cs / ci);
    }

    // ---- proptest ----

    use crate::property::check;
    use proptest::prelude::*;

    pub(crate) const ALPHABET: &str =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    /// 仕様: 1〜42 文字で、全部が base64 の文字表（RFC 4648 Table 1）の文字なら受け付ける
    fn expected_error(s: &str) -> Option<PatternError> {
        if s.is_empty() {
            Some(PatternError::Empty)
        } else if let Some(c) = s.chars().find(|c| !ALPHABET.contains(*c)) {
            Some(PatternError::InvalidChar(c))
        } else if s.len() > 42 {
            Some(PatternError::TooLong {
                len: s.len(),
                max: 42,
            })
        } else {
            None
        }
    }

    /// 単純な文字列検索で同じことをする（固定部分の後ろだけを探す）
    fn naive_find(b64: &str, needle: &str, cs: bool) -> Option<Range<usize>> {
        let (hay, needle) = if cs {
            (b64[SEARCH_START..].to_owned(), needle.to_owned())
        } else {
            (
                b64[SEARCH_START..].to_ascii_lowercase(),
                needle.to_ascii_lowercase(),
            )
        };
        hay.find(&needle)
            .map(|i| SEARCH_START + i..SEARCH_START + i + needle.len())
    }

    /// 公開鍵の base64 と同じ形（固定の 25 文字 + 任意の 43 文字）。
    /// `needle` を `pos` に埋め込むかどうかも選べるようにして、一致する例を十分に作る
    pub(crate) fn b64_with(needle: &str, plant: Option<usize>, rest: &str) -> PublicB64 {
        let mut free: Vec<u8> = rest.as_bytes().to_vec();
        if let Some(pos) = plant {
            let pos = pos % (free.len() - needle.len() + 1);
            free[pos..pos + needle.len()].copy_from_slice(needle.as_bytes());
        }
        let s = [FIXED_PREFIX.as_bytes(), &free].concat();
        s.try_into().unwrap()
    }

    /// 受け付けるのは、ちょうど仕様（1〜42 文字で、全部が base64 の文字表の文字）を満たす文字列だけ
    #[rstest]
    #[case::alphabet_only("[A-Za-z0-9+/]{0,50}")]
    #[case::with_similar_chars("[A-Za-z0-9+/=\\- _.]{0,45}")]
    #[case::any_text(".{0,50}")]
    fn new_accepts_exactly_valid_patterns(
        #[case] texts: &'static str,
        #[values(false, true)] cs: bool,
    ) {
        check(1024, texts, |s| {
            match (Pattern::new(&s, cs), expected_error(&s)) {
                (Ok(p), None) => prop_assert_eq!(p.as_str(), s.as_str()),
                (Err(e), Some(want)) => prop_assert_eq!(e, want),
                (got, want) => prop_assert!(false, "{:?} に対して {:?}（期待 {:?}）", s, got, want),
            }
            Ok(())
        });
    }

    /// 単純な文字列検索と同じ結果になる。`planted` なら探す文字列を必ずどこかに埋め込む。
    /// 長い文字列は埋め込まないとまず一致しない
    #[rstest]
    fn find_matches_naive_search(
        #[values("[A-Za-z0-9+/]{1,6}", "[A-Za-z0-9+/]{7,42}")] needles: &'static str,
        #[values(false, true)] cs: bool,
        #[values(false, true)] planted: bool,
    ) {
        let strategy = ("[A-Za-z0-9+/]{43}", needles, any::<usize>());
        check(1024, strategy, |(rest, needle, pos)| {
            let b64 = b64_with(&needle, planted.then_some(pos), &rest);
            let s = std::str::from_utf8(&b64).unwrap();
            let p = pat(&needle, cs);
            let want = naive_find(s, &needle, cs);
            prop_assert_eq!(p.find(&b64), want.clone());
            prop_assert_eq!(p.is_match(&b64), want.is_some());
            if planted {
                prop_assert!(p.is_match(&b64), "埋め込んだのに見つからない");
            }
            Ok(())
        });
    }

    /// 大小文字を区別しないなら、パターンの大小文字をどう変えても結果は同じ
    #[rstest]
    fn case_insensitive_ignores_case_of_pattern(#[values(false, true)] planted: bool) {
        let strategy = (
            "[A-Za-z0-9+/]{43}",
            "[A-Za-z0-9+/]{1,6}",
            prop::collection::vec(any::<bool>(), 6),
            any::<usize>(),
        );
        check(2048, strategy, |(rest, needle, flips, pos)| {
            let b64 = b64_with(&needle, planted.then_some(pos), &rest);
            let recased: String = needle
                .chars()
                .zip(&flips)
                .map(|(c, &f)| {
                    if f {
                        c.to_ascii_uppercase()
                    } else {
                        c.to_ascii_lowercase()
                    }
                })
                .collect();
            prop_assert_eq!(
                pat(&needle, false).find(&b64),
                pat(&recased, false).find(&b64)
            );
            Ok(())
        });
    }

    /// 期待試行回数は確率の逆数で、大小文字を区別すると増え、1 文字足すと必ず増える
    #[rstest]
    fn expected_tries_is_consistent() {
        check(
            4096,
            ("[A-Za-z0-9+/]{1,41}", "[A-Za-z0-9+/]"),
            |(needle, extra)| {
                let (ci, cs) = (pat(&needle, false), pat(&needle, true));
                prop_assert!((ci.probability() * ci.expected_tries() - 1.0).abs() < 1e-9);
                prop_assert!(cs.expected_tries() >= ci.expected_tries());
                let longer = pat(&format!("{needle}{extra}"), false);
                prop_assert!(longer.expected_tries() > ci.expected_tries());
                Ok(())
            },
        );
    }
}
