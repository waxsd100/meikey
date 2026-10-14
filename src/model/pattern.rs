//! 探す文字列と、公開鍵に含まれるかの判定（探索ループで速く判定するものは [`crate::service::matcher`]）
//!
//! 探し方は [`Mode`] で選ぶ: どこかに含む・末尾・正規表現。正規表現は [`dfa::Dfa`] にもして、
//! 探索ループの判定と当たる確率の計算に使う

pub mod dfa;

use crate::lang::{t, tf};
use crate::model::key::{FIXED_PREFIX, PUBLIC_B64_LEN, PublicB64};
use dfa::Dfa;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

/// 判定に使う範囲の先頭（固定部分の直後）
pub(crate) const SEARCH_START: usize = FIXED_PREFIX.len();

/// 固定部分の直後の1文字は A〜P しか出ないので、自由に変わるのはその次から
const FREE_LEN: usize = PUBLIC_B64_LEN - SEARCH_START - 1;

/// 探す文字列が使えない理由
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PatternError {
    #[error("{}", t!("文字列が空", "the text is empty"))]
    Empty,
    #[error("{}", tf!(
        "使えない文字: '{}'（使える文字: A-Z a-z 0-9 + /）",
        "invalid character: '{}' (allowed: A-Z a-z 0-9 + /)",
        .0
    ))]
    InvalidChar(char),
    #[error("{}", tf!(
        "文字列が長すぎる: {len} 文字（最大 {max} 文字）",
        "the text is too long: {len} characters (at most {max})"
    ))]
    TooLong { len: usize, max: usize },
    #[error("{}", tf!("正規表現が正しくない: {}", "invalid regular expression: {}", .0))]
    Regex(String),
    #[error("{}", tf!(
        "正規表現が複雑すぎる（DFA の状態が {max} 個を超える）",
        "the regular expression is too complex (the DFA has more than {max} states)"
    ))]
    TooComplex { max: usize },
    #[error("{}", t!(
        "空の文字列に一致する正規表現は使えない（どの鍵にも一致する）",
        "a regular expression that matches the empty string cannot be used (it matches every key)"
    ))]
    MatchesEmpty,
    #[error("{}", t!(
        "この正規表現に一致する公開鍵は無い（使える文字は A-Z a-z 0-9 + /、固定部分の直後の 1 文字は A〜P）",
        "no public key matches this regular expression (the characters are A-Z a-z 0-9 + /, and the one right after the fixed part is A-P)"
    ))]
    NeverMatches,
}

/// 探し方
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// 固定部分の後ろのどこかに含む
    #[default]
    Contains,
    /// 公開鍵の最後が一致する
    Suffix,
    /// 固定部分の後ろ（43 文字）が正規表現に一致する（`^` は固定部分の直後、`$` は最後）
    Regex,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Contains => t!("含む", "contains"),
            Mode::Suffix => t!("末尾", "at the end"),
            Mode::Regex => t!("正規表現", "regular expression"),
        }
    }
}

/// 正規表現をコンパイルしたもの（一致した位置を出す regex と、判定と確率に使う DFA）
#[derive(Debug)]
struct Compiled {
    regex: regex::bytes::Regex,
    dfa: Dfa,
    probability: f64,
}

/// 公開鍵に含めたい文字列
#[derive(Debug, Clone)]
pub struct Pattern {
    needle: Box<[u8]>,
    case_sensitive: bool,
    mode: Mode,
    /// 正規表現のとき
    compiled: Option<Arc<Compiled>>,
}

impl PartialEq for Pattern {
    fn eq(&self, other: &Self) -> bool {
        (&self.needle, self.case_sensitive, self.mode)
            == (&other.needle, other.case_sensitive, other.mode)
    }
}

impl Eq for Pattern {}

impl Pattern {
    /// 指定できる最大文字数（正規表現でないとき）
    pub const MAX_LEN: usize = FREE_LEN;

    /// どこかに含む文字列
    pub fn new(s: &str, case_sensitive: bool) -> Result<Self, PatternError> {
        Self::with_mode(s, case_sensitive, Mode::Contains)
    }

    /// 探し方 `mode` の文字列（正規表現なら、その正規表現）
    pub fn with_mode(s: &str, case_sensitive: bool, mode: Mode) -> Result<Self, PatternError> {
        (!s.is_empty()).ok_or(PatternError::Empty)?;
        if mode == Mode::Regex {
            return Self::regex(s, case_sensitive);
        }
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
            mode,
            compiled: None,
        })
    }

    fn regex(s: &str, case_sensitive: bool) -> Result<Self, PatternError> {
        let hir = regex_syntax::ParserBuilder::new()
            .unicode(false)
            .utf8(false)
            .case_insensitive(!case_sensitive)
            .build()
            .parse(s)
            .map_err(|e| PatternError::Regex(e.to_string()))?;
        if hir.properties().minimum_len() == Some(0) {
            return Err(PatternError::MatchesEmpty);
        }
        let regex = regex::bytes::RegexBuilder::new(s)
            .unicode(false)
            .case_insensitive(!case_sensitive)
            .size_limit(64 << 20)
            .build()
            .map_err(|e| PatternError::Regex(e.to_string()))?;
        let dfa = Dfa::new(s, case_sensitive)?;
        let probability = dfa.probability();
        if probability == 0.0 {
            return Err(PatternError::NeverMatches);
        }
        Ok(Self {
            needle: s.as_bytes().into(),
            case_sensitive,
            mode: Mode::Regex,
            compiled: Some(Arc::new(Compiled {
                regex,
                dfa,
                probability,
            })),
        })
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// 正規表現の DFA（正規表現のときだけ）
    pub fn dfa(&self) -> Option<&Dfa> {
        self.compiled.as_ref().map(|c| &c.dfa)
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

    /// 公開鍵の base64 の中で一致した位置（正規表現なら、いちばん左で一致したもの）
    pub fn find(&self, b64: &PublicB64) -> Option<Range<usize>> {
        let hay = &b64[SEARCH_START..];
        match (self.mode, &self.compiled) {
            (Mode::Regex, Some(c)) => {
                let m = c.regex.find(hay)?;
                Some(SEARCH_START + m.start()..SEARCH_START + m.end())
            }
            (Mode::Suffix, _) => {
                let start = PUBLIC_B64_LEN - self.needle.len();
                self.same(&b64[start..]).then_some(start..PUBLIC_B64_LEN)
            }
            _ => {
                let hit = hay.windows(self.needle.len()).find(|w| self.same(w))?;
                b64.subslice_range(hit).map(Into::into)
            }
        }
    }

    pub fn is_match(&self, b64: &PublicB64) -> bool {
        self.find(b64).is_some()
    }

    fn same(&self, w: &[u8]) -> bool {
        if self.case_sensitive {
            w == &self.needle[..]
        } else {
            w.eq_ignore_ascii_case(&self.needle[..])
        }
    }

    /// 1回の試行で当たる確率（どこかに含むときは、正確には1つの鍵に含まれる個数の期待値。
    /// 当たりにくい文字列ではほぼ同じ）
    pub fn probability(&self) -> f64 {
        match (self.mode, &self.compiled) {
            (Mode::Regex, Some(c)) => return c.probability,
            // 最後の n 文字（42 文字までなので、固定部分の直後の A〜P の文字は含まない）
            (Mode::Suffix, _) => return self.needle.iter().map(|&c| self.per_char(c)).product(),
            _ => {}
        }
        let per_char = |c: u8| self.per_char(c);
        let (first, rest) = self.needle.split_first().expect("空でない");
        let rest: f64 = rest.iter().map(|&c| per_char(c)).product();
        // 固定部分の直後の次（26 文字目）から始まる位置
        let positions = (FREE_LEN + 1 - self.needle.len()) as f64;
        let free = positions * per_char(*first) * rest;
        // 固定部分の直後（25 文字目）は A〜P の 16 通りが同じ確率で出るので、先頭がそのどれかなら 1/16
        let head = match self.case_sensitive {
            true => *first,
            false => first.to_ascii_uppercase(),
        };
        let at_start = if (b'A'..=b'P').contains(&head) {
            rest / 16.0
        } else {
            0.0
        };
        free + at_start
    }

    /// 自由に変わる位置の1文字が一致する確率。大小文字を区別しないなら、英字は当たりが2通り
    fn per_char(&self, c: u8) -> f64 {
        if !self.case_sensitive && c.is_ascii_alphabetic() {
            2.0 / 64.0
        } else {
            1.0 / 64.0
        }
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
            Find::Naive => t!("総当たり", "naive"),
        }
    }
}

impl fmt::Display for Find {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", t!("検索", "search"), self.name())
    }
}

fn is_base64_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '+' || c == '/'
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::collection::vec;
    use rstest::rstest;
    use similar_asserts::assert_eq;
    use std::str::from_utf8;

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

    /// 末尾・正規表現で一致した位置（正規表現は、いちばん左で一致したもの）
    #[rstest]
    #[case::suffix(Mode::Suffix, "Gy9", true, Some(65..68))]
    #[case::suffix_insensitive(Mode::Suffix, "gY9", false, Some(65..68))]
    #[case::suffix_not_at_end(Mode::Suffix, "WaKo", true, None)]
    #[case::regex_start(Mode::Regex, "^IQK2", true, Some(25..29))]
    #[case::regex_start_miss(Mode::Regex, "^AQK2", true, None)]
    #[case::regex_any(Mode::Regex, "Wa.o", true, Some(56..60))]
    #[case::regex_insensitive(Mode::Regex, "wa.o", false, Some(56..60))]
    #[case::regex_end(Mode::Regex, "[0-9]$", true, Some(67..68))]
    #[case::regex_leftmost(Mode::Regex, "[0-9]", true, Some(28..29))]
    #[case::regex_fixed_part(Mode::Regex, "NzaC1l", true, None)]
    fn find_with_mode(
        #[case] mode: Mode,
        #[case] s: &str,
        #[case] cs: bool,
        #[case] want: Option<Range<usize>>,
    ) {
        let p = Pattern::with_mode(s, cs, mode).unwrap();
        assert_eq!(p.find(SAMPLE), want);
        assert_eq!(p.is_match(SAMPLE), want.is_some());
        assert_eq!(
            (p.mode(), p.as_str(), p.dfa().is_some()),
            (mode, s, mode == Mode::Regex)
        );
    }

    /// 正規表現で使えないもの（正しくない・空の文字列に一致する・一致する鍵が無い）。末尾は文字の検査が同じ
    #[rstest]
    #[case::syntax(
        Mode::Regex,
        "a(",
        "正規表現が正しくない: regex parse error:\n    a(\n     ^\nerror: unclosed group"
    )]
    #[case::empty_match(
        Mode::Regex,
        "a*|b",
        "空の文字列に一致する正規表現は使えない（どの鍵にも一致する）"
    )]
    #[case::anchor_only(
        Mode::Regex,
        "^",
        "空の文字列に一致する正規表現は使えない（どの鍵にも一致する）"
    )]
    #[case::not_base64(
        Mode::Regex,
        "a=b",
        "この正規表現に一致する公開鍵は無い（使える文字は A-Z a-z 0-9 + /、固定部分の直後の 1 文字は A〜P）"
    )]
    #[case::q_right_after_prefix(
        Mode::Regex,
        "^Q",
        "この正規表現に一致する公開鍵は無い（使える文字は A-Z a-z 0-9 + /、固定部分の直後の 1 文字は A〜P）"
    )]
    #[case::too_long_for_key(
        Mode::Regex,
        "[a-z]{44}",
        "この正規表現に一致する公開鍵は無い（使える文字は A-Z a-z 0-9 + /、固定部分の直後の 1 文字は A〜P）"
    )]
    #[case::too_complex(
        Mode::Regex,
        "a[A-Za-z0-9+/]{20}",
        "正規表現が複雑すぎる（DFA の状態が 16384 個を超える）"
    )]
    #[case::empty(Mode::Regex, "", MSG_EMPTY)]
    #[case::suffix_invalid(
        Mode::Suffix,
        "a-b",
        "使えない文字: '-'（使える文字: A-Z a-z 0-9 + /）"
    )]
    #[case::suffix_too_long(Mode::Suffix, &"a".repeat(43), "文字列が長すぎる: 43 文字（最大 42 文字）")]
    fn invalid_with_mode(#[case] mode: Mode, #[case] s: &str, #[case] message: &str) {
        assert_eq!(
            Pattern::with_mode(s, true, mode).unwrap_err().to_string(),
            message
        );
    }

    /// 同じ文字列でも、探し方が違えば別の Pattern
    #[rstest]
    fn equality_includes_mode() {
        let p = |mode| Pattern::with_mode("ab", true, mode).unwrap();
        assert_eq!(p(Mode::Contains) == p(Mode::Contains), true);
        assert_eq!(p(Mode::Contains) == p(Mode::Suffix), false);
        assert_eq!(p(Mode::Suffix) == p(Mode::Regex), false);
        assert_eq!(p(Mode::Regex) == p(Mode::Regex), true);
    }

    /// 末尾の期待試行回数は、1 か所だけなので 64^n（大小文字を区別しないなら英字1文字ごとに半分）。
    /// 正規表現の文字列そのものなら、位置の数が同じなのでどこかに含むときとほぼ同じ
    #[rstest]
    #[case::suffix_cs(Mode::Suffix, "ab", true, 4096.0)]
    #[case::suffix_ci(Mode::Suffix, "ab", false, 1024.0)]
    #[case::suffix_digits(Mode::Suffix, "12", false, 4096.0)]
    #[case::regex_suffix(Mode::Regex, "ab$", true, 4096.0)]
    #[case::regex_two_choices(Mode::Regex, "[ab]$", true, 32.0)]
    fn expected_tries_with_mode(
        #[case] mode: Mode,
        #[case] s: &str,
        #[case] cs: bool,
        #[case] want: f64,
    ) {
        let got = Pattern::with_mode(s, cs, mode).unwrap().expected_tries();
        assert_eq!(
            (got / want - 1.0).abs() < 1e-9,
            true,
            "{got}（期待 {want}）"
        );
    }

    /// 長い文字列は、どこかに含む（期待値）と正規表現（確率）でほぼ同じ
    #[rstest]
    fn regex_probability_close_to_contains() {
        check(256, ("[A-Za-z0-9]{4,6}", any::<bool>()), |(needle, cs)| {
            let contains = pat(&needle, cs).probability();
            let regex = Pattern::with_mode(&needle, cs, Mode::Regex)
                .unwrap()
                .probability();
            prop_assert!(
                (regex / contains - 1.0).abs() < 1e-3,
                "{} {}",
                regex,
                contains
            );
            Ok(())
        });
    }

    /// --bench の表示。既定は memmem
    #[rstest]
    #[case::memmem(Find::Memmem, "検索=memmem")]
    #[case::naive(Find::Naive, "検索=総当たり")]
    fn find_display(#[case] how: Find, #[case] want: &str) {
        assert_eq!(how.to_string(), want);
        assert_eq!(Find::default(), Find::Memmem);
    }

    /// 期待試行回数（大小文字の区別なし・あり）。26 文字目からの 43 - n か所に、先頭が A〜P（区別しないなら
    /// a〜p も）なら 25 文字目（A〜P の 16 通り）からの 1 か所を足す
    #[rstest]
    // 7文字なら 26 文字目からの位置は 42-7+1 = 36 か所
    #[case::seven_letters("abcdefg", 1.0 / (36.0 * (2.0f64 / 64.0).powi(7) + (2.0f64 / 64.0).powi(6) / 16.0), 64f64.powi(7) / 36.0)]
    #[case::letters_and_digits("gack951", 1.0 / (36.0 * 16.0 / 64f64.powi(7) + 8.0 / 64f64.powi(6) / 16.0), 64f64.powi(7) / 36.0)]
    #[case::eight_letters("wakokara", 64f64.powi(8) / (2f64.powi(8) * 35.0), 64f64.powi(8) / 35.0)]
    #[case::one_letter_after_p("W", 64.0 / (2.0 * 42.0), 64.0 / 42.0)]
    #[case::one_letter_a("A", 1.0 / (2.0 * 42.0 / 64.0 + 1.0 / 16.0), 1.0 / (42.0 / 64.0 + 1.0 / 16.0))]
    #[case::lowercase_p("pZ", 1.0 / (41.0 * 4.0 / 4096.0 + 2.0 / 64.0 / 16.0), 4096.0 / 41.0)]
    #[case::uppercase_p("Pz", 1.0 / (41.0 * 4.0 / 4096.0 + 2.0 / 64.0 / 16.0), 1.0 / (41.0 / 4096.0 + 1.0 / 64.0 / 16.0))]
    #[case::q_is_not_at_start("Qz", 4096.0 / (41.0 * 4.0), 4096.0 / 41.0)]
    #[case::digits_only("2026", 64f64.powi(4) / 39.0, 64f64.powi(4) / 39.0)]
    #[case::symbols_only("+/+/", 64f64.powi(4) / 39.0, 64f64.powi(4) / 39.0)]
    fn expected_tries(#[case] s: &str, #[case] insensitive: f64, #[case] sensitive: f64) {
        for (got, want) in [
            (pat(s, false).expected_tries(), insensitive),
            (pat(s, true).expected_tries(), sensitive),
        ] {
            assert_eq!(
                (got / want - 1.0).abs() < 1e-12,
                true,
                "{s}: {got}（期待 {want}）"
            );
        }
    }

    /// 実際の鍵の公開鍵で、1文字が含まれる個数の平均は [`Pattern::probability`] と同じ
    /// （25 文字目が A〜P だけなので、A は Z より多く含まれる）。4000 鍵の平均なので ±0.05 で比べる
    #[rstest]
    #[case::a("A")]
    #[case::z("Z")]
    #[case::digit("7")]
    fn probability_matches_real_keys(#[case] s: &str) {
        use crate::model::key::VanityKey;
        let p = pat(s, true);
        let n = 4000;
        let total: usize = (0..n as u32)
            .map(|i| {
                let mut seed = [0u8; 32];
                seed[..4].copy_from_slice(&i.to_le_bytes());
                let b64 = VanityKey::from_seed(seed).public_b64();
                b64[SEARCH_START..]
                    .iter()
                    .filter(|&&c| c == s.as_bytes()[0])
                    .count()
            })
            .sum();
        let mean = total as f64 / f64::from(n);
        assert_eq!(
            (mean - p.probability()).abs() < 0.05,
            true,
            "{mean}（期待 {}）",
            p.probability()
        );
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
            let s = from_utf8(&b64).unwrap();
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
            vec(any::<bool>(), 6),
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
