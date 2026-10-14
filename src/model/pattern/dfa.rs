//! 正規表現を、公開鍵の base64 の文字（6 ビットの値 0〜63）で進む DFA にする
//!
//! regex-automata の DFA から、base64 の 64 文字で行ける状態だけを取り出し、番号を詰め直す。
//! 一致した状態はまとめて [`ACCEPT`]（どの文字でもそこに留まる）にするので、判定は
//! 「最後までに ACCEPT に着いたか、最後の文字の後（文字列の終わり）で一致するか」だけになる。
//! 同じ表を CPU の判定・GPU の判定（kernel.cu の `pk_dfa`）・当たる確率の計算に使う

use crate::model::key::{PUBLIC_B64_LEN, PublicB64};
use crate::model::pattern::{PatternError, SEARCH_START};
use regex_automata::dfa::{Automaton as _, StartKind, dense};
use regex_automata::util::{start, syntax};
use regex_automata::{Anchored, nfa::thompson};

/// 一致した状態（どの文字でもここに留まる）
pub(crate) const ACCEPT: u16 = 0;
/// 判定の範囲の最初の文字を読む前の状態
pub(crate) const START: u16 = 1;

/// 状態の数の上限（GPU の表は 状態 × 65 × 2 バイトで、上限なら約 2.1 MB）
pub const MAX_STATES: usize = 1 << 14;

/// regex-automata が DFA を作るときに使ってよいメモリ（超えたら [`PatternError::TooComplex`]）
const BUILD_LIMIT: usize = 16 << 20;

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// base64 の文字の 6 ビットの値（文字表に無い文字は 64）
const VALUE: [u8; 256] = {
    let mut v = [64u8; 256];
    let mut i = 0;
    while i < 64 {
        v[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    v
};

/// 判定の範囲（固定部分の直後から最後まで）の文字数
const LEN: usize = PUBLIC_B64_LEN - SEARCH_START;

/// base64 の文字で進む DFA
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dfa {
    /// 状態 s で値 v の文字を読んだ後の状態（`trans[s * 64 + v]`）
    trans: Box<[u16]>,
    /// 状態 s で文字列が終わると一致するか
    eoi: Box<[bool]>,
}

impl Dfa {
    /// 正規表現 `regex` の DFA。大小文字を区別しないなら、英字はどちらでも一致する
    pub fn new(regex: &str, case_sensitive: bool) -> Result<Dfa, PatternError> {
        let dfa = dense::Builder::new()
            .syntax(
                syntax::Config::new()
                    .unicode(false)
                    .utf8(false)
                    .case_insensitive(!case_sensitive),
            )
            .thompson(thompson::Config::new().utf8(false))
            .configure(
                dense::Config::new()
                    .start_kind(StartKind::Unanchored)
                    .dfa_size_limit(Some(BUILD_LIMIT))
                    .determinize_size_limit(Some(BUILD_LIMIT)),
            )
            .build(regex)
            .map_err(|e| match e.is_size_limit_exceeded() {
                true => PatternError::TooComplex { max: MAX_STATES },
                false => PatternError::Regex(e.to_string()),
            })?;
        let first = dfa
            .start_state(&start::Config::new().anchored(Anchored::No))
            .map_err(|e| PatternError::Regex(e.to_string()))?;

        // 元の状態 → 詰めた番号（0 は ACCEPT なので、ほかは 1 から。最初は START）
        let mut ids = std::collections::HashMap::from([(first, START)]);
        let mut order = vec![first];
        let mut trans = vec![ACCEPT; 64]; // ACCEPT の行（どの文字でも ACCEPT）
        let mut eoi = vec![true];
        let mut i = 0;
        while i < order.len() {
            let s = order[i];
            i += 1;
            for &c in ALPHABET {
                let next = dfa.next_state(s, c);
                if dfa.is_quit_state(next) {
                    return Err(PatternError::Regex(format!("quit state at byte {c}")));
                }
                let id = match dfa.is_match_state(next) {
                    true => ACCEPT,
                    false => *ids.entry(next).or_insert_with(|| {
                        order.push(next);
                        (order.len()) as u16
                    }),
                };
                if order.len() >= MAX_STATES {
                    return Err(PatternError::TooComplex { max: MAX_STATES });
                }
                trans.push(id);
            }
            eoi.push(dfa.is_match_state(dfa.next_eoi_state(s)));
        }
        Ok(Dfa {
            trans: trans.into(),
            eoi: eoi.into(),
        })
    }

    /// 状態の数（[`ACCEPT`] を含む）
    pub fn states(&self) -> usize {
        self.eoi.len()
    }

    #[inline(always)]
    fn next(&self, s: u16, v: u8) -> u16 {
        self.trans[usize::from(s) * 64 + usize::from(v)]
    }

    /// 6 ビットの値の並び `values`（判定の範囲の全部）が一致するか
    #[inline]
    pub fn is_match_values(&self, values: impl IntoIterator<Item = u8>) -> bool {
        let mut s = START;
        for v in values {
            s = self.next(s, v);
            if s == ACCEPT {
                return true;
            }
        }
        self.eoi[usize::from(s)]
    }

    /// 公開鍵の base64 の判定の範囲（固定部分の後ろ）が一致するか
    #[inline]
    pub fn is_match(&self, b64: &PublicB64) -> bool {
        self.is_match_values(
            b64[SEARCH_START..]
                .iter()
                .map(|&c| VALUE[usize::from(c)] & 63),
        )
    }

    /// 公開鍵の base64 `b64` の `pos` 文字目を、`alt` に置き換えても一致するか、置き換えなくても一致するか
    /// （x の符号で変わる文字の両方の場合をまとめて確かめる。途中までは1回だけ進める）
    #[inline]
    pub fn is_match_either(&self, b64: &PublicB64, pos: usize, alt: u8) -> bool {
        let value = |c: u8| VALUE[usize::from(c)] & 63;
        let mut s = START;
        for &c in &b64[SEARCH_START..pos] {
            s = self.next(s, value(c));
        }
        [b64[pos], alt].into_iter().any(|c| {
            let rest = b64[pos + 1..].iter().map(|&c| value(c));
            let mut t = self.next(s, value(c));
            for v in rest {
                t = self.next(t, v);
            }
            t == ACCEPT || self.eoi[usize::from(t)]
        })
    }

    /// 1つの鍵で一致する確率。判定の範囲の最初の文字は A〜P の 16 通り、ほかの文字は 64 通りが同じ確率で出るとして、
    /// 状態ごとの確率を1文字ずつ進めて求める
    pub fn probability(&self) -> f64 {
        let n = self.states();
        let mut p = vec![0f64; n];
        p[usize::from(START)] = 1.0;
        let mut q = vec![0f64; n];
        for i in 0..LEN {
            let choices: u8 = if i == 0 { 16 } else { 64 };
            let w = 1.0 / f64::from(choices);
            q.fill(0.0);
            for (s, &mass) in p.iter().enumerate() {
                if mass == 0.0 {
                    continue;
                }
                for v in 0..choices {
                    q[usize::from(self.next(s as u16, v))] += mass * w;
                }
            }
            std::mem::swap(&mut p, &mut q);
        }
        let ends: f64 = (1..n).filter(|&s| self.eoi[s]).map(|s| p[s]).sum();
        (p[usize::from(ACCEPT)] + ends).min(1.0)
    }

    /// GPU に送る表: 状態ごとに 65 個の u16（リトルエンディアン）。0〜63 は値ごとの次の状態、64 は文字列の
    /// 終わりで一致するか（1・0）
    pub fn gpu_table(&self) -> Vec<u8> {
        (0..self.states())
            .flat_map(|s| {
                let row = &self.trans[s * 64..(s + 1) * 64];
                row.iter()
                    .copied()
                    .chain([u16::from(self.eoi[s])])
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::pattern::tests::SAMPLE;
    use crate::property::check;
    use proptest::prelude::*;
    use rstest::rstest;
    use similar_asserts::assert_eq;

    fn dfa(re: &str, cs: bool) -> Dfa {
        Dfa::new(re, cs).unwrap()
    }

    /// 同じ設定の regex クレートの判定（判定の範囲だけを渡す）
    fn regex(re: &str, cs: bool) -> regex::bytes::Regex {
        regex::bytes::RegexBuilder::new(re)
            .unicode(false)
            .case_insensitive(!cs)
            .build()
            .unwrap()
    }

    #[rstest]
    #[case::literal("WaKo", true, true)]
    #[case::literal_miss("wako", true, false)]
    #[case::insensitive("wako", false, true)]
    #[case::anchored_start("^IQK2", true, true)]
    #[case::anchored_start_miss("^QK2", true, false)]
    #[case::anchored_end("Gy9$", true, true)]
    #[case::anchored_end_miss("Gy$", true, false)]
    #[case::class("[0-9]{2}[a-z]", true, false)]
    #[case::alternation("zzzz|ofX", true, true)]
    #[case::fixed_part_is_not_searched("NTE5", true, false)]
    fn sample(#[case] re: &str, #[case] cs: bool, #[case] want: bool) {
        assert_eq!(dfa(re, cs).is_match(SAMPLE), want);
        assert_eq!(regex(re, cs).is_match(&SAMPLE[SEARCH_START..]), want);
    }

    /// 正しくない正規表現と、状態が多すぎる正規表現はエラー
    #[rstest]
    fn errors() {
        assert_eq!(
            matches!(Dfa::new("(", true), Err(PatternError::Regex(_))),
            true
        );
        assert_eq!(
            Dfa::new("a[A-Za-z0-9+/]{20}", true),
            Err(PatternError::TooComplex { max: MAX_STATES })
        );
    }

    /// regex クレートと同じ判定になる（文字列は base64 の 43 文字。符号の文字を置き換えた判定も同じ）
    #[rstest]
    #[case::literal("ab+c")]
    #[case::classes("[0-9][A-F]x")]
    #[case::anchors("^[A-P]a|z$")]
    #[case::repeat("(ab){2,}")]
    #[case::boundary("\\b9")]
    #[case::any("a.b")]
    fn matches_regex_crate(#[case] re: &str, #[values(false, true)] cs: bool) {
        let (d, r) = (dfa(re, cs), regex(re, cs));
        check(
            1024,
            ("[A-P]", "[A-Za-z0-9+/]{42}", any::<u8>()),
            |(first, rest, alt)| {
                let s = format!("{}{first}{rest}", crate::model::key::FIXED_PREFIX);
                let b64: PublicB64 = s.as_bytes().try_into().unwrap();
                prop_assert_eq!(d.is_match(&b64), r.is_match(&b64[SEARCH_START..]));
                let alt = ALPHABET[usize::from(alt & 63)];
                let mut other = b64;
                other[66] = alt;
                prop_assert_eq!(
                    d.is_match_either(&b64, 66, alt),
                    d.is_match(&b64) || d.is_match(&other)
                );
                Ok(())
            },
        );
    }

    /// 確率: 1 か所の文字（2 文字目から）なら 1 - (63/64)^42、末尾に固定なら 1/64、先頭に A なら 1/16
    #[rstest]
    #[case::one_char("z", 1.0 - (63f64 / 64.0).powi(42))]
    #[case::at_end("z$", 1.0 / 64.0)]
    #[case::at_start("^A", 1.0 / 16.0)]
    #[case::at_start_impossible("^Q", 0.0)]
    #[case::two_at_end("ab$", 1.0 / 4096.0)]
    #[case::insensitive_two_at_end("(?i)ab$", 4.0 / 4096.0)]
    fn probability(#[case] re: &str, #[case] want: f64) {
        let got = dfa(re, true).probability();
        assert_eq!((got - want).abs() < 1e-12, true, "{got}（期待 {want}）");
    }

    /// GPU の表は 状態 × 65 × 2 バイトで、ACCEPT の行は全部 ACCEPT、終わりの印は 1
    #[rstest]
    fn gpu_table_layout() {
        let d = dfa("ab", true);
        let t = d.gpu_table();
        assert_eq!(t.len(), d.states() * 65 * 2);
        let word =
            |s: usize, i: usize| u16::from_le_bytes([t[(s * 65 + i) * 2], t[(s * 65 + i) * 2 + 1]]);
        assert_eq!((0..64).all(|i| word(0, i) == ACCEPT), true);
        assert_eq!(word(0, 64), 1);
        assert_eq!(word(1, 64), 0);
    }
}
