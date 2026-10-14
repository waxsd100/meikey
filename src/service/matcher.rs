//! 探索ループで、公開鍵の base64 に文字列が含まれるかを速く判定する
//!
//! 結果は [`Pattern::find`] と同じ（最初に一致した位置）。どこかに含む文字列を探す方法は [`Find`] で選ぶ。
//! 末尾は最後の文字を比べるだけ、正規表現は DFA（[`Dfa`]）で判定する

use crate::lang::t;
use crate::model::key::{PUBLIC_B64_LEN, PublicB64};
use crate::model::pattern::dfa::Dfa;
use crate::model::pattern::{Find, Mode, Pattern, SEARCH_START};
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
    /// [`Pattern::find`] で判定する（総当たり・末尾）
    Naive(Pattern),
    /// 正規表現の DFA
    Dfa(Pattern),
    Memmem {
        finder: memmem::Finder<'static>,
        /// 公開鍵を小文字にそろえてから探す（大小文字を区別しないとき）
        fold: bool,
    },
}

impl Matcher {
    pub fn new(pattern: &Pattern, find: Find) -> Matcher {
        Matcher(match (pattern.mode(), find) {
            (Mode::Regex, _) => Inner::Dfa(pattern.clone()),
            (Mode::Suffix, _) | (_, Find::Naive) => Inner::Naive(pattern.clone()),
            (Mode::Contains, Find::Memmem) => {
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

    /// 判定のしかたの名前（状態の表示用）
    pub fn name(pattern: Option<&Pattern>, find: Find) -> &'static str {
        match pattern.map(Pattern::mode) {
            Some(Mode::Suffix) => t!("末尾を比べる", "compare the end"),
            Some(Mode::Regex) => "DFA",
            _ => find.name(),
        }
    }

    /// 公開鍵の base64 の中で一致した位置
    #[inline]
    pub fn find(&self, b64: &PublicB64) -> Option<Range<usize>> {
        match &self.0 {
            Inner::Naive(p) | Inner::Dfa(p) => p.find(b64),
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
        match &self.0 {
            Inner::Dfa(p) => dfa(p).is_match(b64),
            _ => self.find(b64).is_some(),
        }
    }

    /// x の符号を 0 にした公開鍵の base64 `b64` に、符号が 0 でも 1 でも文字列が含まれる可能性があるか。
    /// 含まれる鍵ならかならず true（false なら、符号によらず含まれない）。true のときは符号を計算して
    /// [`Matcher::is_match`] で確かめ直す。総当たり（[`Find::Naive`]）は高速化前の判定なので、いつも true
    /// （符号を毎回計算する）
    #[inline]
    pub fn may_match_unsigned(&self, b64: &PublicB64) -> bool {
        let (finder, fold) = match &self.0 {
            Inner::Memmem { finder, fold } => (finder, fold),
            Inner::Dfa(p) => {
                return dfa(p).is_match_either(b64, SIGN_CHAR, with_sign(b64[SIGN_CHAR]));
            }
            Inner::Naive(p) if p.mode() == Mode::Suffix => {
                let mut signed = *b64;
                signed[SIGN_CHAR] = with_sign(b64[SIGN_CHAR]);
                return p.is_match(b64) || p.is_match(&signed);
            }
            Inner::Naive(_) => return true,
        };
        if self.is_match(b64) {
            return true;
        }
        // 符号が 1 なら変わるのは SIGN_CHAR の1文字だけなので、その文字を含む範囲だけ探し直す
        let len = finder.needle().len();
        let (lo, hi) = (
            (SIGN_CHAR + 1).saturating_sub(len).max(SEARCH_START),
            (SIGN_CHAR + len).min(PUBLIC_B64_LEN),
        );
        let mut window = [0u8; PUBLIC_B64_LEN];
        let window = &mut window[..hi - lo];
        window.copy_from_slice(&b64[lo..hi]);
        window[SIGN_CHAR - lo] = with_sign(b64[SIGN_CHAR]);
        if *fold {
            window.iter_mut().for_each(|c| *c = fold_case(*c));
        }
        finder.find(window).is_some()
    }
}

fn dfa(p: &Pattern) -> &Dfa {
    p.dfa().expect("正規表現の Pattern")
}

/// 公開鍵の base64 で、x の符号で変わる文字の位置（blob の最後のバイトの最上位ビットは、この文字の値のビット 1）
pub const SIGN_CHAR: usize = PUBLIC_B64_LEN - 2;

/// 符号が 0 のときの SIGN_CHAR の文字 `c` を、符号が 1 のときの文字にする（値のビット 1 を立てる）
fn with_sign(c: u8) -> u8 {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let v = ALPHABET.iter().position(|&a| a == c).unwrap_or(0);
    ALPHABET[v | 2]
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
    use crate::model::pattern::Mode;
    use crate::model::pattern::tests::{ALPHABET, PREFIX_ONLY, SAMPLE, b64_with};
    use crate::property::check;
    use proptest::prelude::*;
    use rstest::rstest;
    use similar_asserts::assert_eq;

    fn matcher(s: &str, cs: bool, how: Find) -> Matcher {
        Matcher::new(&Pattern::new(s, cs).unwrap(), how)
    }

    /// 文字列 `needle` を探し方 `mode` で探す Pattern（正規表現なら、文字列そのものに一致するよう逃がす）
    fn pattern(needle: &str, cs: bool, mode: Mode) -> Pattern {
        let s = match mode {
            Mode::Regex => regex_syntax::escape(needle),
            _ => needle.to_string(),
        };
        Pattern::with_mode(&s, cs, mode).unwrap()
    }

    /// 末尾・正規表現でも、判定は Pattern と同じ（--no-memmem でも同じ判定器）
    #[rstest]
    #[case::suffix(Mode::Suffix, "Gy9", true)]
    #[case::suffix_miss(Mode::Suffix, "WaKo", false)]
    #[case::regex(Mode::Regex, "W[a-z]Ko", true)]
    #[case::regex_end(Mode::Regex, "9$", true)]
    #[case::regex_miss(Mode::Regex, "^A", false)]
    fn modes(
        #[case] mode: Mode,
        #[case] s: &str,
        #[case] want: bool,
        #[values(Find::Memmem, Find::Naive)] how: Find,
    ) {
        let p = Pattern::with_mode(s, true, mode).unwrap();
        let m = Matcher::new(&p, how);
        assert_eq!((m.is_match(SAMPLE), m.find(SAMPLE)), (want, p.find(SAMPLE)));
    }

    /// 状態の表示の名前: 末尾・正規表現は探し方で決まり、どこかに含むなら検索の方法
    #[rstest]
    fn names() {
        let p = |mode| Pattern::with_mode("ab", true, mode).unwrap();
        assert_eq!(Matcher::name(None, Find::Naive), "総当たり");
        assert_eq!(
            Matcher::name(Some(&p(Mode::Contains)), Find::Memmem),
            "memmem"
        );
        assert_eq!(
            Matcher::name(Some(&p(Mode::Suffix)), Find::Memmem),
            "末尾を比べる"
        );
        assert_eq!(Matcher::name(Some(&p(Mode::Regex)), Find::Naive), "DFA");
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
        #[values(Mode::Contains, Mode::Suffix, Mode::Regex)] mode: Mode,
    ) {
        let strategy = ("[A-Za-z0-9+/]{43}", needles, any::<usize>());
        check(256, strategy, |(rest, needle, pos)| {
            // 末尾なら最後に埋め込む
            let pos = match mode {
                Mode::Suffix => 43 - needle.len(),
                _ => pos,
            };
            let b64 = b64_with(&needle, planted.then_some(pos), &rest);
            let p = pattern(&needle, cs, mode);
            let m = Matcher::new(&p, how);
            prop_assert_eq!(m.find(&b64), p.find(&b64));
            prop_assert_eq!(m.is_match(&b64), p.is_match(&b64));
            if planted {
                prop_assert!(m.is_match(&b64), "埋め込んだのに見つからない");
            }
            Ok(())
        });
    }

    /// 符号のビットは base64 の SIGN_CHAR の文字の値のビット 1 だけを変える
    #[rstest]
    fn sign_bit_changes_only_sign_char() {
        check(256, any::<[u8; 32]>(), |mut pk| {
            pk[31] &= 0x7f;
            let mut unsigned = [0u8; PUBLIC_B64_LEN];
            crate::model::key::encode_public_b64(&pk, &mut unsigned);
            pk[31] |= 0x80;
            let mut signed = [0u8; PUBLIC_B64_LEN];
            crate::model::key::encode_public_b64(&pk, &mut signed);
            for i in 0..PUBLIC_B64_LEN {
                let want = if i == SIGN_CHAR {
                    with_sign(unsigned[i])
                } else {
                    unsigned[i]
                };
                prop_assert_eq!(signed[i], want, "{}", i);
            }
            Ok(())
        });
    }

    /// 符号を 0 にした base64 で may_match_unsigned が false なら、符号が 0 でも 1 でも含まれない。
    /// 符号が 0・1 のどちらかで含まれるなら true（文字列は SIGN_CHAR の近くに埋め込むことが多い）
    #[rstest]
    fn may_match_unsigned_never_misses(
        #[values("[A-Za-z0-9+/]{1,3}", "[A-Za-z0-9+/]{4,8}")] needles: &'static str,
        #[values(false, true)] cs: bool,
        #[values(Find::Memmem, Find::Naive)] how: Find,
        #[values(Mode::Contains, Mode::Suffix, Mode::Regex)] mode: Mode,
    ) {
        let strategy = (any::<[u8; 32]>(), needles, 0usize..12, any::<bool>());
        check(1024, strategy, |(mut pk, needle, back, plant)| {
            let p = pattern(&needle, cs, mode);
            let m = Matcher::new(&p, how);
            let encode = |pk: &[u8; 32]| {
                let mut b = [0u8; PUBLIC_B64_LEN];
                crate::model::key::encode_public_b64(pk, &mut b);
                b
            };
            if plant {
                // 符号が 1 の鍵の base64 の、SIGN_CHAR を含む位置に文字列を書いた鍵を作る（base64 から戻す）
                pk[31] |= 0x80;
                let mut b = encode(&pk);
                let start = (SIGN_CHAR + 1)
                    .saturating_sub(back.min(needle.len()))
                    .min(PUBLIC_B64_LEN - needle.len());
                b[start..start + needle.len()].copy_from_slice(needle.as_bytes());
                use base64::Engine;
                let blob = base64::engine::general_purpose::STANDARD.decode(b).unwrap();
                pk = *blob.last_chunk::<32>().unwrap();
            }
            let mut unsigned = pk;
            unsigned[31] &= 0x7f;
            let mut signed = unsigned;
            signed[31] |= 0x80;
            let maybe = m.may_match_unsigned(&encode(&unsigned));
            for b in [encode(&unsigned), encode(&signed)] {
                if p.is_match(&b) {
                    prop_assert!(maybe, "見逃した: {}", String::from_utf8_lossy(&b));
                }
            }
            Ok(())
        });
    }
}
