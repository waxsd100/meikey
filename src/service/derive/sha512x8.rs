//! AVX-512 で8つのシードの SHA-512 を同時に計算する（秘密スカラーを作るため専用）
//!
//! シードは 32 バイトなので、パディングを含めて SHA-512 のブロック1つに収まる。
//! 512 ビットレジスタの 64 ビットのレーン8つに8つのシードの同じワードを入れ、80 ラウンドを8つ同時に進める。
//! スカラーには SHA-512 の前半 32 バイトだけを使うので、出力も前半だけ計算する。
//!
//! AVX-512 の命令で、AVX2 版（[`crate::service::derive::sha512x4`]）では数命令かかっていた操作を1命令にしている:
//! - 64 ビットの回転: `vprorq`（AVX2 はシフト2回と OR）
//! - 3入力の論理演算（Σ の XOR 3つ・Ch・Maj）: `vpternlogq`（AVX2 は2〜4命令）
//! - 入力と出力の並べ替え: 8シードを 512 ビットずつ読み、`vpermt2q`（2レジスタからの任意の並べ替え）で
//!   「シードごと」と「ワードごと」を入れ替える。バイト順の反転は `vpshufb`

use crate::service::derive::avx512::Avx512;
use crate::service::derive::sha512x4::{IV, K};
use std::arch::x86_64::*;

/// 8つのシードの `SHA-512(seed)[..32]`
pub fn prefixes(avx512: Avx512, seeds: &[[u8; 32]; 8]) -> [[u8; 32]; 8] {
    let _ = avx512;
    // SAFETY: Avx512 があるので、CPU は下の target_feature の命令に対応している
    unsafe { prefixes_avx512(seeds) }
}

// 以下の unsafe fn は AVX-512（F・BW）の命令を呼ぶだけで、対応した CPU でだけ呼ぶこと。
// `#[target_feature]` の関数は `#[inline(always)]` にできないので、入口（prefixes_avx512）だけに付けて
// ほかは展開させる（[`crate::service::derive::avx512`] と同じ）

/// `vpternlogq` の真理値表（ビット `4a + 2b + c` が `f(a, b, c)`）
const XOR3: i32 = 0x96;
/// Ch(e, f, g) = e ? f : g
const CH: i32 = 0xca;
/// Maj(a, b, c) = 3つのうち2つ以上が 1
const MAJ: i32 = 0xe8;

#[inline(always)]
unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
    unsafe { _mm512_add_epi64(a, b) }
}

/// Σ: 3つの右回転の XOR
#[inline(always)]
unsafe fn big_sigma<const A: i32, const B: i32, const C: i32>(x: __m512i) -> __m512i {
    unsafe {
        let (a, b, c) = (
            _mm512_ror_epi64::<A>(x),
            _mm512_ror_epi64::<B>(x),
            _mm512_ror_epi64::<C>(x),
        );
        _mm512_ternarylogic_epi64::<XOR3>(a, b, c)
    }
}

/// σ: 2つの右回転と右シフトの XOR
#[inline(always)]
unsafe fn small_sigma<const A: i32, const B: i32, const C: u32>(x: __m512i) -> __m512i {
    unsafe {
        let (a, b, c) = (
            _mm512_ror_epi64::<A>(x),
            _mm512_ror_epi64::<B>(x),
            _mm512_srli_epi64::<C>(x),
        );
        _mm512_ternarylogic_epi64::<XOR3>(a, b, c)
    }
}

/// t 番目のラウンド（t は定数で呼ぶので、添字や分岐は展開時に消える）
#[inline(always)]
unsafe fn round(s: &mut [__m512i; 8], w: &mut [__m512i; 16], t: usize) {
    unsafe {
        if t >= 16 {
            // W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16]（16 個の輪で持つ）
            let s1 = small_sigma::<19, 61, 6>(w[(t - 2) % 16]);
            let s0 = small_sigma::<1, 8, 7>(w[(t - 15) % 16]);
            w[t % 16] = add(add(s1, w[(t - 7) % 16]), add(s0, w[t % 16]));
        }
        let [a, b, c, d, e, f, g, h] = *s;
        let ch = _mm512_ternarylogic_epi64::<CH>(e, f, g);
        let k = _mm512_set1_epi64(K[t].cast_signed());
        let t1 = add(
            add(h, big_sigma::<14, 18, 41>(e)),
            add(add(ch, k), w[t % 16]),
        );
        let maj = _mm512_ternarylogic_epi64::<MAJ>(a, b, c);
        let t2 = add(big_sigma::<28, 34, 39>(a), maj);
        *s = [add(t1, t2), a, b, c, add(d, t1), e, f, g];
    }
}

/// 80 ラウンドを書き下す（ラウンドの番号が定数になり、W の添字や K の値が展開時に決まる）
macro_rules! rounds {
    ($s:ident, $w:ident; $($t:literal)*) => {
        $( round(&mut $s, &mut $w, $t); )*
    };
}

/// 64 ビットのワードごとにバイト順を反転する（ビッグエンディアン ⇔ リトルエンディアン）
#[inline(always)]
unsafe fn bswap64(x: __m512i) -> __m512i {
    unsafe {
        // 128 ビットごとに同じ並べ替え: 各ワードのバイト 7..0
        let m = _mm512_broadcast_i32x4(_mm_set_epi8(
            8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7,
        ));
        _mm512_shuffle_epi8(x, m)
    }
}

/// `vpermt2q` の添字（0〜7 が1つ目のレジスタ、8〜15 が2つ目のレジスタのワード）
#[inline(always)]
unsafe fn idx(i: [i64; 8]) -> __m512i {
    unsafe { _mm512_set_epi64(i[7], i[6], i[5], i[4], i[3], i[2], i[1], i[0]) }
}

#[inline(always)]
unsafe fn perm(a: __m512i, i: [i64; 8], b: __m512i) -> __m512i {
    unsafe { _mm512_permutex2var_epi64(a, idx(i), b) }
}

/// 8つのシード（各 4 ワード）を、ワードごとのレジスタ4本に並べ替える: `w[j]` のレーン k が
/// k 番目のシードの j 番目のワード
#[inline(always)]
unsafe fn load_words(seeds: &[[u8; 32]; 8]) -> [__m512i; 4] {
    unsafe {
        // r[m] = シード 2m と 2m+1（ワード 0〜3 が2つ）
        let p: *const __m512i = seeds.as_ptr().cast();
        let r: [__m512i; 4] = std::array::from_fn(|m| _mm512_loadu_si512(p.add(m)));
        // a = シード 0〜3 のワード 0 と 1、b = ワード 2 と 3（c, d はシード 4〜7）
        let lo01 = [0, 4, 8, 12, 1, 5, 9, 13];
        let lo23 = [2, 6, 10, 14, 3, 7, 11, 15];
        let (a, b) = (perm(r[0], lo01, r[1]), perm(r[0], lo23, r[1]));
        let (c, d) = (perm(r[2], lo01, r[3]), perm(r[2], lo23, r[3]));
        // 前半4レーンにシード 0〜3、後半にシード 4〜7
        let first = [0, 1, 2, 3, 8, 9, 10, 11];
        let second = [4, 5, 6, 7, 12, 13, 14, 15];
        [
            perm(a, first, c),
            perm(a, second, c),
            perm(b, first, d),
            perm(b, second, d),
        ]
        .map(|v| bswap64(v))
    }
}

/// [`load_words`] の逆: ワードごとのレジスタ4本を、8つの 32 バイトに戻して書く
#[inline(always)]
unsafe fn store_words(h: [__m512i; 4], out: &mut [[u8; 32]; 8]) {
    unsafe {
        let h = h.map(|v| bswap64(v));
        // a = シード 0〜3 のワード (0,1) の組を並べたもの、b = (2,3) の組（c, d はシード 4〜7）
        let lo = [0, 8, 1, 9, 2, 10, 3, 11];
        let hi = [4, 12, 5, 13, 6, 14, 7, 15];
        let (a, b) = (perm(h[0], lo, h[1]), perm(h[2], lo, h[3]));
        let (c, d) = (perm(h[0], hi, h[1]), perm(h[2], hi, h[3]));
        // r[m] = シード 2m と 2m+1 のワード 0〜3
        let first = [0, 1, 8, 9, 2, 3, 10, 11];
        let second = [4, 5, 12, 13, 6, 7, 14, 15];
        let r = [
            perm(a, first, b),
            perm(a, second, b),
            perm(c, first, d),
            perm(c, second, d),
        ];
        let p: *mut __m512i = out.as_mut_ptr().cast();
        for (m, v) in r.into_iter().enumerate() {
            _mm512_storeu_si512(p.add(m), v);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
fn prefixes_avx512(seeds: &[[u8; 32]; 8]) -> [[u8; 32]; 8] {
    // SAFETY: この関数の target_feature で、中の unsafe fn が使う命令はすべて使える
    unsafe {
        // メッセージ: シード 32 バイト（ビッグエンディアンの 64 ビット × 4）、0x80、0 埋め、長さ 256 ビット
        let mut w = [_mm512_setzero_si512(); 16];
        w[..4].copy_from_slice(&load_words(seeds));
        w[4] = _mm512_set1_epi64(1 << 63);
        w[15] = _mm512_set1_epi64(256);

        let mut s = IV.map(|h| _mm512_set1_epi64(h.cast_signed()));
        rounds!(s, w;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
            20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
            40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59
            60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
        );

        // 前半 32 バイト = H0〜H3（ビッグエンディアン）
        let h = std::array::from_fn(|j| add(s[j], _mm512_set1_epi64(IV[j].cast_signed())));
        let mut out = [[0u8; 32]; 8];
        store_words(h, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::check;
    use crate::simd_support::available;
    use proptest::prelude::*;
    use rstest::rstest;
    use sha2::{Digest, Sha512};
    use similar_asserts::assert_eq;
    use std::array::from_fn;
    use std::mem::transmute;

    fn want(seed: &[u8; 32]) -> [u8; 32] {
        *Sha512::digest(seed).first_chunk().unwrap()
    }

    /// 8つのシードの前半 32 バイトが、sha2 クレートの SHA-512 と一致する（レーンが混ざらない）。
    /// `fixed` は1つのレーンに入れる決まったシード
    #[rstest]
    #[case::random(None)]
    #[case::zero(Some([0u8; 32]))]
    #[case::ff(Some([0xffu8; 32]))]
    #[case::bytes_0_to_31(Some(from_fn(|i| i as u8)))]
    fn matches_sha2(#[case] fixed: Option<[u8; 32]>, #[values(0, 3, 4, 7)] lane: usize) {
        let Some(avx512) = available("avx512", Avx512::detect()) else {
            return;
        };
        check(256, any::<[[u8; 32]; 8]>(), |mut seeds| {
            if let Some(f) = fixed {
                seeds[lane] = f;
            }
            let got = prefixes(avx512, &seeds);
            for k in 0..8 {
                prop_assert_eq!(got[k], want(&seeds[k]), "#{}", k);
            }
            Ok(())
        });
    }

    /// 並べ替え（load_words / store_words）だけを確かめる: 各シードの各ワードが正しいレーンに入り、
    /// 戻すと元どおり
    #[rstest]
    fn transpose_round_trips() {
        let Some(_) = available("avx512", Avx512::detect()) else {
            return;
        };
        let seeds: [[u8; 32]; 8] = from_fn(|k| from_fn(|i| (k * 32 + i) as u8));
        // SAFETY: available で対応を確かめた
        let (w, back) = unsafe {
            let w = load_words(&seeds);
            let mut back = [[0u8; 32]; 8];
            store_words(w, &mut back);
            (w.map(|v| transmute::<__m512i, [u64; 8]>(v)), back)
        };
        for (j, wj) in w.iter().enumerate() {
            for (k, s) in seeds.iter().enumerate() {
                let word = u64::from_be_bytes(*s[8 * j..].first_chunk().unwrap());
                assert_eq!(wj[k], word, "シード {k} のワード {j}");
            }
        }
        assert_eq!(back, seeds);
    }
}
