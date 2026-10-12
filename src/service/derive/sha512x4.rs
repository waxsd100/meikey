//! AVX2 で4つのシードの SHA-512 を同時に計算する（秘密スカラーを作るため専用）
//!
//! シードは 32 バイトなので、パディングを含めて SHA-512 のブロック1つに収まる。
//! 256 ビットレジスタの 64 ビットのレーン4つに4つのシードの同じワードを入れ、80 ラウンドを4つ同時に進める。
//! スカラーには SHA-512 の前半 32 バイトだけを使うので、出力も前半だけ計算する。

use crate::service::derive::avx2::Avx2;
use std::arch::x86_64::*;

pub(crate) const K: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

/// 初期値 H0〜H7
pub(crate) const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

/// 4つのシードの `SHA-512(seed)[..32]`
pub fn prefixes(avx2: Avx2, seeds: &[[u8; 32]; 4]) -> [[u8; 32]; 4] {
    let _ = avx2;
    // SAFETY: Avx2 があるので、CPU は AVX2 に対応している
    unsafe { prefixes_avx2(seeds) }
}

macro_rules! rotr {
    ($x:expr, $n:literal) => {
        _mm256_or_si256(
            _mm256_srli_epi64::<$n>($x),
            _mm256_slli_epi64::<{ 64 - $n }>($x),
        )
    };
}

// 以下の unsafe fn は AVX2 の命令を呼ぶだけで、AVX2 に対応した CPU でだけ呼ぶこと。
// `#[target_feature]` の関数は `#[inline(always)]` にできないので、入口（prefixes_avx2）だけに付けて
// ほかは展開させる（[`crate::service::derive::avx2`] と同じ）

#[inline(always)]
unsafe fn xor3(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    unsafe { _mm256_xor_si256(_mm256_xor_si256(a, b), c) }
}

#[inline(always)]
unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
    unsafe { _mm256_add_epi64(a, b) }
}

/// 8 ビットの右回転はバイトの並べ替え1命令でできる
#[inline(always)]
unsafe fn rotr8(x: __m256i) -> __m256i {
    unsafe {
        let m = _mm256_setr_epi8(
            1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10,
            11, 12, 13, 14, 15, 8,
        );
        _mm256_shuffle_epi8(x, m)
    }
}

/// t 番目のラウンド（t は定数で呼ぶので、添字や分岐は展開時に消える）
#[inline(always)]
unsafe fn round(s: &mut [__m256i; 8], w: &mut [__m256i; 16], t: usize) {
    unsafe {
        if t >= 16 {
            // W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16]（16 個の輪で持つ）
            let (w2, w15) = (w[(t - 2) % 16], w[(t - 15) % 16]);
            let s1 = xor3(rotr!(w2, 19), rotr!(w2, 61), _mm256_srli_epi64::<6>(w2));
            let s0 = xor3(rotr!(w15, 1), rotr8(w15), _mm256_srli_epi64::<7>(w15));
            w[t % 16] = add(add(s1, w[(t - 7) % 16]), add(s0, w[t % 16]));
        }
        let [a, b, c, d, e, f, g, h] = *s;
        let sum1 = xor3(rotr!(e, 14), rotr!(e, 18), rotr!(e, 41));
        let ch = _mm256_xor_si256(_mm256_and_si256(e, f), _mm256_andnot_si256(e, g));
        let k = _mm256_set1_epi64x(K[t].cast_signed());
        let t1 = add(add(h, sum1), add(add(ch, k), w[t % 16]));
        let sum0 = xor3(rotr!(a, 28), rotr!(a, 34), rotr!(a, 39));
        let maj = _mm256_or_si256(
            _mm256_and_si256(a, b),
            _mm256_and_si256(c, _mm256_or_si256(a, b)),
        );
        *s = [add(t1, add(sum0, maj)), a, b, c, add(d, t1), e, f, g];
    }
}

/// 80 ラウンドを書き下す（ラウンドの番号が定数になり、W の添字や K の値が展開時に決まる）
macro_rules! rounds {
    ($s:ident, $w:ident; $($t:literal)*) => {
        $( round(&mut $s, &mut $w, $t); )*
    };
}

#[target_feature(enable = "avx2")]
fn prefixes_avx2(seeds: &[[u8; 32]; 4]) -> [[u8; 32]; 4] {
    // メッセージ: シード 32 バイト（ビッグエンディアンの 64 ビット × 4）、0x80、0 埋め、長さ 256 ビット
    let mut w = [_mm256_setzero_si256(); 16];
    for (j, wj) in w.iter_mut().take(4).enumerate() {
        let word =
            |k: usize| i64::from_be_bytes(*seeds[k][8 * j..].first_chunk().expect("32 バイト"));
        *wj = _mm256_set_epi64x(word(3), word(2), word(1), word(0));
    }
    w[4] = _mm256_set1_epi64x(1 << 63);
    w[15] = _mm256_set1_epi64x(256);

    let mut s = IV.map(|h| _mm256_set1_epi64x(h.cast_signed()));
    // SAFETY: この関数は AVX2 を有効にしてコンパイルされ、AVX2 に対応した CPU でだけ呼ばれる
    unsafe {
        rounds!(s, w;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
            20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
            40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59
            60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
        );
    }

    // 前半 32 バイト = H0〜H3（ビッグエンディアン）
    let mut out = [[0u8; 32]; 4];
    for j in 0..4 {
        let h = _mm256_add_epi64(s[j], _mm256_set1_epi64x(IV[j].cast_signed()));
        // SAFETY: __m256i と [u64; 4] は同じ大きさで、どんなビット列も有効な値
        let lanes = unsafe { std::mem::transmute::<__m256i, [u64; 4]>(h) };
        for (o, v) in out.iter_mut().zip(lanes) {
            o[8 * j..8 * j + 8].copy_from_slice(&v.to_be_bytes());
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::check;
    use proptest::prelude::*;
    use rstest::rstest;
    use sha2::{Digest, Sha512};

    fn want(seed: &[u8; 32]) -> [u8; 32] {
        *Sha512::digest(seed).first_chunk().unwrap()
    }

    /// 4つのシードの前半 32 バイトが、sha2 クレートの SHA-512 と一致する（レーンが混ざらない）
    #[rstest]
    #[case::random(None)]
    #[case::zero(Some([0u8; 32]))]
    #[case::ff(Some([0xffu8; 32]))]
    fn matches_sha2(#[case] fixed: Option<[u8; 32]>) {
        let Some(avx2) = crate::simd_support::available("avx2", Avx2::detect()) else {
            return;
        };
        check(1024, any::<[[u8; 32]; 4]>(), |mut seeds| {
            if let Some(f) = fixed {
                seeds[1] = f;
            }
            let got = prefixes(avx2, &seeds);
            for k in 0..4 {
                prop_assert_eq!(got[k], want(&seeds[k]), "#{}", k);
            }
            Ok(())
        });
    }
}
