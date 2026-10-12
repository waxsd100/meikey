//! AVX-512 IFMA の体演算: 51 ビット × 5 リム（[`crate::service::derive::field::Fe`] と同じ切り方）
//!
//! `vpmadd52luq` / `vpmadd52huq` は、各レーンの下位 52 ビットどうしの積（104 ビット）の下位 52 ビット・
//! 上位 52 ビットを 64 ビットのアキュムレータに足す。51 ビットのリムなら積を1組2命令で作れるので、
//! 1回の乗算は 25 組 = 50 命令（`vpmuludq` で 25.5 ビット × 10 リムなら 100 回の乗算と 100 回の加算）。
//!
//! 気をつける点は、IFMA が入力の下位 52 ビットしか見ないこと。2^52 以上のリムを渡すと上のビットが
//! 黙って捨てられ、結果が壊れる。そこで「乗算に渡す値はリムが 2^52 未満」を型の約束にし、
//! `add`・`sub`・`mul` は結果を必ず繰り上げて 2^51 + 2^17 未満にする（[`Fe5`] を参照）。

use super::{Field8, Ifma, Table8};
use crate::service::derive::comb::Extended;
use crate::service::derive::field::Fe;
use std::arch::x86_64::*;
use std::sync::LazyLock;

const MASK: u64 = (1 << 51) - 1;

/// 2p のリム（引き算・符号反転で負にならないように足す）: 2^52 - 38, 2^52 - 2, …
const P2: [u64; 5] = [2 * (MASK - 18), 2 * MASK, 2 * MASK, 2 * MASK, 2 * MASK];

/// 体の元を8つ（51 ビット × 5 リム、値は `Σ l[i]·2^(51·i)`）。
///
/// 各演算が正しく動く入力の条件と、結果の大きさ（tests の proptest で、この境界ちょうどまで検証している）:
/// - `mul(a, b)`: a・b ともリムが 2^52 未満（IFMA は下位 52 ビットしか見ない）
/// - `add(a, b)`: a + b のリムが 2^63 未満
/// - `sub(a, b)`, `neg_masked`: b（反転する値）のリムが 2p のリム（2^52 - 38）以下、a は 2^62 未満
///
/// `add`・`sub`・`mul`・`carry` の結果はリムが 2^51 + 2^17 未満（＝乗算にも引き算の b にも渡せる）。
/// `neg_masked` の結果は 2^52 未満（＝乗算に渡せる）。
#[derive(Clone, Copy)]
pub(crate) struct Fe5([__m512i; 5]);

/// 繰り上げの結果のリムの上限（テストで確かめる）
#[cfg(test)]
const CARRIED: u64 = (1 << 51) + (1 << 17);

#[inline(always)]
unsafe fn splat5(l: [u64; 5]) -> [__m512i; 5] {
    l.map(|x| _mm512_set1_epi64(x as i64))
}

/// 19·x（x < 2^59。シフト2回と加算2回: 19 = 16 + 2 + 1）
#[inline(always)]
unsafe fn times19(x: __m512i) -> __m512i {
    _mm512_add_epi64(
        _mm512_add_epi64(x, _mm512_slli_epi64::<1>(x)),
        _mm512_slli_epi64::<4>(x),
    )
}

/// 各リムの 51 ビットより上をまとめて1つ上のリムへ送る（5 リム同時。最上位からの繰り上がりは
/// 2^255 = 19 なので 19 倍して最下位へ）。入力のリムが 2^63 未満なら、繰り上がりは 2^12 未満、
/// その 19 倍は 2^17 未満なので、結果のリムは 2^51 + 2^17 未満
#[inline(always)]
unsafe fn carry(r: [__m512i; 5]) -> Fe5 {
    let mask = _mm512_set1_epi64(MASK as i64);
    let mut l = [_mm512_setzero_si512(); 5];
    let mut c = [_mm512_setzero_si512(); 5];
    for i in 0..5 {
        l[i] = _mm512_and_si512(r[i], mask);
        c[i] = _mm512_srli_epi64::<51>(r[i]);
    }
    // 最上位からの繰り上がり（2^12 未満）の 19 倍は 52 ビットに収まるので、IFMA 1命令で足せる
    let nineteen = _mm512_set1_epi64(19);
    Fe5([
        _mm512_madd52lo_epu64(l[0], c[4], nineteen),
        _mm512_add_epi64(l[1], c[0]),
        _mm512_add_epi64(l[2], c[1]),
        _mm512_add_epi64(l[3], c[2]),
        _mm512_add_epi64(l[4], c[3]),
    ])
}

impl Field8 for Fe5 {
    fn pack(l: [u64; 5]) -> [u64; 5] {
        l
    }

    #[inline(always)]
    unsafe fn unpack(w: &[__m512i; 5]) -> Fe5 {
        Fe5(*w)
    }

    #[inline(always)]
    unsafe fn splat(l: [u64; 5]) -> Fe5 {
        Fe5(splat5(l))
    }

    #[inline(always)]
    unsafe fn add(&self, b: &Fe5) -> Fe5 {
        let mut r = self.0;
        for i in 0..5 {
            r[i] = _mm512_add_epi64(self.0[i], b.0[i]);
        }
        carry(r)
    }

    /// a + 2p - b
    #[inline(always)]
    unsafe fn sub(&self, b: &Fe5) -> Fe5 {
        let p2 = splat5(P2);
        let mut r = self.0;
        for i in 0..5 {
            r[i] = _mm512_sub_epi64(_mm512_add_epi64(self.0[i], p2[i]), b.0[i]);
        }
        carry(r)
    }

    /// 積 `a[i]·b[j] = lo + hi·2^52` を、lo は `51·(i+j)` ビット目、hi は `51·(i+j+1) + 1` ビット目に足す。
    /// hi は1つ上のリムの位置から 1 ビットずれるので、まとめてから 2 倍する
    #[inline(always)]
    unsafe fn mul(&self, b: &Fe5) -> Fe5 {
        let (a, b) = (&self.0, &b.0);
        // lo[k] = Σ_{i+j=k} lo52(a[i]·b[j])、hi[k] = Σ_{i+j=k} hi52(a[i]·b[j])。
        // どちらも 5 項以下で、各項は 2^52 未満なので 5·2^52 未満
        let z = _mm512_setzero_si512();
        let (mut lo, mut hi) = ([z; 9], [z; 9]);
        for i in 0..5 {
            for j in 0..5 {
                lo[i + j] = _mm512_madd52lo_epu64(lo[i + j], a[i], b[j]);
                hi[i + j] = _mm512_madd52hi_epu64(hi[i + j], a[i], b[j]);
            }
        }
        // c[k] = lo[k] + 2·hi[k-1]（< 15·2^52 < 2^56）は 51·k ビット目の値。
        // 2^255 = 19 (mod p) なので、c[k+5] は 19 倍して c[k] に足す（< 20·2^56 < 2^61）
        let c = |k: usize| match k {
            0 => lo[0],
            9 => _mm512_add_epi64(hi[8], hi[8]),
            _ => _mm512_add_epi64(lo[k], _mm512_add_epi64(hi[k - 1], hi[k - 1])),
        };
        let mut r = [z; 5];
        for k in 0..5 {
            r[k] = _mm512_add_epi64(c(k), times19(c(k + 5)));
        }
        carry(r)
    }

    /// `add`・`sub`・`mul` の結果は繰り上げ済みなので何もしない
    #[inline(always)]
    unsafe fn carry(&self) -> Fe5 {
        *self
    }

    #[inline(always)]
    unsafe fn blend(k: __mmask8, a: &Fe5, b: &Fe5) -> Fe5 {
        let mut r = a.0;
        for i in 0..5 {
            r[i] = _mm512_mask_blend_epi64(k, a.0[i], b.0[i]);
        }
        Fe5(r)
    }

    /// k のレーンだけ 2p - x
    #[inline(always)]
    unsafe fn neg_masked(&self, k: __mmask8) -> Fe5 {
        let p2 = splat5(P2);
        let mut r = self.0;
        for i in 0..5 {
            r[i] = _mm512_mask_sub_epi64(self.0[i], k, p2[i], self.0[i]);
        }
        Fe5(r)
    }

    fn lanes(&self) -> [Fe; 8] {
        // SAFETY: __m512i と [u64; 8] は同じ大きさで、どんなビット列も有効な値
        let l = self
            .0
            .map(|v| unsafe { std::mem::transmute::<__m512i, [u64; 8]>(v) });
        std::array::from_fn(|k| Fe(std::array::from_fn(|i| l[i][k])))
    }
}

static TABLE: LazyLock<Table8> = LazyLock::new(Table8::build::<Fe5>);

/// 表を作っておく（探索の時間に含めないため）
pub fn prepare(_: Ifma) {
    LazyLock::force(&TABLE);
}

/// [`crate::service::derive::comb::Table::mul_base_batch`] と同じ計算を、8鍵ずつ AVX-512 IFMA で（`N` は 8 の倍数）
pub fn mul_base_batch<const N: usize>(
    ifma: Ifma,
    scalars: &[[u8; 32]; N],
    out: &mut [Extended; N],
) {
    let _ = ifma;
    // SAFETY: Ifma があるので、CPU は下の target_feature の命令に対応している
    unsafe { run(&TABLE, scalars, out) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512ifma")]
unsafe fn run<const N: usize>(table: &Table8, scalars: &[[u8; 32]; N], out: &mut [Extended; N]) {
    // SAFETY: 呼び出し元が CPU の対応を確かめている
    unsafe { super::mul_base_batch::<Fe5, N>(table, scalars, out) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::check;
    use crate::rfc8032;
    use crate::simd_support::available;
    use num_bigint::BigInt;
    use num_integer::Integer;
    use proptest::prelude::*;
    use rstest::rstest;

    fn ifma() -> Option<Ifma> {
        available("avx512-ifma", Ifma::detect())
    }

    fn fe5(lanes: [[u64; 5]; 8]) -> Fe5 {
        Fe5(std::array::from_fn(|i| {
            let v: [u64; 8] = std::array::from_fn(|k| lanes[k][i]);
            // SAFETY: 同じ大きさで、どんなビット列も有効な値
            unsafe { std::mem::transmute::<[u64; 8], __m512i>(v) }
        }))
    }

    fn raw_lanes(f: &Fe5) -> [[u64; 5]; 8] {
        f.lanes().map(|fe| fe.0)
    }

    /// リムが表す整数（mod を取らない）
    fn value(l: &[u64; 5]) -> BigInt {
        (0..5).fold(BigInt::from(0), |acc, i| {
            acc + (BigInt::from(l[i]) << (51 * i))
        })
    }

    fn modp(x: &BigInt) -> BigInt {
        x.mod_floor(&rfc8032::P)
    }

    /// 各リムが `max` 以下の元を8レーン。境界の値（0、`max`、2^51 前後）を多めに混ぜる
    fn fe5_upto(max: [u64; 5]) -> impl Strategy<Value = [[u64; 5]; 8]> {
        any::<[[u64; 5]; 8]>().prop_map(move |r| {
            r.map(|l| {
                std::array::from_fn(|i| match l[i] % 11 {
                    0 => 0,
                    1..=3 => max[i],
                    4 => max[i].min(MASK),
                    5 => max[i].min(MASK + 1),
                    _ => (l[i] >> 4) % (max[i] + 1),
                })
            })
        })
    }

    #[derive(Debug, Clone, Copy)]
    enum Arith {
        Add,
        Sub,
        Mul,
        Neg,
    }

    /// 8レーンの体演算が、多倍長整数の計算と一致する（入力は [`Fe5`] に書いた条件ちょうどまで）。
    /// 結果の大きさも [`Fe5`] のとおり。`neg_masked` はマスクの立ったレーンだけ反転する
    #[rstest]
    fn arithmetic_matches_bigint(
        #[values(Arith::Add, Arith::Sub, Arith::Mul, Arith::Neg)] op: Arith,
    ) {
        let Some(_) = ifma() else { return };
        // (a のリムの上限, b のリムの上限)
        let below52 = [(1 << 52) - 1; 5];
        let (a_max, b_max) = match op {
            Arith::Add => ([(1 << 62) - 1; 5], [(1 << 62) - 1; 5]),
            Arith::Sub => ([(1 << 62) - 1; 5], P2),
            Arith::Mul => (below52, below52),
            Arith::Neg => (P2, below52),
        };
        check(
            1024,
            (fe5_upto(a_max), fe5_upto(b_max), any::<u8>()),
            |(a, b, k)| {
                // SAFETY: ifma() で対応を確かめた
                let got = unsafe {
                    let (x, y) = (fe5(a), fe5(b));
                    match op {
                        Arith::Add => x.add(&y),
                        Arith::Sub => x.sub(&y),
                        Arith::Mul => x.mul(&y),
                        Arith::Neg => x.neg_masked(k),
                    }
                };
                let g = raw_lanes(&got);
                for l in 0..8 {
                    let (va, vb, vg) = (value(&a[l]), value(&b[l]), value(&g[l]));
                    match op {
                        Arith::Add => prop_assert_eq!(modp(&vg), modp(&(va + vb))),
                        Arith::Sub => prop_assert_eq!(modp(&vg), modp(&(va - vb))),
                        Arith::Mul => prop_assert_eq!(modp(&vg), modp(&(va * vb))),
                        Arith::Neg if k >> l & 1 == 1 => prop_assert_eq!(modp(&vg), modp(&-va)),
                        Arith::Neg => prop_assert_eq!(g[l], a[l]),
                    }
                    let limit = match op {
                        Arith::Neg => 1 << 52,
                        _ => CARRIED,
                    };
                    prop_assert!(g[l].iter().all(|&x| x < limit), "上限を超えた: {:?}", g[l]);
                }
                Ok(())
            },
        );
    }

    /// 2^52 以上のリムは IFMA に黙って切り捨てられる（＝ [`Fe5`] の約束が要る理由）
    #[rstest]
    fn ifma_ignores_bits_above_52() {
        let Some(_) = ifma() else { return };
        let a = [[1 << 52, 0, 0, 0, 0]; 8];
        let one = [[1, 0, 0, 0, 0]; 8];
        // SAFETY: ifma() で対応を確かめた
        let got = unsafe { fe5(a).mul(&fe5(one)) };
        assert_eq!(raw_lanes(&got)[0], [0; 5], "2^52 × 1 が 0 になる");
    }
}
