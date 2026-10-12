//! GF(2^255 - 19) の体演算（51ビット × 5 リム）
//!
//! 探索の高速化専用。定数時間ではないので、秘密を扱う汎用用途には使わないこと。

use std::ops::{Add, Mul, Sub};

const MASK: u64 = (1 << 51) - 1;

/// 体 GF(2^255 - 19) の元。値は `Σ l[i]·2^(51·i) mod p`（同じ値に複数の表し方がある）
///
/// 各演算が正しく動く入力の条件（tests の proptest で、この境界ちょうどまで検証している）:
/// - `*`・`square`: 両方とも各リム 2^54 未満
/// - `-`・`neg`: 引く側の各リム 2^54 未満
/// - `to_bytes`: 各リム 2^63 未満
///
/// `*` と `-` の結果は各リム 2^52 未満になるので、その和（2^53 未満）や
/// 和と結果の和（2^54 未満）はそのまま `*` に渡せる。
#[derive(Clone, Copy, Debug)]
pub struct Fe(pub [u64; 5]);

impl Fe {
    pub const ZERO: Fe = Fe([0; 5]);
    pub const ONE: Fe = Fe([1, 0, 0, 0, 0]);

    /// リトルエンディアン32バイトから（最上位ビットは無視）。AMX の経路とテストで使う
    #[cfg(any(test, all(target_arch = "x86_64", feature = "amx")))]
    pub fn from_bytes(b: &[u8; 32]) -> Fe {
        let load = |i: usize| u64::from_le_bytes(*b[i..].first_chunk().unwrap());
        Fe([
            load(0) & MASK,
            (load(6) >> 3) & MASK,
            (load(12) >> 6) & MASK,
            (load(19) >> 1) & MASK,
            (load(24) >> 12) & MASK,
        ])
    }

    /// 正規化してリトルエンディアン32バイトに
    pub fn to_bytes(self) -> [u8; 32] {
        let l = self.reduce();
        let mut out = [0u8; 32];
        let words = [
            l[0] | l[1] << 51,
            l[1] >> 13 | l[2] << 38,
            l[2] >> 26 | l[3] << 25,
            l[3] >> 39 | l[4] << 12,
        ];
        for (chunk, w) in out.as_chunks_mut::<8>().0.iter_mut().zip(words) {
            *chunk = w.to_le_bytes();
        }
        out
    }

    /// 正規化したリム（0 以上 p 未満の値で、各リム 2^51 未満）
    pub fn reduce(self) -> [u64; 5] {
        let mut l = self.carry().0;
        // l < 2p なので、p 以上なら p を引く（l + 19 が 2^255 を超えるかで判定）
        let mut q = (l[0] + 19) >> 51;
        q = (l[1] + q) >> 51;
        q = (l[2] + q) >> 51;
        q = (l[3] + q) >> 51;
        q = (l[4] + q) >> 51;
        l[0] += 19 * q;
        l[1] += l[0] >> 51;
        l[0] &= MASK;
        l[2] += l[1] >> 51;
        l[1] &= MASK;
        l[3] += l[2] >> 51;
        l[2] &= MASK;
        l[4] += l[3] >> 51;
        l[3] &= MASK;
        l[4] &= MASK;
        l
    }

    /// 各リムを 2^51 + α に収める
    #[inline(always)]
    fn carry(self) -> Fe {
        let mut l = self.0;
        let c = l[0] >> 51;
        l[0] &= MASK;
        l[1] += c;
        let c = l[1] >> 51;
        l[1] &= MASK;
        l[2] += c;
        let c = l[2] >> 51;
        l[2] &= MASK;
        l[3] += c;
        let c = l[3] >> 51;
        l[3] &= MASK;
        l[4] += c;
        let c = l[4] >> 51;
        l[4] &= MASK;
        l[0] += c * 19;
        Fe(l)
    }

    #[inline(always)]
    pub fn square(self) -> Fe {
        self * self
    }

    pub fn neg(self) -> Fe {
        Fe::ZERO - self
    }

    /// 逆元（フェルマーの小定理: a^(p-2)）
    pub fn invert(self) -> Fe {
        // p - 2 = 2^255 - 21 = 0b0111...1110_1011
        // 上位 250 ビットはすべて 1、下位 5 ビットは 01011
        let a = self;
        let a2 = a.square();
        let a9 = a2.square().square() * a; // a^9
        let a11 = a9 * a2; // a^11
        let a_5_0 = a11.square() * a9; // a^(2^5 - 1) = a^31
        let a_10_0 = a_5_0.pow2k(5) * a_5_0;
        let a_20_0 = a_10_0.pow2k(10) * a_10_0;
        let a_40_0 = a_20_0.pow2k(20) * a_20_0;
        let a_50_0 = a_40_0.pow2k(10) * a_10_0;
        let a_100_0 = a_50_0.pow2k(50) * a_50_0;
        let a_200_0 = a_100_0.pow2k(100) * a_100_0;
        let a_250_0 = a_200_0.pow2k(50) * a_50_0; // a^(2^250 - 1)
        a_250_0.pow2k(5) * a11 // a^(2^255 - 32 + 11) = a^(p - 2)
    }

    fn pow2k(self, k: u32) -> Fe {
        (0..k).fold(self, |x, _| x.square())
    }

    /// 複数の元の逆元をまとめて計算する（Montgomery の一括逆元: 逆元1回 + 乗算 3(n-1) 回）
    pub fn batch_invert(xs: &mut [Fe], scratch: &mut Vec<Fe>) {
        scratch.clear();
        let mut acc = Fe::ONE;
        for x in xs.iter() {
            scratch.push(acc);
            acc = acc * *x;
        }
        let mut inv = acc.invert();
        for (x, prefix) in xs.iter_mut().zip(scratch.iter()).rev() {
            let next = inv * *x;
            *x = inv * *prefix;
            inv = next;
        }
    }
}

impl Add for Fe {
    type Output = Fe;
    #[inline(always)]
    fn add(self, b: Fe) -> Fe {
        let (a, b) = (self.0, b.0);
        Fe([
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
            a[3] + b[3],
            a[4] + b[4],
        ])
    }
}

impl Sub for Fe {
    type Output = Fe;
    #[inline(always)]
    fn sub(self, b: Fe) -> Fe {
        // 負にならないよう 16p を足してから引く（b のリムは 2^54 未満が前提）
        const P16_0: u64 = 16 * ((1 << 51) - 19);
        const P16_N: u64 = 16 * ((1 << 51) - 1);
        let (a, b) = (self.0, b.0);
        Fe([
            a[0] + P16_0 - b[0],
            a[1] + P16_N - b[1],
            a[2] + P16_N - b[2],
            a[3] + P16_N - b[3],
            a[4] + P16_N - b[4],
        ])
        .carry()
    }
}

impl Mul for Fe {
    type Output = Fe;
    #[inline(always)]
    fn mul(self, b: Fe) -> Fe {
        #[inline(always)]
        fn m(x: u64, y: u64) -> u128 {
            u128::from(x) * u128::from(y)
        }
        let (a, b) = (self.0, b.0);
        // 2^255 = 19 (mod p) なので、桁あふれ分は 19 倍して下に回す
        let b1 = b[1] * 19;
        let b2 = b[2] * 19;
        let b3 = b[3] * 19;
        let b4 = b[4] * 19;

        let c0 = m(a[0], b[0]) + m(a[4], b1) + m(a[3], b2) + m(a[2], b3) + m(a[1], b4);
        let mut c1 = m(a[1], b[0]) + m(a[0], b[1]) + m(a[4], b2) + m(a[3], b3) + m(a[2], b4);
        let mut c2 = m(a[2], b[0]) + m(a[1], b[1]) + m(a[0], b[2]) + m(a[4], b3) + m(a[3], b4);
        let mut c3 = m(a[3], b[0]) + m(a[2], b[1]) + m(a[1], b[2]) + m(a[0], b[3]) + m(a[4], b4);
        let mut c4 = m(a[4], b[0]) + m(a[3], b[1]) + m(a[2], b[2]) + m(a[1], b[3]) + m(a[0], b[4]);

        c1 += u128::from((c0 >> 51) as u64);
        let mut l0 = (c0 as u64) & MASK;
        c2 += u128::from((c1 >> 51) as u64);
        let l1 = (c1 as u64) & MASK;
        c3 += u128::from((c2 >> 51) as u64);
        let l2 = (c2 as u64) & MASK;
        c4 += u128::from((c3 >> 51) as u64);
        let l3 = (c3 as u64) & MASK;
        let carry = (c4 >> 51) as u64;
        let l4 = (c4 as u64) & MASK;

        l0 += carry * 19;
        let l1 = l1 + (l0 >> 51);
        l0 &= MASK;
        Fe([l0, l1, l2, l3, l4])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    /// p + k（k ≤ 18 なら 255 ビットに収まる）
    const fn p_plus(k: u8) -> [u8; 32] {
        let mut b = [0xff; 32];
        b[0] = 0xed + k;
        b[31] = 0x7f;
        b
    }

    const fn small(k: u8) -> [u8; 32] {
        let mut b = [0; 32];
        b[0] = k;
        b
    }

    const P_MINUS_1: [u8; 32] = {
        let mut b = p_plus(0);
        b[0] -= 1;
        b
    };
    const P_MINUS_2: [u8; 32] = {
        let mut b = p_plus(0);
        b[0] -= 2;
        b
    };
    /// 1/2 = (p + 1) / 2
    const INV2: [u8; 32] = {
        let mut b = [0xff; 32];
        b[0] = 0xf7;
        b[31] = 0x3f;
        b
    };
    const SAMPLE: [u8; 32] = [
        0x13, 0x37, 0xc0, 0xde, 0x42, 0x00, 0xff, 0x7a, 0x55, 0xaa, 0x01, 0x02, 0x03, 0x04, 0x05,
        0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14,
        0x15, 0x16,
    ];

    #[derive(Debug, Clone, Copy)]
    enum Op {
        /// to_bytes（RFC 8032 Section 5.1.2 の正規形）
        Normalize,
        /// 引数との積（交換法則も確かめる）
        Mul([u8; 32]),
        Invert,
    }

    /// 答えが分かっている値（p = 2^255 - 19 の境界、-1、1/2 など）
    #[rstest]
    #[case::normalize_p(Op::Normalize, p_plus(0), small(0))]
    #[case::normalize_p_plus_1(Op::Normalize, p_plus(1), small(1))]
    #[case::normalize_p_plus_2(Op::Normalize, p_plus(2), small(2))]
    #[case::normalize_2_255_minus_1(Op::Normalize, p_plus(18), small(18))]
    #[case::normalize_p_minus_1(Op::Normalize, P_MINUS_1, P_MINUS_1)]
    #[case::mul_neg_one_squared(Op::Mul(P_MINUS_1), P_MINUS_1, small(1))]
    #[case::mul_two_times_half(Op::Mul(INV2), small(2), small(1))]
    #[case::mul_neg_one_times_two(Op::Mul(small(2)), P_MINUS_1, P_MINUS_2)]
    #[case::mul_zero(Op::Mul(SAMPLE), small(0), small(0))]
    #[case::mul_one(Op::Mul(SAMPLE), small(1), SAMPLE)]
    #[case::invert_one(Op::Invert, small(1), small(1))]
    #[case::invert_neg_one(Op::Invert, P_MINUS_1, P_MINUS_1)]
    #[case::invert_two(Op::Invert, small(2), INV2)]
    #[case::invert_half(Op::Invert, INV2, small(2))]
    #[case::invert_zero_is_zero(Op::Invert, small(0), small(0))]
    fn known_values(#[case] op: Op, #[case] a: [u8; 32], #[case] want: [u8; 32]) {
        let x = Fe::from_bytes(&a);
        let got = match op {
            Op::Normalize => x.to_bytes(),
            Op::Mul(b) => {
                let b = Fe::from_bytes(&b);
                assert_eq!((b * x).to_bytes(), (x * b).to_bytes(), "交換法則");
                (x * b).to_bytes()
            }
            Op::Invert => x.invert().to_bytes(),
        };
        assert_eq!(got, want);
    }

    // ---------------------------------------------------------------------------
    // proptest: 多倍長整数（num-bigint）で計算した値と比べる
    // ---------------------------------------------------------------------------

    use crate::property::check;
    use crate::rfc8032;
    use crate::service::derive::BATCH;
    use num_bigint::BigInt;
    use num_integer::Integer;
    use proptest::prelude::*;
    use std::ops::Range;

    /// Fe が表す整数（mod を取らない）
    fn value(x: Fe) -> BigInt {
        x.0.iter()
            .rev()
            .fold(BigInt::from(0), |acc, &l| (acc << 51) + BigInt::from(l))
    }

    fn modp(x: &BigInt) -> BigInt {
        x.mod_floor(&rfc8032::P)
    }

    /// 0 以上 2^bits 未満のリム。境界の値（0、最大、2^51 前後）を多めに混ぜる
    fn limb(bits: u32) -> impl Strategy<Value = u64> {
        let max = (1u64 << bits) - 1;
        prop_oneof![
            1 => Just(0),
            1 => Just(max),
            1 => Just(MASK.min(max)),
            1 => Just((MASK + 1).min(max)),
            6 => 0..=max,
        ]
    }

    fn fe(bits: u32) -> impl Strategy<Value = Fe> {
        prop::array::uniform5(limb(bits)).prop_map(Fe)
    }

    fn limbs_below(x: Fe, bits: u32) -> bool {
        x.0.iter().all(|&l| l < 1 << bits)
    }

    /// 多倍長整数で同じ計算をして比べる演算
    #[derive(Debug, Clone, Copy)]
    enum Arith {
        /// mod を取らずにリムごとに足すだけ（値は整数としてそのまま足し算になる）
        Add,
        Sub,
        Neg,
        Mul,
        Square,
        /// RFC 8032 Section 5.1.1: x^-1 = x^(p-2)（0 の逆元は 0 になる）
        Invert,
    }

    impl Arith {
        /// 入力の各リムの上限（ビット数）。`Fe` のドキュメントに書いた条件ちょうど
        fn input_bits(self) -> u32 {
            match self {
                Arith::Add => 53,
                _ => 54,
            }
        }

        fn apply(self, a: Fe, b: Fe) -> Fe {
            match self {
                Arith::Add => a + b,
                Arith::Sub => a - b,
                Arith::Neg => a.neg(),
                Arith::Mul => a * b,
                Arith::Square => a.square(),
                Arith::Invert => a.invert(),
            }
        }

        /// 多倍長整数での答え（加算は整数のまま、ほかは mod p）
        fn want(self, a: &BigInt, b: &BigInt) -> BigInt {
            match self {
                Arith::Add => a + b,
                Arith::Sub => modp(&(a - b)),
                Arith::Neg => modp(&-a),
                Arith::Mul => modp(&(a * b)),
                Arith::Square => modp(&(a * a)),
                Arith::Invert => a.modpow(&(&*rfc8032::P - 2), &rfc8032::P),
            }
        }
    }

    /// 体演算が多倍長整数の計算と一致する（入力は各リムの上限ちょうどまで）。
    /// 加算以外の結果は各リム 2^52 未満で、その和はそのまま乗算に渡せる（`Fe` の約束）
    #[rstest]
    fn arithmetic_matches_bigint(
        #[values(
            Arith::Add,
            Arith::Sub,
            Arith::Neg,
            Arith::Mul,
            Arith::Square,
            Arith::Invert
        )]
        op: Arith,
    ) {
        let bits = op.input_bits();
        check(4096, (fe(bits), fe(bits)), |(a, b)| {
            let (got, want) = (op.apply(a, b), op.want(&value(a), &value(b)));
            if let Arith::Add = op {
                prop_assert_eq!(value(got), want);
            } else {
                prop_assert_eq!(modp(&value(got)), want);
                prop_assert!(limbs_below(got, 52), "結果のリムが 2^52 以上: {:?}", got);
            }
            Ok(())
        });
    }

    /// 乗算・減算の結果どうしの和は、そのまま乗算に渡せる（文書にした条件の確認）
    #[rstest]
    fn sums_of_results_can_be_multiplied() {
        check(4096, (fe(54), fe(54), fe(54), fe(54)), |(a, b, c, d)| {
            let (x, y) = (a * b, c - d);
            let z = (x + y) + x; // 2^52 + 2^52 + 2^52 < 2^54
            prop_assert!(limbs_below(z, 54));
            prop_assert_eq!(modp(&value(z * z)), modp(&(value(z) * value(z))));
            Ok(())
        });
    }

    /// RFC 8032 Section 5.1.2: 0 以上 p 未満の値を 32 バイトのリトルエンディアンで（正規形）
    #[rstest]
    fn to_bytes_is_canonical_encoding() {
        check(4096, fe(63), |a| {
            prop_assert_eq!(a.to_bytes(), rfc8032::to_le32(&modp(&value(a))));
            Ok(())
        });
    }

    /// 最上位ビット（RFC 8032 では x の符号）を無視して読む
    #[rstest]
    fn from_bytes_ignores_top_bit() {
        check(4096, any::<[u8; 32]>(), |b| {
            let x = Fe::from_bytes(&b);
            prop_assert!(limbs_below(x, 51));
            let want = rfc8032::from_le(&b) & ((BigInt::from(1) << 255) - 1);
            prop_assert_eq!(value(x), want);
            Ok(())
        });
    }

    /// 一括逆元の各結果は、元に掛けると 1 になる（= 逆元）。両端は単独の逆元とも比べる。
    /// 乗算と単独の逆元の正しさは `arithmetic_matches_bigint` で多倍長整数と照合済み。
    /// 1件で最大 256 要素を扱うので、回数は少なめ（要素数の合計では十分に多い）
    #[rstest]
    #[case::lengths_1_to_63(1..64, 448)]
    #[case::batch_of_256(BATCH..BATCH + 1, 64)] // 探索で実際に使う長さ
    fn batch_invert_gives_inverses(#[case] len: Range<usize>, #[case] cases: u32) {
        check(cases, prop::collection::vec(fe(54), len), |xs| {
            // 0 を含むと全体が 0 になる（呼び出し側は 0 を渡さない）ので除く
            let xs: Vec<Fe> = xs.into_iter().filter(|x| x.to_bytes() != [0; 32]).collect();
            let mut inv = xs.clone();
            Fe::batch_invert(&mut inv, &mut Vec::new());
            for (x, i) in xs.iter().zip(&inv) {
                prop_assert_eq!((*x * *i).to_bytes(), Fe::ONE.to_bytes());
            }
            for k in [0, xs.len().saturating_sub(1)] {
                if let (Some(x), Some(i)) = (xs.get(k), inv.get(k)) {
                    prop_assert_eq!(i.to_bytes(), x.invert().to_bytes());
                }
            }
            Ok(())
        });
    }
}
