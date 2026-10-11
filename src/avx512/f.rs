//! AVX-512F まで（IFMA の無い Skylake-SP・Cascade Lake など）の体演算: 25.5 ビット × 10 リム
//!
//! この世代の 512 ビットの乗算は `vpmuludq`（32×32→64 ビット）だけなので、積の和が 64 ビットに
//! 収まるようリムを 25.5 ビットにする（偶数番目のリムは 26 ビット、奇数番目は 25 ビット。値は
//! `Σ l[i]·2^⌈25.5·i⌉`）。式は [`crate::avx2`] と同じで、1 命令で 8 鍵ぶん進む。
//! 表には隣り合う2リムを 64 ビットの1ワードに入れておき（下位 32 ビットに偶数番目、上位に奇数番目）、
//! `vpgatherqq` 1 命令で8鍵の2リムを読む。

use super::{Avx512, Field8, Table8};
use crate::comb::Extended;
use crate::field::Fe;
use std::arch::x86_64::*;
use std::sync::LazyLock;

const MASK26: u64 = (1 << 26) - 1;
const MASK25: u64 = (1 << 25) - 1;

/// 2p のリム（引き算・符号反転で負にならないように足す）。p の最下位リムは 2^26 - 19、
/// ほかは 2^26 - 1 か 2^25 - 1
const P2: [u64; 10] = {
    let mut p = [0u64; 10];
    let mut i = 0;
    while i < 10 {
        p[i] = 2 * if i % 2 == 0 { MASK26 } else { MASK25 };
        i += 1;
    }
    p[0] -= 2 * 18;
    p
};

/// 51 ビット × 5 の正規化したリムを 25.5 ビット × 10 に分ける
const fn split(l: [u64; 5]) -> [u64; 10] {
    let mut out = [0u64; 10];
    let mut i = 0;
    while i < 5 {
        out[2 * i] = l[i] & MASK26;
        out[2 * i + 1] = l[i] >> 26;
        i += 1;
    }
    out
}

/// 体の元を8つ（25.5 ビット × 10 リム）。
///
/// 各演算が正しく動く入力の条件（リムの上限を、偶数番目 2^26・奇数番目 2^25 の何倍かで書く。
/// tests の proptest で、この境界ちょうどまで検証している）:
/// - `mul(a, b)`: a は 5 倍未満、b は 3.25 倍未満（b のリムを 19 倍しても 32 ビットに収まり、
///   積の和が 64 ビットに収まる）
/// - `sub(a, b)`, `neg_masked`: b（反転する値）は 1.5 倍未満（2p を足してから引くので負にならない）
///
/// `mul` と `carry` の結果は 1.01 倍未満になる。
#[derive(Clone, Copy)]
pub(crate) struct Fe10([__m512i; 10]);

#[inline(always)]
unsafe fn splat10(l: [u64; 10]) -> [__m512i; 10] {
    let mut r = [_mm512_setzero_si512(); 10];
    for i in 0..10 {
        r[i] = _mm512_set1_epi64(l[i] as i64);
    }
    r
}

/// `c[k] += a·bs[k]`（a は k が偶数なら `a_even`、奇数なら `a_odd`）
#[inline(always)]
unsafe fn row(c: &mut [__m512i; 10], a_even: __m512i, a_odd: __m512i, bs: [__m512i; 10]) {
    for k in 0..10usize {
        let a = if k.is_multiple_of(2) { a_even } else { a_odd };
        c[k] = _mm512_add_epi64(c[k], _mm512_mul_epu32(a, bs[k]));
    }
}

/// 各リムを 26/25 ビット + α に収める（ref10 と同じ順に、2本の繰り上がりの列を交互に進める）
#[inline(always)]
unsafe fn carry(mut c: [__m512i; 10]) -> Fe10 {
    // ループにすると展開されず、添字の範囲検査と分岐が残るので書き下す
    carry_step(&mut c, 0);
    carry_step(&mut c, 4);
    carry_step(&mut c, 1);
    carry_step(&mut c, 5);
    carry_step(&mut c, 2);
    carry_step(&mut c, 6);
    carry_step(&mut c, 3);
    carry_step(&mut c, 7);
    carry_step(&mut c, 4);
    carry_step(&mut c, 8);
    carry_step(&mut c, 9);
    carry_step(&mut c, 0);
    Fe10(c)
}

#[inline(always)]
unsafe fn carry_step(c: &mut [__m512i; 10], k: usize) {
    let (hi, lo) = if k.is_multiple_of(2) {
        let mask = _mm512_set1_epi64(MASK26 as i64);
        (_mm512_srli_epi64::<26>(c[k]), _mm512_and_si512(c[k], mask))
    } else {
        let mask = _mm512_set1_epi64(MASK25 as i64);
        (_mm512_srli_epi64::<25>(c[k]), _mm512_and_si512(c[k], mask))
    };
    c[k] = lo;
    if k == 9 {
        // 繰り上がりは 32 ビットを超えうるので、乗算ではなく 19 = 1 + 2 + 16 で
        let t = _mm512_add_epi64(hi, _mm512_slli_epi64::<1>(hi));
        c[0] = _mm512_add_epi64(c[0], _mm512_add_epi64(t, _mm512_slli_epi64::<4>(hi)));
    } else {
        c[k + 1] = _mm512_add_epi64(c[k + 1], hi);
    }
}

impl Field8 for Fe10 {
    /// 下位 32 ビットに偶数番目のリム、上位 32 ビットに奇数番目のリム
    fn pack(l: [u64; 5]) -> [u64; 5] {
        let s = split(l);
        std::array::from_fn(|i| s[2 * i] | s[2 * i + 1] << 32)
    }

    #[inline(always)]
    unsafe fn unpack(w: &[__m512i; 5]) -> Fe10 {
        let lo = _mm512_set1_epi64(0xffff_ffff);
        let mut r = [_mm512_setzero_si512(); 10];
        for i in 0..5 {
            r[2 * i] = _mm512_and_si512(w[i], lo);
            r[2 * i + 1] = _mm512_srli_epi64::<32>(w[i]);
        }
        Fe10(r)
    }

    #[inline(always)]
    unsafe fn splat(l: [u64; 5]) -> Fe10 {
        Fe10(splat10(split(l)))
    }

    #[inline(always)]
    unsafe fn add(&self, b: &Fe10) -> Fe10 {
        let mut r = self.0;
        for i in 0..10 {
            r[i] = _mm512_add_epi64(self.0[i], b.0[i]);
        }
        Fe10(r)
    }

    /// a + 2p - b
    #[inline(always)]
    unsafe fn sub(&self, b: &Fe10) -> Fe10 {
        let p2 = splat10(P2);
        let mut r = self.0;
        for i in 0..10 {
            r[i] = _mm512_sub_epi64(_mm512_add_epi64(self.0[i], p2[i]), b.0[i]);
        }
        Fe10(r)
    }

    #[inline(always)]
    unsafe fn mul(&self, b: &Fe10) -> Fe10 {
        let (a, b) = (&self.0, &b.0);
        // 2^255 = 19 (mod p) なので、桁あふれ分は 19 倍して下に回す。
        // 奇数番目のリムどうしの積は、桁の位置が半ビットずつずれるので 2 倍して合わせる。
        // c[k] = Σ a[i]·b[k-i]（k-i < 0 なら b[k-i+10]·19）。ループだと展開されないので書き下す
        let nineteen = _mm512_set1_epi64(19);
        let mut b19 = [_mm512_setzero_si512(); 10];
        for j in 1..10 {
            b19[j] = _mm512_mul_epu32(b[j], nineteen);
        }
        let mut a2 = [_mm512_setzero_si512(); 10];
        for i in (1..10).step_by(2) {
            a2[i] = _mm512_add_epi64(a[i], a[i]);
        }
        let mut c = [_mm512_setzero_si512(); 10];
        row(
            &mut c,
            a[0],
            a[0],
            [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9]],
        );
        row(
            &mut c,
            a2[1],
            a[1],
            [b19[9], b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8]],
        );
        row(
            &mut c,
            a[2],
            a[2],
            [
                b19[8], b19[9], b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ],
        );
        row(
            &mut c,
            a2[3],
            a[3],
            [
                b19[7], b19[8], b19[9], b[0], b[1], b[2], b[3], b[4], b[5], b[6],
            ],
        );
        row(
            &mut c,
            a[4],
            a[4],
            [
                b19[6], b19[7], b19[8], b19[9], b[0], b[1], b[2], b[3], b[4], b[5],
            ],
        );
        row(
            &mut c,
            a2[5],
            a[5],
            [
                b19[5], b19[6], b19[7], b19[8], b19[9], b[0], b[1], b[2], b[3], b[4],
            ],
        );
        row(
            &mut c,
            a[6],
            a[6],
            [
                b19[4], b19[5], b19[6], b19[7], b19[8], b19[9], b[0], b[1], b[2], b[3],
            ],
        );
        row(
            &mut c,
            a2[7],
            a[7],
            [
                b19[3], b19[4], b19[5], b19[6], b19[7], b19[8], b19[9], b[0], b[1], b[2],
            ],
        );
        row(
            &mut c,
            a[8],
            a[8],
            [
                b19[2], b19[3], b19[4], b19[5], b19[6], b19[7], b19[8], b19[9], b[0], b[1],
            ],
        );
        row(
            &mut c,
            a2[9],
            a[9],
            [
                b19[1], b19[2], b19[3], b19[4], b19[5], b19[6], b19[7], b19[8], b19[9], b[0],
            ],
        );
        carry(c)
    }

    #[inline(always)]
    unsafe fn carry(&self) -> Fe10 {
        carry(self.0)
    }

    #[inline(always)]
    unsafe fn blend(k: __mmask8, a: &Fe10, b: &Fe10) -> Fe10 {
        let mut r = a.0;
        for i in 0..10 {
            r[i] = _mm512_mask_blend_epi64(k, a.0[i], b.0[i]);
        }
        Fe10(r)
    }

    /// k のレーンだけ 2p - x
    #[inline(always)]
    unsafe fn neg_masked(&self, k: __mmask8) -> Fe10 {
        let p2 = splat10(P2);
        let mut r = self.0;
        for i in 0..10 {
            r[i] = _mm512_mask_sub_epi64(self.0[i], k, p2[i], self.0[i]);
        }
        Fe10(r)
    }

    /// 隣り合う2リムをまとめて 51 ビットのリムにする（`l[2i] + l[2i+1]·2^26`）
    fn lanes(&self) -> [Fe; 8] {
        // SAFETY: __m512i と [u64; 8] は同じ大きさで、どんなビット列も有効な値
        let l = self
            .0
            .map(|v| unsafe { std::mem::transmute::<__m512i, [u64; 8]>(v) });
        std::array::from_fn(|k| {
            Fe(std::array::from_fn(|i| {
                l[2 * i][k] + (l[2 * i + 1][k] << 26)
            }))
        })
    }
}

static TABLE: LazyLock<Table8> = LazyLock::new(Table8::build::<Fe10>);

/// 表を作っておく（探索の時間に含めないため）
pub fn prepare(_: Avx512) {
    LazyLock::force(&TABLE);
}

/// [`crate::comb::Table::mul_base_batch`] と同じ計算を、8鍵ずつ AVX-512F で（`N` は 8 の倍数）
pub fn mul_base_batch<const N: usize>(
    avx512: Avx512,
    scalars: &[[u8; 32]; N],
    out: &mut [Extended; N],
) {
    let _ = avx512;
    // SAFETY: Avx512 があるので、CPU は下の target_feature の命令に対応している
    unsafe { run(&TABLE, scalars, out) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
unsafe fn run<const N: usize>(table: &Table8, scalars: &[[u8; 32]; N], out: &mut [Extended; N]) {
    // SAFETY: 呼び出し元が CPU の対応を確かめている
    unsafe { super::mul_base_batch::<Fe10, N>(table, scalars, out) }
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

    fn avx512() -> Option<Avx512> {
        available("avx512", Avx512::detect())
    }

    /// 8レーンぶんのリム（lanes[k][i] = k 番目の元の i 番目のリム）から
    fn fe10(lanes: [[u64; 10]; 8]) -> Fe10 {
        Fe10(std::array::from_fn(|i| {
            let v: [u64; 8] = std::array::from_fn(|k| lanes[k][i]);
            // SAFETY: 同じ大きさで、どんなビット列も有効な値
            unsafe { std::mem::transmute::<[u64; 8], __m512i>(v) }
        }))
    }

    fn raw_lanes(f: &Fe10) -> [[u64; 10]; 8] {
        // SAFETY: 同じ大きさで、どんなビット列も有効な値
        let l =
            f.0.map(|v| unsafe { std::mem::transmute::<__m512i, [u64; 8]>(v) });
        std::array::from_fn(|k| std::array::from_fn(|i| l[i][k]))
    }

    /// リムが表す整数（mod を取らない）
    fn value(l: &[u64; 10]) -> BigInt {
        (0..10).fold(BigInt::from(0), |acc, i| {
            acc + (BigInt::from(l[i]) << (i * 51).div_ceil(2))
        })
    }

    fn modp(x: &BigInt) -> BigInt {
        x.mod_floor(&rfc8032::P)
    }

    fn base(i: usize) -> u64 {
        if i.is_multiple_of(2) {
            1 << 26
        } else {
            1 << 25
        }
    }

    /// 各リムが `base * num / den` 未満の元を8レーン。境界の値（0、最大）を多めに混ぜる
    fn fe10_below(num: u64, den: u64) -> impl Strategy<Value = [[u64; 10]; 8]> {
        any::<[[u64; 10]; 8]>().prop_map(move |r| {
            r.map(|l| {
                std::array::from_fn(|i| {
                    let max = base(i) * num / den - 1;
                    match l[i] % 9 {
                        0 => 0,
                        1 | 2 => max,
                        _ => (l[i] >> 4) % (max + 1),
                    }
                })
            })
        })
    }

    fn below(f: &Fe10, num: u64, den: u64) -> bool {
        raw_lanes(f)
            .iter()
            .all(|l| (0..10).all(|i| l[i] < base(i) * num / den))
    }

    #[derive(Debug, Clone, Copy)]
    enum Arith {
        Add,
        Sub,
        Mul,
        Carry,
        Neg,
    }

    /// 8レーンの体演算が、多倍長整数の計算と一致する（入力は [`Fe10`] に書いた条件ちょうどまで）。
    /// 乗算と繰り上げの結果は 1.01 倍未満。`neg_masked` はマスクの立ったレーンだけ反転する
    #[rstest]
    fn arithmetic_matches_bigint(
        #[values(Arith::Add, Arith::Sub, Arith::Mul, Arith::Carry, Arith::Neg)] op: Arith,
    ) {
        let Some(_) = avx512() else { return };
        // (a の上限, b の上限) を分数で
        let ((an, ad), (bn, bd)) = match op {
            Arith::Add => ((4, 1), (4, 1)),
            Arith::Sub => ((4, 1), (3, 2)),
            Arith::Mul => ((5, 1), (13, 4)),
            Arith::Carry => ((1 << 37, 1), (1, 1)),
            Arith::Neg => ((3, 2), (1, 1)),
        };
        check(
            1024,
            (fe10_below(an, ad), fe10_below(bn, bd), any::<u8>()),
            |(a, b, k)| {
                // SAFETY: avx512() で対応を確かめた
                let got = unsafe {
                    let (x, y) = (fe10(a), fe10(b));
                    match op {
                        Arith::Add => x.add(&y),
                        Arith::Sub => x.sub(&y),
                        Arith::Mul => x.mul(&y),
                        Arith::Carry => Field8::carry(&x),
                        Arith::Neg => x.neg_masked(k),
                    }
                };
                for l in 0..8 {
                    let (va, vb) = (value(&a[l]), value(&b[l]));
                    let g = value(&raw_lanes(&got)[l]);
                    match op {
                        Arith::Add => prop_assert_eq!(g, va + vb),
                        Arith::Sub => prop_assert_eq!(modp(&g), modp(&(va - vb))),
                        Arith::Mul => prop_assert_eq!(modp(&g), modp(&(va * vb))),
                        Arith::Carry => prop_assert_eq!(modp(&g), modp(&va)),
                        Arith::Neg if k >> l & 1 == 1 => prop_assert_eq!(modp(&g), modp(&-va)),
                        Arith::Neg => prop_assert_eq!(raw_lanes(&got)[l], a[l]),
                    }
                }
                if let Arith::Mul | Arith::Carry = op {
                    prop_assert!(below(&got, 101, 100), "1.01 倍以上: {:?}", raw_lanes(&got));
                }
                if let Arith::Neg = op {
                    prop_assert!(
                        below(&got, 2, 1),
                        "反転した値が 2 倍以上: {:?}",
                        raw_lanes(&got)
                    );
                }
                Ok(())
            },
        );
    }
}
