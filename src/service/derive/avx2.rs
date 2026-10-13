//! AVX2 で4鍵の基点倍算を同時に進める（[`crate::service::derive::comb`] の櫛形テーブルと同じ計算を4レーンで）
//!
//! 体の元は 25.5 ビット × 10 リム（偶数番目のリムは 26 ビット、奇数番目は 25 ビット。値は
//! `Σ l[i]·2^⌈25.5·i⌉`）で表し、256 ビットレジスタの 64 ビットのレーン4つに、4鍵の同じリムを入れる。
//! `vpmuludq`（32×32→64 ビットの乗算を4レーン同時）で、4鍵ぶんの乗算を1命令で進める。
//! 隣り合う2リムを `l[2i] + l[2i+1]·2^26` とまとめると [`Fe`] の 51 ビットのリムになるので、
//! 倍算が終わったら [`Extended`] に戻し、圧縮は今までどおり行う。
//!
//! テーブルのアクセス位置が秘密に依存する（定数時間ではない）ので、探索の高速化専用。
//!
//! `#[target_feature]` の付いた関数は `#[inline(always)]` にできず、体演算が関数呼び出しのまま
//! 残って遅くなる。そこで `#[target_feature]` は入口（[`Table4::mul_base_batch`] の中の safe な
//! `#[target_feature] fn`）だけに付け、中で使う関数はすべて `#[inline(always)]` の `unsafe fn` にして
//! 入口に展開させる。これらの `unsafe fn` は、AVX2 に対応した CPU でだけ呼ぶこと。
//! `#[target_feature]` の無い関数の中では AVX2 の命令が unsafe なので、本体を `unsafe` ブロックで囲む
//! （入口の中では命令そのものは safe で、`unsafe` が要るのは `unsafe fn` の呼び出しとポインタの読み書きだけ）。

#![expect(
    clippy::needless_range_loop,
    reason = "リムの添字の計算がそのまま式になるので、添字でループする"
)]

use crate::service::derive::comb::{COLS, Extended, ROWS, Table, signed_digits};
use crate::service::derive::field::Fe;
use std::arch::x86_64::*;
use std::sync::LazyLock;

/// CPU が AVX2 に対応していることの証明（[`Avx2::detect`] でしか作れない）。
/// AVX2 の命令を使う関数はこれを受け取るので、非対応の CPU で呼ばれることはない
#[derive(Clone, Copy, Debug)]
pub struct Avx2(());

impl Avx2 {
    pub fn detect() -> Option<Avx2> {
        std::arch::is_x86_feature_detected!("avx2").then_some(Avx2(()))
    }
}

const MASK26: u64 = (1 << 26) - 1;
const MASK25: u64 = (1 << 25) - 1;

/// 2p のリム（引き算で負にならないように足す）。p の最下位リムは 2^26 - 19、ほかは 2^26 - 1 か 2^25 - 1
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

/// 体の元を4つ（レーン k が k 番目の鍵）。
///
/// 各演算が正しく動く入力の条件（リムの上限を、偶数番目 2^26・奇数番目 2^25 の何倍かで書く。
/// tests の proptest で、この境界ちょうどまで検証している）:
/// - `mul(a, b)`: a は 5 倍未満、b は 3.25 倍未満（b のリムを 19 倍しても 32 ビットに収まり、
///   積の和が 64 ビットに収まる）
/// - `sub(a, b)`: b は 1.5 倍未満（2p を足してから引くので負にならない）
///
/// `mul` と `carry` の結果は 1.01 倍未満になる。
#[derive(Clone, Copy)]
struct Fe4([__m256i; 10]);

/// 拡張座標 (X:Y:Z:T) の点を4つ
#[derive(Clone, Copy)]
struct Ext4 {
    x: Fe4,
    y: Fe4,
    z: Fe4,
    t: Fe4,
}

/// 加算用に前計算したアフィン点 (y+x, y-x, 2dxy) を4つ
#[derive(Clone, Copy)]
struct Niels4 {
    ypx: Fe4,
    ymx: Fe4,
    xy2d: Fe4,
}

/// 全レーンに同じ値
#[inline(always)]
unsafe fn splat(l: [u64; 10]) -> Fe4 {
    unsafe {
        let mut r = [_mm256_setzero_si256(); 10];
        for i in 0..10 {
            r[i] = _mm256_set1_epi64x(l[i].cast_signed());
        }
        Fe4(r)
    }
}

#[inline(always)]
unsafe fn add(a: &Fe4, b: &Fe4) -> Fe4 {
    unsafe {
        let mut r = a.0;
        for i in 0..10 {
            r[i] = _mm256_add_epi64(a.0[i], b.0[i]);
        }
        Fe4(r)
    }
}

/// a + 2p - b
#[inline(always)]
unsafe fn sub(a: &Fe4, b: &Fe4) -> Fe4 {
    unsafe {
        let p2 = splat(P2);
        let mut r = a.0;
        for i in 0..10 {
            r[i] = _mm256_sub_epi64(_mm256_add_epi64(a.0[i], p2.0[i]), b.0[i]);
        }
        Fe4(r)
    }
}

/// `mask` のレーンは a、それ以外は b
#[inline(always)]
unsafe fn select(mask: __m256i, a: &Fe4, b: &Fe4) -> Fe4 {
    unsafe {
        let mut r = a.0;
        for i in 0..10 {
            r[i] = _mm256_blendv_epi8(b.0[i], a.0[i], mask);
        }
        Fe4(r)
    }
}

/// `c[k] += a·bs[k]`（a は k が偶数なら `a_even`、奇数なら `a_odd`）
#[inline(always)]
unsafe fn row(c: &mut [__m256i; 10], a_even: __m256i, a_odd: __m256i, bs: [__m256i; 10]) {
    unsafe {
        for k in 0..10usize {
            let a = if k.is_multiple_of(2) { a_even } else { a_odd };
            c[k] = _mm256_add_epi64(c[k], _mm256_mul_epu32(a, bs[k]));
        }
    }
}

#[inline(always)]
unsafe fn mul(a: &Fe4, b: &Fe4) -> Fe4 {
    unsafe {
        let (a, b) = (&a.0, &b.0);
        // 2^255 = 19 (mod p) なので、桁あふれ分は 19 倍して下に回す。
        // 奇数番目のリムどうしの積は、桁の位置が半ビットずつずれるので 2 倍して合わせる。
        // c[k] = Σ a[i]·b[k-i]（k-i < 0 なら b[k-i+10]·19）。ループだと展開されないので書き下す
        let nineteen = _mm256_set1_epi64x(19);
        let mut b19 = [_mm256_setzero_si256(); 10];
        for j in 1..10 {
            b19[j] = _mm256_mul_epu32(b[j], nineteen);
        }
        let mut a2 = [_mm256_setzero_si256(); 10];
        for i in (1..10).step_by(2) {
            a2[i] = _mm256_add_epi64(a[i], a[i]);
        }
        let mut c = [_mm256_setzero_si256(); 10];
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
}

/// 各リムを 26/25 ビット + α に収める（ref10 と同じ順に、2本の繰り上がりの列を交互に進める）
#[inline(always)]
unsafe fn carry(mut c: [__m256i; 10]) -> Fe4 {
    unsafe {
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
        Fe4(c)
    }
}

#[inline(always)]
unsafe fn carry_step(c: &mut [__m256i; 10], k: usize) {
    unsafe {
        let (hi, lo) = if k.is_multiple_of(2) {
            let mask = _mm256_set1_epi64x(MASK26.cast_signed());
            (_mm256_srli_epi64::<26>(c[k]), _mm256_and_si256(c[k], mask))
        } else {
            let mask = _mm256_set1_epi64x(MASK25.cast_signed());
            (_mm256_srli_epi64::<25>(c[k]), _mm256_and_si256(c[k], mask))
        };
        c[k] = lo;
        if k == 9 {
            // 繰り上がりは 32 ビットを超えうるので、乗算ではなく 19 = 1 + 2 + 16 で
            let t = _mm256_add_epi64(hi, _mm256_slli_epi64::<1>(hi));
            c[0] = _mm256_add_epi64(c[0], _mm256_add_epi64(t, _mm256_slli_epi64::<4>(hi)));
        } else {
            c[k + 1] = _mm256_add_epi64(c[k + 1], hi);
        }
    }
}

/// 拡張座標に（[`crate::service::derive::comb`] の `Niels::to_extended` と同じ。X, Y は次の加算に渡すため繰り上げておく）
#[inline(always)]
unsafe fn to_extended(n: &Niels4) -> Ext4 {
    unsafe {
        const TWO: [u64; 10] = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        // 1/2
        const INV2: [u64; 10] = split([
            2251799813685239,
            2251799813685247,
            2251799813685247,
            2251799813685247,
            1125899906842623,
        ]);
        let x = carry(sub(&n.ypx, &n.ymx).0);
        let y = carry(add(&n.ypx, &n.ymx).0);
        let t = mul(&mul(&x, &y), &splat(INV2));
        Ext4 {
            x,
            y,
            z: splat(TWO),
            t,
        }
    }
}

/// 拡張座標 + アフィン点（[`crate::service::derive::comb`] の `add_niels_impl` と同じ式）。
/// 入力の X, Y, Z, T は 1.01 倍未満（`mul` か `carry` の結果）、`q` は正規化した値か、その符号反転
#[inline(always)]
unsafe fn add_niels<const WITH_T: bool>(p: &Ext4, q: &Niels4) -> Ext4 {
    unsafe {
        let pp = mul(&add(&p.y, &p.x), &q.ypx); // 2.02 倍 × 1 倍
        let mm = mul(&sub(&p.y, &p.x), &q.ymx); // 3.01 倍 × 1 倍
        let txy = mul(&p.t, &q.xy2d); // 1.01 倍 × 2 倍
        let z2 = add(&p.z, &p.z);
        let (x, y) = (sub(&pp, &mm), add(&pp, &mm)); // 3.01 倍, 2.02 倍
        let (z, t) = (add(&z2, &txy), sub(&z2, &txy)); // 3.03 倍, 4.02 倍
        Ext4 {
            x: mul(&t, &x),
            y: mul(&y, &z),
            z: mul(&t, &z),
            t: if WITH_T { mul(&x, &y) } else { p.t },
        }
    }
}

/// 4レーンを [`Fe`] 4つに（隣り合う2リムをまとめて 51 ビットのリムにする）
fn lanes(f: &Fe4) -> [Fe; 4] {
    // SAFETY: __m256i と [u64; 4] は同じ大きさで、どんなビット列も有効な値
    let l =
        f.0.map(|v| unsafe { std::mem::transmute::<__m256i, [u64; 4]>(v) });
    std::array::from_fn(|k| {
        Fe(std::array::from_fn(|i| {
            l[2 * i][k] + (l[2 * i + 1][k] << 26)
        }))
    })
}

/// テーブルの1エントリ: y+x, y-x, 2dxy の 25.5 ビットのリム（各10個）と未使用の2個。
/// 16 バイトずつ読むので、64 バイト境界にそろえた 128 バイトにしている
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Entry([u32; 32]);

/// [`Table`] と同じ値を 25.5 ビットのリムで持つテーブル（W = 12 で約 5.8MB）
pub struct Table4(Box<[Entry]>);

impl Table4 {
    /// プロセスで1つだけ作って共有する
    pub fn get(_: Avx2) -> &'static Table4 {
        static TABLE: LazyLock<Table4> = LazyLock::new(Table4::build);
        &TABLE
    }

    fn build() -> Table4 {
        // 51 ビット版は変換にだけ使って捨てる（AVX2 を使うときは要らないので、共有のものは作らない）
        let table = Table::build();
        let mut entries = vec![Entry([0; 32]); ROWS * COLS];
        for (n, e) in entries.iter_mut().enumerate() {
            for (f, fe) in table.entry(n / COLS, n % COLS).into_iter().enumerate() {
                for (dst, src) in e.0[10 * f..10 * f + 10].iter_mut().zip(split(fe.reduce())) {
                    *dst = src as u32;
                }
            }
        }
        Table4(entries.into_boxed_slice())
    }

    fn row(&self, i: usize) -> &[Entry; COLS] {
        &self.0.as_chunks::<COLS>().0[i]
    }

    /// [`Table::mul_base_batch`] と同じ計算を、4鍵ずつ AVX2 で（`N` は 4 の倍数）
    pub fn mul_base_batch<const N: usize>(
        &self,
        avx2: Avx2,
        scalars: &[[u8; 32]; N],
        out: &mut [Extended; N],
    ) {
        let _ = avx2;
        // SAFETY: Avx2 があるので、CPU は AVX2 に対応している
        unsafe { self.mul_base_batch_avx2(scalars, out) }
    }

    #[target_feature(enable = "avx2")]
    fn mul_base_batch_avx2<const N: usize>(
        &self,
        scalars: &[[u8; 32]; N],
        out: &mut [Extended; N],
    ) {
        for (p, o) in self
            .points_avx2(scalars)
            .iter()
            .zip(out.as_chunks_mut::<4>().0)
        {
            let (x, y, z) = (lanes(&p.x), lanes(&p.y), lanes(&p.z));
            for k in 0..4 {
                o[k] = Extended {
                    x: x[k],
                    y: y[k],
                    z: z[k],
                    t: Fe::ZERO,
                };
            }
        }
    }

    /// 4鍵ずつの組の点（最後の加算まで。T は計算しない）
    #[target_feature(enable = "avx2")]
    fn points_avx2<const N: usize>(&self, scalars: &[[u8; 32]; N]) -> Vec<Ext4> {
        const { assert!(N.is_multiple_of(4)) };
        // digits[i][k] = k 番目の鍵の i 桁目
        let mut digits = [[0i16; N]; ROWS];
        for (k, s) in scalars.iter().enumerate() {
            for (i, d) in signed_digits(s).into_iter().enumerate() {
                digits[i][k] = d;
            }
        }
        // SAFETY: この関数は AVX2 を有効にしてコンパイルされ、AVX2 に対応した CPU でだけ呼ばれる
        unsafe {
            // 次の4鍵が使う表の値を先に読み始めておく。表は L2 に収まらず、1回の加算は数百サイクル
            // かかるので、1つ先を読んでおけば待ち時間を計算に重ねられる
            let prefetch_next = |i: usize, g: usize| {
                let (i, g) = if g + 1 < N / 4 {
                    (i, g + 1)
                } else {
                    (i + 1, 0)
                };
                if i < ROWS {
                    prefetch(self.row(i), &digits[i].as_chunks::<4>().0[g]);
                }
            };
            // 1行目は単位元への加算なので、表の値をそのまま座標にする
            let mut acc: Vec<Ext4> = Vec::with_capacity(N / 4);
            let row = self.row(0);
            for (g, d) in digits[0].as_chunks::<4>().0.iter().enumerate() {
                prefetch_next(0, g);
                acc.push(to_extended(&gather(row, d)));
            }
            // 桁が 0 のレーンは単位元（表の0列目）を足す。4レーンまとめて計算するので飛ばさない
            for (i, ds) in digits.iter().enumerate().take(ROWS - 1).skip(1) {
                let row = self.row(i);
                for (g, (a, d)) in acc.iter_mut().zip(ds.as_chunks::<4>().0).enumerate() {
                    prefetch_next(i, g);
                    *a = add_niels::<true>(a, &gather(row, d));
                }
            }
            let row = self.row(ROWS - 1);
            let last = digits[ROWS - 1].as_chunks::<4>().0;
            for (g, (a, d)) in acc.iter_mut().zip(last).enumerate() {
                prefetch_next(ROWS - 1, g);
                *a = add_niels::<false>(a, &gather(row, d));
            }
            acc
        }
    }
}

/// 4鍵ぶんの表の値 `d[k] * 2^(W*i) * B` を引き、レーンにそろえる（負の桁は符号を反転する）
#[inline(always)]
unsafe fn gather(row: &[Entry; COLS], d: &[i16; 4]) -> Niels4 {
    unsafe {
        let e = d.map(|x| &row[usize::from(x.unsigned_abs())].0);
        // 各エントリの4リムずつを 64 ビットに広げて読み、4×4 を転置してレーンにそろえる
        let mut l = [_mm256_setzero_si256(); 32];
        for q in 0..8 {
            let v = [
                load4(e[0], q),
                load4(e[1], q),
                load4(e[2], q),
                load4(e[3], q),
            ];
            let (t0, t1) = (
                _mm256_unpacklo_epi64(v[0], v[1]),
                _mm256_unpackhi_epi64(v[0], v[1]),
            );
            let (t2, t3) = (
                _mm256_unpacklo_epi64(v[2], v[3]),
                _mm256_unpackhi_epi64(v[2], v[3]),
            );
            l[4 * q] = _mm256_permute2x128_si256::<0x20>(t0, t2);
            l[4 * q + 1] = _mm256_permute2x128_si256::<0x20>(t1, t3);
            l[4 * q + 2] = _mm256_permute2x128_si256::<0x31>(t0, t2);
            l[4 * q + 3] = _mm256_permute2x128_si256::<0x31>(t1, t3);
        }
        let fe = |r: usize| Fe4(*l[r..].first_chunk().expect("32 リムのうち 10 個"));
        let (ypx, ymx, xy2d) = (fe(0), fe(10), fe(20));
        // -P = (-x, y): y+x と y-x を入れ替え、2dxy を反転する
        let d = d.map(i64::from);
        let neg = _mm256_cmpgt_epi64(
            _mm256_setzero_si256(),
            _mm256_set_epi64x(d[3], d[2], d[1], d[0]),
        );
        let zero = splat([0; 10]);
        Niels4 {
            ypx: select(neg, &ymx, &ypx),
            ymx: select(neg, &ypx, &ymx),
            xy2d: select(neg, &sub(&zero, &xy2d), &xy2d),
        }
    }
}

/// 4鍵ぶんの表の値（1エントリ = 64 バイトの行2つ）を L1 に読み込み始める
#[inline(always)]
unsafe fn prefetch(row: &[Entry; COLS], d: &[i16; 4]) {
    unsafe {
        for &x in d {
            let p: *const i8 = (&raw const row[usize::from(x.unsigned_abs())]).cast();
            _mm_prefetch::<_MM_HINT_T0>(p);
            _mm_prefetch::<_MM_HINT_T0>(p.wrapping_add(64));
        }
    }
}

/// エントリの `4q`〜`4q+3` 番目のリムを 64 ビットのレーンに広げて読む
#[inline(always)]
unsafe fn load4(e: &[u32; 32], q: usize) -> __m256i {
    unsafe {
        let s: &[u32; 4] = e[4 * q..].first_chunk().expect("エントリは 32 リム");
        // s は 16 バイトの有効な参照（_mm_loadu_si128 は境界をそろえなくてよい）
        _mm256_cvtepu32_epi64(_mm_loadu_si128(s.as_ptr().cast()))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::check;
    use crate::rfc8032;
    use crate::rfc8032::secret_expand;
    use crate::rfc8032::to_le32;
    use crate::rfc8032::{Vector, vectors};
    use crate::service::derive::comb::compress_with_zinv;
    use crate::simd_support::available;
    use curve25519_dalek::edwards::EdwardsPoint;
    use curve25519_dalek::scalar::clamp_integer;
    use num_bigint::BigInt;
    use num_integer::Integer;
    use proptest::collection::vec;
    use proptest::option::weighted;
    use proptest::prelude::*;
    use rstest::{fixture, rstest};
    use similar_asserts::assert_eq;
    use std::array::from_fn;
    use std::mem::transmute;

    /// このマシンで AVX2 が使えなければテストを飛ばす（MEIKEY_REQUIRE_SIMD に avx2 があれば失敗させる）
    #[fixture]
    fn avx2() -> Option<Avx2> {
        available("avx2", Avx2::detect())
    }

    /// 4レーンぶんのリム（lanes[k][i] = k 番目の元の i 番目のリム）から
    fn fe4(lanes: [[u64; 10]; 4]) -> Fe4 {
        Fe4(from_fn(|i| {
            let v = [lanes[0][i], lanes[1][i], lanes[2][i], lanes[3][i]];
            // SAFETY: 同じ大きさで、どんなビット列も有効な値
            unsafe { transmute::<[u64; 4], __m256i>(v) }
        }))
    }

    fn raw_lanes(f: &Fe4) -> [[u64; 10]; 4] {
        // SAFETY: 同じ大きさで、どんなビット列も有効な値
        let l = f.0.map(|v| unsafe { transmute::<__m256i, [u64; 4]>(v) });
        from_fn(|k| from_fn(|i| l[i][k]))
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

    /// 各リムが `base * num / den` 未満の元を4レーン。境界の値（0、最大）を多めに混ぜる
    fn fe4_below(num: u64, den: u64) -> impl Strategy<Value = [[u64; 10]; 4]> {
        any::<[[u64; 10]; 4]>().prop_map(move |r| {
            r.map(|l| {
                from_fn(|i| {
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

    fn below(f: &Fe4, num: u64, den: u64) -> bool {
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
    }

    /// 4レーンの体演算が、多倍長整数の計算と一致する（入力は `Fe4` に書いた条件ちょうどまで）。
    /// 乗算と繰り上げの結果は 1.01 倍未満
    #[rstest]
    fn arithmetic_matches_bigint(
        #[values(Arith::Add, Arith::Sub, Arith::Mul, Arith::Carry)] op: Arith,
        avx2: Option<Avx2>,
    ) {
        let Some(_) = avx2 else { return };
        // (a の上限, b の上限) を分数で
        let ((an, ad), (bn, bd)) = match op {
            Arith::Add => ((4, 1), (4, 1)),
            Arith::Sub => ((4, 1), (3, 2)),
            Arith::Mul => ((5, 1), (13, 4)),
            Arith::Carry => ((1 << 37, 1), (1, 1)),
        };
        check(1024, (fe4_below(an, ad), fe4_below(bn, bd)), |(a, b)| {
            // SAFETY: avx2() で対応を確かめた
            let got = unsafe {
                let (x, y) = (fe4(a), fe4(b));
                match op {
                    Arith::Add => add(&x, &y),
                    Arith::Sub => sub(&x, &y),
                    Arith::Mul => mul(&x, &y),
                    Arith::Carry => carry(x.0),
                }
            };
            for k in 0..4 {
                let (va, vb) = (value(&a[k]), value(&b[k]));
                let g = value(&raw_lanes(&got)[k]);
                match op {
                    Arith::Add => prop_assert_eq!(g, va + vb),
                    Arith::Sub => prop_assert_eq!(modp(&g), modp(&(va - vb))),
                    Arith::Mul => prop_assert_eq!(modp(&g), modp(&(va * vb))),
                    Arith::Carry => prop_assert_eq!(modp(&g), modp(&va)),
                }
            }
            if let Arith::Mul | Arith::Carry = op {
                prop_assert!(below(&got, 101, 100), "1.01 倍以上: {:?}", raw_lanes(&got));
            }
            Ok(())
        });
    }

    /// [`lanes`] は隣り合う2リムをまとめて、同じ値の 51 ビットのリムにする
    #[rstest]
    fn lanes_keep_value() {
        check(256, fe4_below(2, 1), |a| {
            for (k, fe) in lanes(&fe4(a)).into_iter().enumerate() {
                let v =
                    fe.0.iter()
                        .rev()
                        .fold(BigInt::from(0), |acc, &l| (acc << 51) + l);
                prop_assert_eq!(v, value(&a[k]));
            }
            Ok(())
        });
    }

    fn compress(p: &Extended) -> [u8; 32] {
        compress_with_zinv(p, p.z.invert())
    }

    fn dalek(s: &[u8; 32]) -> [u8; 32] {
        EdwardsPoint::mul_base(&curve25519_dalek::Scalar::from_bytes_mod_order(*s))
            .compress()
            .to_bytes()
    }

    /// 8鍵（4レーン × 2組）をまとめて倍算すると、どの位置の鍵も curve25519-dalek と一致する。
    /// 桁が 0（m = 0）・負（符号の反転）・最大の鍵を混ぜる（どれもクランプ済み）
    #[rstest]
    fn mul_base_batch_matches_dalek(avx2: Option<Avx2>) {
        let Some(avx2) = avx2 else { return };
        let special = [
            {
                let mut s = [0u8; 32];
                s[31] = 0x40; // 2^254: m = 0 で、桁がすべて 0
                s
            },
            clamp_integer([0xff; 32]), // 全桁で繰り上がる
            clamp_integer([0x88; 32]), // 桁が正・負に入り混じる
            clamp_integer([0x77; 32]),
        ];
        let strategy = (
            vec(any::<[u8; 32]>(), 8),
            vec(weighted(0.25, 0..special.len()), 8),
        );
        check(64, strategy, |(random, pick)| {
            let s: [[u8; 32]; 8] = from_fn(|k| match pick[k] {
                Some(i) => special[i],
                None => clamp_integer(random[k]),
            });
            let mut out = [Extended::IDENTITY; 8];
            Table4::get(avx2).mul_base_batch(avx2, &s, &mut out);
            for (k, (p, s)) in out.iter().zip(&s).enumerate() {
                prop_assert_eq!(compress(p), dalek(s), "#{}", k);
            }
            Ok(())
        });
    }

    /// RFC 8032 Section 7.1 のテストベクタの秘密スカラーで、参照実装と一致する
    #[rstest]
    fn mul_base_batch_matches_rfc_vectors(avx2: Option<Avx2>, vectors: Vec<Vector>) {
        let Some(avx2) = avx2 else { return };
        let v = vectors;
        let s: [[u8; 32]; 4] = from_fn(|k| to_le32(&secret_expand(&v[k].secret).0));
        let mut out = [Extended::IDENTITY; 4];
        Table4::get(avx2).mul_base_batch(avx2, &s, &mut out);
        for (k, p) in out.iter().enumerate() {
            assert_eq!(compress(p), v[k].public, "TEST {}", v[k].name);
        }
    }
}
