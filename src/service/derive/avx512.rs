//! AVX-512 で8鍵の基点倍算を同時に進める（[`crate::service::derive::comb`] の櫛形テーブルと同じ計算を8レーンで）
//!
//! 512 ビットレジスタの 64 ビットのレーン k に、k 番目の鍵の同じリムを入れる（8鍵を縦に並べる）。
//! 体演算は、CPU が持っている乗算命令に合わせて2通り用意している。
//!
//! | | 対象の CPU | 乗算命令 | リム | 1回の乗算 |
//! |---|---|---|---|---|
//! | [`ifma`] | AVX-512 IFMA あり（Intel は Ice Lake・Sapphire Rapids 以降、AMD は Zen 4 以降） | `vpmadd52luq` / `vpmadd52huq`（52×52→104 ビットの積和） | 51 ビット × 5 | 積和 50 命令 |
//! | [`f`] | AVX-512F まで（Skylake-SP・Cascade Lake） | `vpmuludq`（32×32→64 ビット） | 25.5 ビット × 10 | 乗算 109 命令 |
//!
//! IFMA が AVX-512 で鍵の計算が速くなる本命で、52 ビットの積を1命令で作れるのでリムが半分になり、
//! 乗算の命令数が約4分の1になる。IFMA の無い CPU では、乗算命令が AVX2 と同じ `vpmuludq` なので
//! リムの切り方も同じ（[`crate::service::derive::avx2`] と同じ式）だが、表引き・符号反転・レジスタの使い方は AVX-512 の
//! 命令で書いている。
//!
//! 体演算以外（このファイル）は2通りで共通:
//! - 表: 1エントリ = 64 ビット × 16 ワード（128 バイト、キャッシュライン2本）。y+x, y-x, 2dxy を
//!   5 ワードずつ（[`Field8::pack`] の形）並べ、最後の1ワードは空き
//! - 表引き: `vpgatherqq` で8鍵ぶんの同じワードを1命令で集める（1エントリ 15 命令）
//! - 符号反転: 桁の符号ビットをそのままマスクレジスタ（k）にして、負の桁のレーンだけ y+x と y-x を
//!   入れ替え（`vpblendmq`）、2dxy を反転する（マスク付きの `vpsubq`）
//!
//! `#[target_feature]` の付いた関数は `#[inline(always)]` にできず、体演算が関数呼び出しのまま
//! 残って遅くなる。そこで `#[target_feature]` は入口（[`ifma::mul_base_batch`]・[`f::mul_base_batch`]
//! の中の safe な `#[target_feature] fn`）だけに付け、中で使う関数はすべて `#[inline(always)]` の
//! `unsafe fn` にして入口に展開させる。これらの `unsafe fn` は、[`Avx512::detect`]（IFMA の命令は
//! [`Ifma::detect`]）が通った CPU でだけ呼ぶこと。`#[target_feature]` の無い関数の中では AVX-512 の
//! 命令が unsafe なので、本体を `unsafe` ブロックで囲む。

#![expect(
    clippy::needless_range_loop,
    reason = "リムの添字の計算がそのまま式になるので、添字でループする"
)]

pub mod f;
pub mod ifma;

use crate::service::derive::comb::{COLS, Extended, ROWS, Table, signed_digits};
use crate::service::derive::field::Fe;
use std::arch::x86_64::*;

/// CPU が AVX-512 の F・BW・DQ・VL（x86-64-v4。Skylake-SP 以降の AVX-512 対応 CPU はすべて持っている）
/// に対応していることの証明（[`Avx512::detect`] でしか作れない）。
/// AVX-512 の命令を使う関数はこれを受け取るので、非対応の CPU で呼ばれることはない
#[derive(Clone, Copy, Debug)]
pub struct Avx512(());

impl Avx512 {
    /// OS が 512 ビットレジスタを保存・復元するか（XCR0）も含めて確かめる
    pub fn detect() -> Option<Avx512> {
        (is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl"))
        .then_some(Avx512(()))
    }
}

/// [`Avx512`] に加えて AVX-512 IFMA（52 ビットの積和）にも対応していることの証明
#[derive(Clone, Copy, Debug)]
pub struct Ifma(());

impl Ifma {
    pub fn detect() -> Option<Ifma> {
        Avx512::detect()
            .filter(|_| is_x86_feature_detected!("avx512ifma"))
            .map(|_| Ifma(()))
    }

    /// IFMA に対応していれば AVX-512 の基本命令にも対応している（SHA-512 は基本命令で計算する）
    pub fn avx512(self) -> Avx512 {
        Avx512(())
    }
}

/// 8鍵ぶんの体の元（レーン k が k 番目の鍵）。[`ifma`] と [`f`] で、リムの切り方と演算を差し替える。
///
/// どの演算がどんな入力を受け付けるかは実装ごとに違う（各実装のドキュメントを参照）。
/// 点の演算（[`to_extended`]・[`add_niels`]）は、両方の条件を満たす順番で呼んでいる。
pub(crate) trait Field8: Copy {
    /// 正規化した 51 ビット × 5 リムを、表に入れる 64 ビット × 5 ワードに
    fn pack(l: [u64; 5]) -> [u64; 5];
    /// 表から集めた 5 ワード（各レーンが [`Field8::pack`] の結果）から
    unsafe fn unpack(w: &[__m512i; 5]) -> Self;
    /// 全レーンに同じ値（正規化した 51 ビット × 5 リム）
    unsafe fn splat(l: [u64; 5]) -> Self;
    unsafe fn add(&self, b: &Self) -> Self;
    unsafe fn sub(&self, b: &Self) -> Self;
    unsafe fn mul(&self, b: &Self) -> Self;
    /// リムを小さくする（値は同じ）。次の `add`・`sub` に渡せる大きさにする
    unsafe fn carry(&self) -> Self;
    /// マスク `k` のビットが立ったレーンは `b`、ほかは `a`
    unsafe fn blend(k: __mmask8, a: &Self, b: &Self) -> Self;
    /// マスク `k` のビットが立ったレーンだけ符号を反転する（表の値＝正規化した値に使う）
    unsafe fn neg_masked(&self, k: __mmask8) -> Self;
    /// レーンごとの [`Fe`] に（同じ値。リムは 2^54 未満なので [`Fe`] の乗算にそのまま渡せる）
    fn lanes(&self) -> [Fe; 8];
}

/// 拡張座標 (X:Y:Z:T) の点を8つ
#[derive(Clone, Copy)]
pub(crate) struct Ext8<F> {
    x: F,
    y: F,
    z: F,
    t: F,
}

/// 加算用に前計算したアフィン点 (y+x, y-x, 2dxy) を8つ
#[derive(Clone, Copy)]
pub(crate) struct Niels8<F> {
    ypx: F,
    ymx: F,
    xy2d: F,
}

/// 拡張座標に（[`crate::service::derive::comb`] の `Niels::to_extended` と同じ: X = 2x, Y = 2y, Z = 2, T = XY/Z）。
/// X, Y は次の加算で足し引きするので繰り上げておく
#[inline(always)]
unsafe fn to_extended<F: Field8>(n: &Niels8<F>) -> Ext8<F> {
    unsafe {
        const TWO: [u64; 5] = [2, 0, 0, 0, 0];
        // 1/2
        const INV2: [u64; 5] = [
            2251799813685239,
            2251799813685247,
            2251799813685247,
            2251799813685247,
            1125899906842623,
        ];
        let x = n.ypx.sub(&n.ymx).carry();
        let y = n.ypx.add(&n.ymx).carry();
        Ext8 {
            x,
            y,
            z: F::splat(TWO),
            t: x.mul(&y).mul(&F::splat(INV2)),
        }
    }
}

/// 拡張座標 + アフィン点（[`crate::service::derive::comb`] の `add_niels_impl` と同じ式）。`WITH_T` が false なら
/// T を計算しない（最後の加算用。結果の T は使えない）。
///
/// 右の数字は [`f`] での各リムの大きさ（[`f::Fe10`] を参照）。`mul(a, b)` は a が 5 倍未満、
/// b が 3.25 倍未満でなければならないので、大きいほうを左に置いている。[`ifma`] は
/// `add`・`sub`・`mul` の結果が常に繰り上げ済みなので、順番によらない
///
/// 結果は `p` に上書きする（点8つは [`f`] で 2.5KB あり、値で返すと memcpy の呼び出しになるため）
#[inline(always)]
unsafe fn add_niels<F: Field8, const WITH_T: bool>(p: &mut Ext8<F>, q: &Niels8<F>) {
    unsafe {
        let pp = p.y.add(&p.x).mul(&q.ypx); // 2.02 倍 × 1 倍
        let mm = p.y.sub(&p.x).mul(&q.ymx); // 3.01 倍 × 1 倍
        let txy = p.t.mul(&q.xy2d); // 1.01 倍 × 2 倍（符号反転した値）
        let z2 = p.z.add(&p.z); // 2.02 倍
        let (x, y) = (pp.sub(&mm), pp.add(&mm)); // 3.01 倍, 2.02 倍
        let (z, t) = (z2.add(&txy), z2.sub(&txy)); // 3.03 倍, 4.02 倍
        p.x = t.mul(&x);
        p.y = y.mul(&z);
        p.z = t.mul(&z);
        if WITH_T {
            p.t = x.mul(&y);
        }
    }
}

/// 表の1エントリ: y+x, y-x, 2dxy を [`Field8::pack`] した 5 ワードずつと、空きの1ワード。
/// 128 バイト（キャッシュライン2本）にそろえる
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(crate) struct Entry([u64; 16]);

/// [`Table`] と同じ値を [`Field8::pack`] した形で持つテーブル（W = 12 で約 5.8MB）
pub(crate) struct Table8(Box<[Entry]>);

impl Table8 {
    pub(crate) fn build<F: Field8>() -> Table8 {
        // 51 ビット版は変換にだけ使って捨てる（AVX-512 を使うときは要らないので、共有のものは作らない）
        let table = Table::build();
        let mut entries = vec![Entry([0; 16]); ROWS * COLS];
        for (n, e) in entries.iter_mut().enumerate() {
            for (f, fe) in table.entry(n / COLS, n % COLS).into_iter().enumerate() {
                e.0[5 * f..5 * f + 5].copy_from_slice(&F::pack(fe.reduce()));
            }
        }
        Table8(entries.into_boxed_slice())
    }

    #[inline(always)]
    fn row(&self, i: usize) -> &[Entry; COLS] {
        &self.0.as_chunks::<COLS>().0[i]
    }
}

/// 8鍵ぶんの表の値 `d[k] * 2^(W*i) * B` を集め、レーンにそろえる（負の桁は符号を反転する）
#[inline(always)]
unsafe fn gather<F: Field8>(row: &[Entry; COLS], d: &[i16; 8]) -> Niels8<F> {
    unsafe {
        let d = _mm512_cvtepi16_epi64(_mm_loadu_si128(d.as_ptr().cast()));
        // 桁の符号ビットがそのまま「負のレーン」のマスクになる
        let neg = _mm512_cmplt_epi64_mask(d, _mm512_setzero_si512());
        // エントリの番号 × 16 ワード（gather の位置は 8 バイト単位）。|d| ≤ 2^(W-1) < COLS なので行の中
        let idx = _mm512_slli_epi64::<4>(_mm512_abs_epi64(d));
        let base: *const i64 = row.as_ptr().cast();
        let mut w = [_mm512_setzero_si512(); 15];
        for q in 0..15 {
            w[q] = _mm512_i64gather_epi64::<8>(idx, base.add(q));
        }
        let field = |r: usize| F::unpack(w[r..].first_chunk().expect("15 ワードのうち 5 個"));
        let (ypx, ymx, xy2d) = (field(0), field(5), field(10));
        // -P = (-x, y): y+x と y-x を入れ替え、2dxy を反転する
        Niels8 {
            ypx: F::blend(neg, &ypx, &ymx),
            ymx: F::blend(neg, &ymx, &ypx),
            xy2d: xy2d.neg_masked(neg),
        }
    }
}

/// 8鍵ぶんの表の値（1エントリ = キャッシュライン2本）を L1 に読み込み始める
#[inline(always)]
unsafe fn prefetch(row: &[Entry; COLS], d: &[i16; 8]) {
    unsafe {
        for &x in d {
            let p: *const i8 = (&raw const row[usize::from(x.unsigned_abs())]).cast();
            _mm_prefetch::<_MM_HINT_T0>(p);
            _mm_prefetch::<_MM_HINT_T0>(p.wrapping_add(64));
        }
    }
}

/// `i` 行目の `g` 番目の8鍵の次（同じ行の次の8鍵、行の最後なら次の行の最初の8鍵）が使う表の値を
/// 読み始める。表は L2 に収まらず、1回の加算は数百サイクルかかるので、1つ先を読んでおけば
/// 待ち時間を計算に重ねられる
#[inline(always)]
unsafe fn prefetch_next<const N: usize>(
    table: &Table8,
    digits: &[[i16; N]; ROWS],
    i: usize,
    g: usize,
) {
    unsafe {
        let (i, g) = if g + 1 < N / 8 {
            (i, g + 1)
        } else {
            (i + 1, 0)
        };
        if i < ROWS {
            prefetch(table.row(i), &digits[i].as_chunks::<8>().0[g]);
        }
    }
}

/// [`Table::mul_base_batch`] と同じ計算を8鍵ずつ（`N` は 8 の倍数）
#[inline(always)]
unsafe fn mul_base_batch<F: Field8, const N: usize>(
    table: &Table8,
    scalars: &[[u8; 32]; N],
    out: &mut [Extended; N],
) {
    unsafe {
        const { assert!(N > 0 && N.is_multiple_of(8)) };
        // digits[i][k] = k 番目の鍵の i 桁目
        let mut digits = [[0i16; N]; ROWS];
        for (k, s) in scalars.iter().enumerate() {
            for (i, d) in signed_digits(s).into_iter().enumerate() {
                digits[i][k] = d;
            }
        }
        // 1行目は単位元への加算なので、表の値をそのまま座標にする
        let mut acc: Vec<Ext8<F>> = Vec::with_capacity(N / 8);
        let row = table.row(0);
        for (g, d) in digits[0].as_chunks::<8>().0.iter().enumerate() {
            prefetch_next(table, &digits, 0, g);
            acc.push(to_extended(&gather::<F>(row, d)));
        }
        // 桁が 0 のレーンは単位元（表の0列目）を足す。8レーンまとめて計算するので飛ばさない
        for i in 1..ROWS - 1 {
            let row = table.row(i);
            for (g, (a, d)) in acc.iter_mut().zip(digits[i].as_chunks::<8>().0).enumerate() {
                prefetch_next(table, &digits, i, g);
                add_niels::<F, true>(a, &gather(row, d));
            }
        }
        let row = table.row(ROWS - 1);
        let last = digits[ROWS - 1].as_chunks::<8>().0;
        for ((p, d), o) in acc.iter_mut().zip(last).zip(out.as_chunks_mut::<8>().0) {
            add_niels::<F, false>(p, &gather(row, d));
            let (x, y, z) = (p.x.lanes(), p.y.lanes(), p.z.lanes());
            for k in 0..8 {
                o[k] = Extended {
                    x: x[k],
                    y: y[k],
                    z: z[k],
                    t: Fe::ZERO,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::check;
    use crate::rfc8032::secret_expand;
    use crate::rfc8032::to_le32;
    use crate::rfc8032::{Vector, vectors};
    use crate::service::derive::comb::compress_with_zinv;
    use crate::service::derive::field::Fe;
    use crate::simd_support::available;
    use curve25519_dalek::edwards::EdwardsPoint;
    use curve25519_dalek::scalar::clamp_integer;
    use proptest::array::uniform8;
    use proptest::collection::vec;
    use proptest::option::weighted;
    use proptest::prelude::*;
    use rstest::rstest;
    use similar_asserts::assert_eq;
    use std::array::from_fn;
    use std::mem::transmute;

    /// AVX-512 の2通りの経路
    #[derive(Debug, Clone, Copy)]
    enum Path {
        /// AVX-512F まで（25.5 ビット × 10 リム）
        F,
        /// AVX-512 IFMA（51 ビット × 5 リム）
        Ifma,
    }

    /// `path` で `scalars[k] * B` を計算する（CPU が対応していなければ None）
    fn mul_base<const N: usize>(path: Path, scalars: &[[u8; 32]; N]) -> Option<[Extended; N]> {
        let mut out = [Extended::IDENTITY; N];
        match path {
            Path::F => f::mul_base_batch(available("avx512", Avx512::detect())?, scalars, &mut out),
            Path::Ifma => {
                ifma::mul_base_batch(available("avx512-ifma", Ifma::detect())?, scalars, &mut out)
            }
        }
        Some(out)
    }

    fn compress(p: &Extended) -> [u8; 32] {
        compress_with_zinv(p, p.z.invert())
    }

    /// curve25519-dalek で `s * B`（s は 2^255 未満の任意の値）
    fn dalek(s: &[u8; 32]) -> [u8; 32] {
        EdwardsPoint::mul_base(&curve25519_dalek::Scalar::from_bytes_mod_order(*s))
            .compress()
            .to_bytes()
    }

    /// 桁が 0（表の0列目 = 単位元を足す）・負（符号の反転）・最大・最小になりやすいスカラー
    const SPECIAL: [[u8; 32]; 5] = {
        let mut s = [[0u8; 32]; 5];
        s[0][31] = 0x40; // 2^254: 途中の桁がすべて 0
        // s[1] = 0: すべての桁が 0
        let mut i = 0;
        while i < 32 {
            s[2][i] = 0xff; // 全桁で繰り上がる（最上位バイトは下で 2^255 未満に）
            s[3][i] = 0x88; // 桁が正・負に入り混じる
            s[4][i] = 0x77;
            i += 1;
        }
        s[2][31] = 0x7f;
        s[3][31] = 0x08;
        s
    };

    /// スカラーの組（`clamp` ならクランプ、そうでなければ 2^255 未満の任意の値）に、[`SPECIAL`] を
    /// 確率 1/4 で混ぜる
    fn scalars<const N: usize>(clamp: bool) -> impl Strategy<Value = [[u8; 32]; N]> {
        (
            vec(any::<[u8; 32]>(), N),
            vec(weighted(0.25, 0..SPECIAL.len()), N),
        )
            .prop_map(move |(random, pick)| {
                from_fn(|k| match pick[k] {
                    Some(i) => SPECIAL[i],
                    None if clamp => clamp_integer(random[k]),
                    None => {
                        let mut s = random[k];
                        s[31] &= 0x7f;
                        s
                    }
                })
            })
    }

    /// IFMA があれば AVX-512 の基本命令もある。MEIKEY_REQUIRE_SIMD で必須にした命令セットは、
    /// このマシンで本当に使える
    #[rstest]
    fn detection_is_consistent() {
        if let Some(i) = Ifma::detect() {
            let _ = i.avx512();
            assert_eq!(Avx512::detect().is_some(), true);
        }
        let _ = available("avx512", Avx512::detect());
        let _ = available("avx512-ifma", Ifma::detect());
    }

    /// 正規化した値を表の形にして（pack）読み戻すと（unpack）、同じ値になる。
    /// blend はマスクの立ったレーンだけ2つ目の値を選ぶ
    fn pack_unpack_blend<F: Field8>(
        a: [[u64; 5]; 8],
        b: [[u64; 5]; 8],
        k: u8,
    ) -> Result<(), TestCaseError> {
        let words = |v: [[u64; 5]; 8]| -> [__m512i; 5] {
            let p = v.map(F::pack);
            from_fn(|i| {
                let lane: [u64; 8] = from_fn(|l| p[l][i]);
                // SAFETY: 同じ大きさで、どんなビット列も有効な値
                unsafe { transmute::<[u64; 8], __m512i>(lane) }
            })
        };
        // SAFETY: 呼び出し元が CPU の対応を確かめている
        let (x, y, z) = unsafe {
            let (x, y) = (F::unpack(&words(a)), F::unpack(&words(b)));
            (x.lanes(), y.lanes(), F::blend(k, &x, &y).lanes())
        };
        for l in 0..8 {
            prop_assert_eq!(x[l].reduce(), a[l]);
            prop_assert_eq!(y[l].reduce(), b[l]);
            let want = if k >> l & 1 == 1 { b[l] } else { a[l] };
            prop_assert_eq!(z[l].reduce(), want);
        }
        Ok(())
    }

    #[rstest]
    fn pack_unpack_blend_keep_values(#[values(Path::F, Path::Ifma)] path: Path) {
        let fe = any::<[u8; 32]>().prop_map(|mut b| {
            b[31] &= 0x7f;
            Fe::from_bytes(&b).reduce()
        });
        let lanes = uniform8(fe);
        let ok = match path {
            Path::F => available("avx512", Avx512::detect()).is_some(),
            Path::Ifma => available("avx512-ifma", Ifma::detect()).is_some(),
        };
        if !ok {
            return;
        }
        check(
            256,
            (lanes.clone(), lanes, any::<u8>()),
            |(a, b, k)| match path {
                Path::F => pack_unpack_blend::<f::Fe10>(a, b, k),
                Path::Ifma => pack_unpack_blend::<ifma::Fe5>(a, b, k),
            },
        );
    }

    /// 8・16・256 鍵（8鍵の組が1つ・2つ・32）をまとめて倍算すると、どの位置の鍵も curve25519-dalek と
    /// 一致する。組が複数あると、先読み（次の組・次の行）とアキュムレータの切り替えも通る
    #[rstest]
    fn mul_base_batch_matches_dalek(
        #[values(Path::F, Path::Ifma)] path: Path,
        #[values(true, false)] clamp: bool,
        #[values(8, 16, 256)] n: usize,
    ) {
        fn run<const N: usize>(path: Path, clamp: bool, cases: u32) {
            check(cases, scalars::<N>(clamp), |s| {
                let Some(out) = mul_base(path, &s) else {
                    return Ok(());
                };
                for (k, (p, s)) in out.iter().zip(&s).enumerate() {
                    prop_assert_eq!(compress(p), dalek(s), "#{}", k);
                }
                Ok(())
            });
        }
        match n {
            8 => run::<8>(path, clamp, 64),
            16 => run::<16>(path, clamp, 32),
            _ => run::<256>(path, clamp, 2),
        }
    }

    /// RFC 8032 Section 7.1 の5本のテストベクタの秘密スカラーで、参照実装と一致する
    /// （8レーンに5本を並べ、残りの3レーンには先頭から繰り返す）
    #[rstest]
    fn mul_base_batch_matches_rfc_vectors(
        #[values(Path::F, Path::Ifma)] path: Path,
        vectors: Vec<Vector>,
    ) {
        let v = vectors;
        assert_eq!(v.len(), 5);
        let s: [[u8; 32]; 8] = from_fn(|k| to_le32(&secret_expand(&v[k % v.len()].secret).0));
        let Some(out) = mul_base(path, &s) else {
            return;
        };
        for (k, p) in out.iter().enumerate() {
            let want = &v[k % v.len()];
            assert_eq!(compress(p), want.public, "TEST {}", want.name);
        }
    }
}
