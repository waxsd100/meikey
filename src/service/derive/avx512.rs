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
//! - 表引き: 8鍵ぶんのエントリを読み込み、8×8 の転置でワードごとのレジスタにする。定数時間の倍算（[`ct`]、
//!   `*_ct` の関数）では、行の全エントリを読み込み、`vpermt2q` で各レーンの桁のエントリを選ぶ（[`CtTable8`]）
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
use crate::service::derive::ct;
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
pub(crate) struct Table8 {
    entries: Box<[Entry]>,
    /// 1行のエントリ数
    cols: usize,
}

impl Table8 {
    pub(crate) fn build<F: Field8>() -> Table8 {
        // 51 ビット版は変換にだけ使って捨てる（AVX-512 を使うときは要らないので、共有のものは作らない）
        let table = Table::build();
        let entries = (0..ROWS * COLS).map(|n| table.entry(n / COLS, n % COLS));
        Table8::pack::<F>(entries, COLS)
    }

    fn pack<F: Field8>(entries: impl Iterator<Item = [Fe; 3]>, cols: usize) -> Table8 {
        let entries = entries
            .map(|entry| {
                let mut e = Entry([0; 16]);
                for (f, fe) in entry.into_iter().enumerate() {
                    e.0[5 * f..5 * f + 5].copy_from_slice(&F::pack(fe.reduce()));
                }
                e
            })
            .collect();
        Table8 { entries, cols }
    }

    #[inline(always)]
    fn row(&self, i: usize) -> &[Entry] {
        &self.entries[i * self.cols..(i + 1) * self.cols]
    }
}

/// 8鍵ぶんの表の値 `d[k] * 2^(W*i) * B` を集め、レーンにそろえる（負の桁は符号を反転する）。
/// 8つのエントリをそれぞれ 512 ビット × 2 で読み、8×8 の転置でワードごとのレジスタにする
/// （`vpgatherqq` で 15 回集めるより速かった）
#[inline(always)]
unsafe fn gather<F: Field8>(row: &[Entry], d: &[i16; 8]) -> Niels8<F> {
    unsafe {
        // |d| ≤ 2^(W-1) < COLS なので行の中
        let e: [&Entry; 8] = std::array::from_fn(|k| &row[usize::from(d[k].unsigned_abs())]);
        let mut w = [_mm512_setzero_si512(); 15];
        for half in 0..2 {
            let r: [__m512i; 8] =
                std::array::from_fn(|k| _mm512_load_si512(e[k].0[8 * half..].as_ptr().cast()));
            let t = transpose8(&r);
            for q in 0..8.min(15 - 8 * half) {
                w[8 * half + q] = t[q];
            }
        }
        // 桁の符号ビットがそのまま「負のレーン」のマスクになる
        let d = _mm512_cvtepi16_epi64(_mm_loadu_si128(d.as_ptr().cast()));
        let neg = _mm512_cmplt_epi64_mask(d, _mm512_setzero_si512());
        niels(&w, neg)
    }
}

/// 8×8 の 64 ビットの転置（`r[k]` の `q` 番目のワードが、結果の `q` 番目のレジスタのレーン `k` になる）
#[inline(always)]
unsafe fn transpose8(r: &[__m512i; 8]) -> [__m512i; 8] {
    unsafe {
        let t: [__m512i; 8] = std::array::from_fn(|i| {
            let (a, b) = (r[i / 2 * 2], r[i / 2 * 2 + 1]);
            match i % 2 {
                0 => _mm512_unpacklo_epi64(a, b),
                _ => _mm512_unpackhi_epi64(a, b),
            }
        });
        // t[0..8] = lo(r0,r1), hi(r0,r1), lo(r2,r3), hi(r2,r3), ...
        let u = [
            _mm512_shuffle_i64x2::<0x88>(t[0], t[2]),
            _mm512_shuffle_i64x2::<0x88>(t[1], t[3]),
            _mm512_shuffle_i64x2::<0xdd>(t[0], t[2]),
            _mm512_shuffle_i64x2::<0xdd>(t[1], t[3]),
            _mm512_shuffle_i64x2::<0x88>(t[4], t[6]),
            _mm512_shuffle_i64x2::<0x88>(t[5], t[7]),
            _mm512_shuffle_i64x2::<0xdd>(t[4], t[6]),
            _mm512_shuffle_i64x2::<0xdd>(t[5], t[7]),
        ];
        [
            _mm512_shuffle_i64x2::<0x88>(u[0], u[4]),
            _mm512_shuffle_i64x2::<0x88>(u[1], u[5]),
            _mm512_shuffle_i64x2::<0x88>(u[2], u[6]),
            _mm512_shuffle_i64x2::<0x88>(u[3], u[7]),
            _mm512_shuffle_i64x2::<0xdd>(u[0], u[4]),
            _mm512_shuffle_i64x2::<0xdd>(u[1], u[5]),
            _mm512_shuffle_i64x2::<0xdd>(u[2], u[6]),
            _mm512_shuffle_i64x2::<0xdd>(u[3], u[7]),
        ]
    }
}

/// 定数時間の倍算（[`ct`]）の表。並べ替え命令で選べるよう、行・ワードごとに 1 番目以降のエントリの
/// 同じワードを 16 個ずつ（512 ビットのレジスタ2本分）並べ、0 番目のエントリは別に持つ
pub(crate) struct CtTable8 {
    /// [行][ワード q（15）][16 個の組 c][前半・後半] の 8 ワード。組 c の j 番目は 1 + 16c + j 番目のエントリ
    perm: Box<[Words8]>,
    /// [行] の 0 番目のエントリの 15 ワード（最上位の行以外は単位元）
    zero: Box<[[u64; 16]]>,
    /// 1行の 16 個の組の数
    groups: usize,
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Words8([u64; 8]);

impl CtTable8 {
    /// 幅 `w` の表（`w` は 4 以上。1 番目以降のエントリが 2^(w-1) 個で、16 個の組にそろえる）
    pub(crate) fn build<F: Field8>(w: usize) -> CtTable8 {
        let cols = ct::cols(w);
        let groups = (cols - 1).div_ceil(16);
        assert!(groups <= 16, "表引きのマスクは 16 組まで");
        let entries: Vec<[u64; 15]> = ct::table(w)
            .into_iter()
            .map(|entry| {
                let mut words = [0u64; 15];
                for (f, fe) in entry.into_iter().enumerate() {
                    words[5 * f..5 * f + 5].copy_from_slice(&F::pack(fe.reduce()));
                }
                words
            })
            .collect();
        let mut perm = Vec::new();
        let mut zero = Vec::new();
        for row in entries.chunks(cols) {
            let mut z = [0u64; 16];
            z[..15].copy_from_slice(&row[0]);
            zero.push(z);
            for q in 0..15 {
                for c in 0..groups {
                    for half in 0..2 {
                        perm.push(Words8(std::array::from_fn(|l| {
                            row.get(1 + 16 * c + 8 * half + l).map_or(0, |e| e[q])
                        })));
                    }
                }
            }
        }
        CtTable8 {
            perm: perm.into_boxed_slice(),
            zero: zero.into_boxed_slice(),
            groups,
        }
    }
}

/// [`gather`] の定数時間版: `i` 行目の全エントリを読み込み、`vpermt2q`（2本のレジスタの 16 個から、
/// レーンごとの番号で1つを選ぶ）で |d| - 1 番目を選ぶ。16 個の組が複数あれば全部の組で選び、
/// 番号の組と一致したレーンにだけ移す。桁が 0 のレーンは 0 番目のエントリにする。
/// 読む場所と命令の列は桁の値によらない
#[inline(always)]
unsafe fn select_ct<F: Field8>(table: &CtTable8, i: usize, d: &[i16; 8]) -> Niels8<F> {
    unsafe {
        let d = _mm512_cvtepi16_epi64(_mm_loadu_si128(d.as_ptr().cast()));
        let neg = _mm512_cmplt_epi64_mask(d, _mm512_setzero_si512());
        let abs = _mm512_abs_epi64(d);
        let is_zero = _mm512_cmpeq_epi64_mask(abs, _mm512_setzero_si512());
        // vpermt2q は番号の下 4 ビットだけを使う。桁が 0 のレーンの番号（-1）は、最後に上書きする
        let idx = _mm512_sub_epi64(abs, _mm512_set1_epi64(1));
        let group = _mm512_srli_epi64::<4>(idx);
        let groups = table.groups;
        let mut in_group = [0 as __mmask8; 16];
        for (c, k) in in_group.iter_mut().enumerate().take(groups) {
            *k = _mm512_cmpeq_epi64_mask(group, _mm512_set1_epi64(c as i64));
        }
        let zero = &table.zero[i];
        let words = &table.perm[i * 15 * groups * 2..(i + 1) * 15 * groups * 2];
        let mut w = [_mm512_setzero_si512(); 15];
        for q in 0..15 {
            let g = &words[q * groups * 2..(q + 1) * groups * 2];
            let pick = |c: usize| {
                let (lo, hi) = (g[2 * c].0.as_ptr(), g[2 * c + 1].0.as_ptr());
                _mm512_permutex2var_epi64(
                    _mm512_load_si512(lo.cast()),
                    idx,
                    _mm512_load_si512(hi.cast()),
                )
            };
            let mut r = pick(0);
            for c in 1..groups {
                r = _mm512_mask_mov_epi64(r, in_group[c], pick(c));
            }
            w[q] = _mm512_mask_set1_epi64(r, is_zero, zero[q] as i64);
        }
        niels(&w, neg)
    }
}

/// 集めた 15 ワード（y+x, y-x, 2dxy の 5 ワードずつ）を体の元に戻し、マスク `neg` のレーンの符号を反転する
#[inline(always)]
unsafe fn niels<F: Field8>(w: &[__m512i; 15], neg: __mmask8) -> Niels8<F> {
    unsafe {
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
unsafe fn prefetch(row: &[Entry], d: &[i16; 8]) {
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

/// [`Table::mul_base_batch`] の最後の加算の前まで（8鍵ずつの組の点と、最後の行の桁を返す）
#[inline(always)]
unsafe fn mul_base_until_last<F: Field8, const N: usize>(
    table: &Table8,
    scalars: &[[u8; 32]; N],
) -> (Vec<Ext8<F>>, [i16; N]) {
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
        (acc, digits[ROWS - 1])
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
        let (mut acc, last) = mul_base_until_last::<F, N>(table, scalars);
        let row = table.row(ROWS - 1);
        for ((p, d), o) in acc
            .iter_mut()
            .zip(last.as_chunks::<8>().0)
            .zip(out.as_chunks_mut::<8>().0)
        {
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

/// [`mul_base_batch`] に続けて、1/Z と y の圧縮形式まで8鍵ずつ計算する（`N` は 8 の倍数。[`compress`]）
#[inline(always)]
unsafe fn mul_base_compress<F: Field8, const N: usize>(
    table: &Table8,
    scalars: &[[u8; 32]; N],
    ys: &mut [[u8; 32]; N],
    xs: &mut [Fe; N],
    zinvs: &mut [Fe; N],
) {
    unsafe {
        let (mut acc, last) = mul_base_until_last::<F, N>(table, scalars);
        let row = table.row(ROWS - 1);
        for (p, d) in acc.iter_mut().zip(last.as_chunks::<8>().0) {
            add_niels::<F, false>(p, &gather(row, d));
        }
        compress::<F, N>(&acc, ys, xs, zinvs);
    }
}

/// 8鍵ずつの組の点から、1/Z と y の圧縮形式を計算する。
/// `ys[k]` は k 番目の公開鍵の圧縮形式で x の符号のビットを 0 にしたもの、`xs[k]`・`zinvs[k]` は
/// 符号を後から計算するための X と 1/Z。
///
/// 1/Z は Montgomery の一括逆元をレーンごとに（8本の鎖を並べて）する: 組を順に掛けた積の逆元を
/// 1回だけ計算し、後ろの組から 1/Z を取り出す
#[inline(always)]
unsafe fn compress<F: Field8, const N: usize>(
    acc: &[Ext8<F>],
    ys: &mut [[u8; 32]; N],
    xs: &mut [Fe; N],
    zinvs: &mut [Fe; N],
) {
    unsafe {
        // prefix[g] = 0〜g 番目の組の Z の積（レーンごと）
        let mut prefix: Vec<F> = Vec::with_capacity(N / 8);
        for (g, p) in acc.iter().enumerate() {
            prefix.push(match g {
                0 => p.z,
                _ => prefix[g - 1].mul(&p.z),
            });
        }
        let mut inv = invert(&prefix[N / 8 - 1]);
        for g in (0..N / 8).rev() {
            let p = &acc[g];
            let zinv = match g {
                0 => inv,
                _ => inv.mul(&prefix[g - 1]),
            };
            if g > 0 {
                inv = inv.mul(&p.z);
            }
            let (y, x, zi) = (p.y.mul(&zinv).lanes(), p.x.lanes(), zinv.lanes());
            for k in 0..8 {
                ys[8 * g + k] = y[k].to_bytes();
                xs[8 * g + k] = x[k];
                zinvs[8 * g + k] = zi[k];
            }
        }
    }
}

/// 定数時間の倍算（[`ct`]。幅 `W`、行数 `R`）で、8鍵ずつの組の点を最後の加算まで（T は計算しない）
#[inline(always)]
unsafe fn points_ct<F: Field8, const W: usize, const R: usize, const N: usize>(
    table: &CtTable8,
    scalars: &[[u8; 32]; N],
) -> Vec<Ext8<F>> {
    unsafe {
        const { assert!(N > 0 && N.is_multiple_of(8)) };
        // digits[i][k] = k 番目の鍵の i 桁目
        let mut digits = [[0i16; N]; R];
        for (k, s) in scalars.iter().enumerate() {
            for (i, d) in ct::digits::<W, R>(s).into_iter().enumerate() {
                digits[i][k] = d;
            }
        }
        let mut acc: Vec<Ext8<F>> = digits[0]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|d| to_extended(&select_ct::<F>(table, 0, d)))
            .collect();
        for (i, ds) in digits.iter().enumerate().take(R - 1).skip(1) {
            for (a, d) in acc.iter_mut().zip(ds.as_chunks::<8>().0) {
                add_niels::<F, true>(a, &select_ct(table, i, d));
            }
        }
        for (a, d) in acc.iter_mut().zip(digits[R - 1].as_chunks::<8>().0) {
            add_niels::<F, false>(a, &select_ct(table, R - 1, d));
        }
        acc
    }
}

/// [`mul_base_batch`] の定数時間版（表は [`CtTable8::build`] で幅 `W` のもの）
#[inline(always)]
unsafe fn mul_base_batch_ct<F: Field8, const W: usize, const R: usize, const N: usize>(
    table: &CtTable8,
    scalars: &[[u8; 32]; N],
    out: &mut [Extended; N],
) {
    unsafe {
        let acc = points_ct::<F, W, R, N>(table, scalars);
        for (p, o) in acc.iter().zip(out.as_chunks_mut::<8>().0) {
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

/// [`mul_base_compress`] の定数時間版（表は [`CtTable8::build`] で幅 `W` のもの）
#[inline(always)]
unsafe fn mul_base_compress_ct<F: Field8, const W: usize, const R: usize, const N: usize>(
    table: &CtTable8,
    scalars: &[[u8; 32]; N],
    ys: &mut [[u8; 32]; N],
    xs: &mut [Fe; N],
    zinvs: &mut [Fe; N],
) {
    unsafe {
        let acc = points_ct::<F, W, R, N>(table, scalars);
        compress::<F, N>(&acc, ys, xs, zinvs);
    }
}

/// 1/a = a^(p-2)（レーンごと。field.rs の invert と同じ加算連鎖で、2乗は乗算でする）
#[inline(always)]
unsafe fn invert<F: Field8>(a: &F) -> F {
    unsafe {
        let sq = |x: &F| x.mul(x);
        let pow2k = |x: &F, k: usize| (0..k).fold(*x, |x, _| sq(&x));
        let a2 = sq(a);
        let a9 = sq(&sq(&a2)).mul(a);
        let a11 = a9.mul(&a2);
        let a_5_0 = sq(&a11).mul(&a9);
        let a_10_0 = pow2k(&a_5_0, 5).mul(&a_5_0);
        let a_20_0 = pow2k(&a_10_0, 10).mul(&a_10_0);
        let a_40_0 = pow2k(&a_20_0, 20).mul(&a_20_0);
        let a_50_0 = pow2k(&a_40_0, 10).mul(&a_10_0);
        let a_100_0 = pow2k(&a_50_0, 50).mul(&a_50_0);
        let a_200_0 = pow2k(&a_100_0, 100).mul(&a_100_0);
        let a_250_0 = pow2k(&a_200_0, 50).mul(&a_50_0);
        pow2k(&a_250_0, 5).mul(&a11)
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

    /// `path` で `scalars[k] * B` を計算する（`ct` なら定数時間の倍算。CPU が対応していなければ None）
    fn mul_base<const N: usize>(
        path: Path,
        ct: bool,
        scalars: &[[u8; 32]; N],
    ) -> Option<[Extended; N]> {
        let mut out = [Extended::IDENTITY; N];
        match (path, ct) {
            (Path::F, false) => {
                f::mul_base_batch(available("avx512", Avx512::detect())?, scalars, &mut out)
            }
            (Path::F, true) => {
                f::mul_base_batch_ct(available("avx512", Avx512::detect())?, scalars, &mut out)
            }
            (Path::Ifma, false) => {
                ifma::mul_base_batch(available("avx512-ifma", Ifma::detect())?, scalars, &mut out)
            }
            (Path::Ifma, true) => ifma::mul_base_batch_ct(
                available("avx512-ifma", Ifma::detect())?,
                scalars,
                &mut out,
            ),
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

    /// 桁が 0（m = 0）・負（符号の反転）・最大・最小になりやすいスカラー（どれもクランプ済み）
    const SPECIAL: [[u8; 32]; 5] = {
        let mut s = [[0u8; 32]; 5];
        let mut i = 0;
        while i < 32 {
            s[2][i] = 0xff; // 全桁で繰り上がる
            s[3][i] = 0x88; // 桁が正・負に入り混じる
            s[4][i] = 0x77;
            i += 1;
        }
        // s[0] = 2^254（m = 0 で、桁がすべて 0）、s[1] = 2^254 + 8（m = 1）
        s[1][0] = 8;
        let mut k = 0;
        while k < 5 {
            s[k][0] &= 0xf8;
            s[k][31] = s[k][31] & 0x3f | 0x40;
            k += 1;
        }
        s
    };

    /// クランプしたスカラーの組に、[`SPECIAL`] を確率 1/4 で混ぜる
    fn scalars<const N: usize>() -> impl Strategy<Value = [[u8; 32]; N]> {
        (
            vec(any::<[u8; 32]>(), N),
            vec(weighted(0.25, 0..SPECIAL.len()), N),
        )
            .prop_map(move |(random, pick)| {
                from_fn(|k| match pick[k] {
                    Some(i) => SPECIAL[i],
                    None => clamp_integer(random[k]),
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
    /// 一致する（テーブルの倍算と定数時間の倍算）。組が複数あると、先読み（次の組・次の行）と
    /// アキュムレータの切り替えも通る
    #[rstest]
    fn mul_base_batch_matches_dalek(
        #[values(Path::F, Path::Ifma)] path: Path,
        #[values(false, true)] ct: bool,
        #[values(8, 16, 256)] n: usize,
    ) {
        fn run<const N: usize>(path: Path, ct: bool, cases: u32) {
            check(cases, scalars::<N>(), |s| {
                let Some(out) = mul_base(path, ct, &s) else {
                    return Ok(());
                };
                for (k, (p, s)) in out.iter().zip(&s).enumerate() {
                    prop_assert_eq!(compress(p), dalek(s), "#{}", k);
                }
                Ok(())
            });
        }
        match n {
            8 => run::<8>(path, ct, 64),
            16 => run::<16>(path, ct, 32),
            _ => run::<256>(path, ct, 2),
        }
    }

    /// RFC 8032 Section 7.1 の5本のテストベクタの秘密スカラーで、参照実装と一致する
    /// （8レーンに5本を並べ、残りの3レーンには先頭から繰り返す）
    #[rstest]
    fn mul_base_batch_matches_rfc_vectors(
        #[values(Path::F, Path::Ifma)] path: Path,
        #[values(false, true)] ct: bool,
        vectors: Vec<Vector>,
    ) {
        let v = vectors;
        assert_eq!(v.len(), 5);
        let s: [[u8; 32]; 8] = from_fn(|k| to_le32(&secret_expand(&v[k % v.len()].secret).0));
        let Some(out) = mul_base(path, ct, &s) else {
            return;
        };
        for (k, p) in out.iter().enumerate() {
            let want = &v[k % v.len()];
            assert_eq!(compress(p), want.public, "TEST {}", want.name);
        }
    }
}
