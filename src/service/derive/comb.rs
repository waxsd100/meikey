//! 可変時間の固定基点倍算（符号付き 2^W 進の櫛形テーブル）
//!
//! `s * B` を `Σ d_i * (2^(W*i) * B)` として計算する。各 `d_i * 2^(W*i) * B` は表から直接引くので、
//! 倍算は加算 `ROWS` 回だけで済み、倍加は不要。テーブルのアクセス位置が秘密に依存する
//! （＝定数時間ではない）ため、探索の高速化専用。
//!
//! 複数の鍵をまとめて、表の1行ずつ処理する（[`Table::mul_base_batch`]）。表全体は数MBあるが、
//! 同時に使うのは1行（数百KB）だけなので、全コアで回してもキャッシュに収まりやすい。

use crate::service::derive::field::Fe;
use std::sync::LazyLock;

/// 1桁のビット数（8〜14 で計測して 12 に決めた。13 は全スレッドではわずかに速いが、1スレッドでは遅く、テーブルも倍近くになる）
pub const W: usize = 12;
/// 桁数。クランプ済みスカラーは 2^255 未満なので floor(255/W)+1 桁で足りる
pub(crate) const ROWS: usize = 255 / W + 1;
/// 1行のエントリ数（0..=2^(W-1)。負の桁は符号反転で表す）
pub(crate) const COLS: usize = (1 << (W - 1)) + 1;
/// 桁の絶対値の上限 2^(W-1)
const HALF: i32 = 1 << (W - 1);

// 桁は i16 に入れる。最上位桁は繰り上げないので、2^255 未満の値の残りのビットに
// 下からの繰り上がり 1 を足しても 2^(W-1) 以下（＝表の範囲内）に収まる必要がある
const _: () = assert!(W >= 2 && W <= 15);
const _: () = assert!(255 - W * (ROWS - 1) < W);

/// 2d（d はエドワーズ曲線の係数 -121665/121666）
const D2: Fe = Fe([
    1859910466990425,
    932731440258426,
    1072319116312658,
    1815898335770999,
    633789495995903,
]);
/// 基点 B の座標（RFC 8032）
const BX: Fe = Fe([
    1738742601995546,
    1146398526822698,
    2070867633025821,
    562264141797630,
    587772402128613,
]);
const BY: Fe = Fe([
    1801439850948184,
    1351079888211148,
    450359962737049,
    900719925474099,
    1801439850948198,
]);

/// 1/2
const INV2: Fe = Fe([
    2251799813685239,
    2251799813685247,
    2251799813685247,
    2251799813685247,
    1125899906842623,
]);
const TWO: Fe = Fe([2, 0, 0, 0, 0]);

/// 拡張座標 (X:Y:Z:T)、x = X/Z, y = Y/Z, xy = T/Z
#[derive(Clone, Copy, Debug)]
pub struct Extended {
    pub x: Fe,
    pub y: Fe,
    pub z: Fe,
    pub t: Fe,
}

/// 加算用に前計算したアフィン点 (y+x, y-x, 2dxy)
#[derive(Clone, Copy, Debug)]
struct Niels {
    ypx: Fe,
    ymx: Fe,
    xy2d: Fe,
}

impl Niels {
    const IDENTITY: Niels = Niels {
        ypx: Fe::ONE,
        ymx: Fe::ONE,
        xy2d: Fe::ZERO,
    };

    fn from_affine(x: Fe, y: Fe) -> Niels {
        Niels {
            ypx: y + x,
            ymx: y - x,
            xy2d: x * y * D2,
        }
    }

    /// 拡張座標に（乗算2回）。X = y+x - (y-x) = 2x, Y = 2y, Z = 2, T = XY/Z = 2xy
    #[inline(always)]
    fn to_extended(self) -> Extended {
        let x = self.ypx - self.ymx;
        let y = self.ypx + self.ymx;
        Extended {
            x,
            y,
            z: TWO,
            t: x * y * INV2,
        }
    }

    /// -P = (-x, y)
    #[inline(always)]
    fn neg(self) -> Niels {
        Niels {
            ypx: self.ymx,
            ymx: self.ypx,
            xy2d: self.xy2d.neg(),
        }
    }
}

impl Extended {
    pub const IDENTITY: Extended = Extended {
        x: Fe::ZERO,
        y: Fe::ONE,
        z: Fe::ONE,
        t: Fe::ZERO,
    };

    fn from_affine(x: Fe, y: Fe) -> Extended {
        Extended {
            x,
            y,
            z: Fe::ONE,
            t: x * y,
        }
    }

    /// 拡張座標 + アフィン点（乗算 7 回、a = -1 のツイステッドエドワーズ曲線の完全加算公式）
    #[inline(always)]
    fn add_niels(&self, q: &Niels) -> Extended {
        self.add_niels_impl::<true>(q)
    }

    /// `add_niels` の T を計算しない版（最後の加算用。結果の T は使えない）
    #[inline(always)]
    fn add_niels_last(&self, q: &Niels) -> Extended {
        self.add_niels_impl::<false>(q)
    }

    #[inline(always)]
    fn add_niels_impl<const WITH_T: bool>(&self, q: &Niels) -> Extended {
        let pp = (self.y + self.x) * q.ypx;
        let mm = (self.y - self.x) * q.ymx;
        let txy = self.t * q.xy2d;
        let z2 = self.z + self.z;
        let (x, y) = (pp - mm, pp + mm);
        let (z, t) = (z2 + txy, z2 - txy);
        Extended {
            x: x * t,
            y: y * z,
            z: z * t,
            t: if WITH_T { x * y } else { Fe::ZERO },
        }
    }

    /// アフィン座標に戻す（逆元1回）
    fn to_affine(self) -> (Fe, Fe) {
        let zi = self.z.invert();
        (self.x * zi, self.y * zi)
    }
}

/// 圧縮形式（y の32バイト、最上位ビットに x の符号）。`z_inv` は事前に計算した 1/Z
#[inline(always)]
pub fn compress_with_zinv(p: &Extended, z_inv: Fe) -> [u8; 32] {
    let mut y = (p.y * z_inv).to_bytes();
    let x_sign = (p.x * z_inv).to_bytes()[0] & 1;
    y[31] |= x_sign << 7;
    y
}

/// 前計算テーブル: `row(i)[k] = k * 2^(W*i) * B`。
/// 1行が大きい（W = 12 で約 240KB）ので、スタックに置かないよう1本の Vec に並べている
pub struct Table(Box<[Niels]>);

impl Table {
    /// プロセスで1つだけ作って共有する（W = 12 で約 5.4MB、作成は最適化ビルドで約 10ms）
    pub fn get() -> &'static Table {
        static TABLE: LazyLock<Table> = LazyLock::new(Table::build);
        &TABLE
    }

    #[inline(always)]
    fn row(&self, i: usize) -> &[Niels; COLS] {
        &self.0.as_chunks::<COLS>().0[i]
    }

    /// `i` 行目の `k` 番目のエントリ `[y+x, y-x, 2dxy]`（AVX2・AVX-512・GPU 用のテーブルを作るときに使う）
    #[cfg(any(target_arch = "x86_64", feature = "cuda"))]
    pub(crate) fn entry(&self, i: usize, k: usize) -> [Fe; 3] {
        let n = self.row(i)[k];
        [n.ypx, n.ymx, n.xy2d]
    }

    pub(crate) fn build() -> Table {
        let mut entries = vec![Niels::IDENTITY; ROWS * COLS];
        // 各行の基準点 2^(W*i) * B（アフィン）
        let (mut bx, mut by) = (BX, BY);
        let mut scratch = Vec::new();
        for row in entries.as_chunks_mut::<COLS>().0 {
            let base = Niels::from_affine(bx, by);
            let mut acc = Extended::IDENTITY;
            let mut multiples = Vec::with_capacity(COLS);
            for _ in 1..COLS {
                acc = acc.add_niels(&base);
                multiples.push(acc);
            }
            // 1/Z をまとめて計算してアフィンにする
            let mut zs: Vec<Fe> = multiples.iter().map(|p| p.z).collect();
            Fe::batch_invert(&mut zs, &mut scratch);
            for (k, (p, zi)) in multiples.iter().zip(zs).enumerate() {
                row[k + 1] = Niels::from_affine(p.x * zi, p.y * zi);
            }
            // 次の行: 2^W * (今の行の基準点) = 2 * (2^(W-1) * 基準点)
            let half = multiples[COLS - 2]; // 2^(W-1) 倍
            let (hx, hy) = half.to_affine();
            (bx, by) = Extended::from_affine(hx, hy)
                .add_niels(&Niels::from_affine(hx, hy))
                .to_affine();
        }
        Table(entries.into_boxed_slice())
    }

    /// `scalars[k] * B` を `out[k]` に（scalar はリトルエンディアン、2^255 未満であること）。
    ///
    /// 鍵ごとに倍算を終わらせるのではなく、表の1行目を全部の鍵に足し、次に2行目を…と1行ずつ進める。
    /// 同時に使う表が1行だけになるので、表全体がキャッシュに収まらなくても速い。
    /// 戻り値の T は計算していない（圧縮には X, Y, Z だけ使う）。
    pub fn mul_base_batch<const N: usize>(&self, scalars: &[[u8; 32]; N], out: &mut [Extended; N]) {
        // digits[i][k] = k 番目の鍵の i 桁目（行ごとに鍵を順に読めるよう転置しておく）
        let mut digits = [[0i16; N]; ROWS];
        for (k, s) in scalars.iter().enumerate() {
            for (i, d) in signed_digits(s).into_iter().enumerate() {
                digits[i][k] = d;
            }
        }
        let entry = |row: &[Niels; COLS], d: i16| {
            let p = row[usize::from(d.unsigned_abs())];
            if d < 0 { p.neg() } else { p }
        };

        // 最初の桁は単位元への加算なので、表の値をそのまま座標にする
        let row = self.row(0);
        for (acc, &d) in out.iter_mut().zip(&digits[0]) {
            *acc = entry(row, d).to_extended();
        }
        for (i, ds) in digits.iter().enumerate().take(ROWS - 1).skip(1) {
            let row = self.row(i);
            for (acc, &d) in out.iter_mut().zip(ds) {
                if d != 0 {
                    *acc = acc.add_niels(&entry(row, d));
                }
            }
        }
        // 最上位桁（クランプで 2^254 のビットが立つので通常 0 ではない。0 でも単位元の加算になるだけ）
        let row = self.row(ROWS - 1);
        for (acc, &d) in out.iter_mut().zip(&digits[ROWS - 1]) {
            *acc = acc.add_niels_last(&entry(row, d));
        }
    }
}

/// スカラーを符号付き 2^W 進の桁（-2^(W-1) ≤ d < 2^(W-1)、最上位桁のみ 0 ≤ d ≤ 2^(W-1)）に分解する。
/// 下の桁から W ビットずつ取り出して繰り上がりを足し、2^(W-1) 以上なら 2^W を引いて上へ繰り上げる
#[inline(always)]
pub(crate) fn signed_digits(s: &[u8; 32]) -> [i16; ROWS] {
    // 5語目は、最後の桁が4語目からはみ出して読むときの 0
    let mut words = [0u64; 5];
    for (w, c) in words.iter_mut().zip(s.as_chunks::<8>().0) {
        *w = u64::from_le_bytes(*c);
    }
    let mut out = [0i16; ROWS];
    let mut carry = 0i32;
    for (i, o) in out.iter_mut().enumerate() {
        let (q, r) = (i * W / 64, i * W % 64);
        let mut bits = words[q] >> r;
        if r + W > 64 {
            bits |= words[q + 1] << (64 - r);
        }
        let v = (bits & ((1 << W) - 1)) as i32 + carry;
        if i == ROWS - 1 {
            // 最上位桁は繰り上げない（ファイル先頭の assert により 2^(W-1) 以下）
            *o = v as i16;
        } else {
            carry = (v + HALF) >> W;
            *o = (v - (carry << W)) as i16;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rfc8032;
    use crate::rfc8032::base_mul_encoded;
    use crate::rfc8032::from_le;
    use crate::rfc8032::is_identity;
    use crate::rfc8032::point_add;
    use crate::rfc8032::point_compress;
    use crate::rfc8032::point_mul;
    use crate::rfc8032::to_affine;
    use crate::rfc8032::to_le32;
    use curve25519_dalek::edwards::EdwardsPoint;
    use curve25519_dalek::scalar::clamp_integer;
    use proptest::collection::vec;
    use rstest::{fixture, rstest};
    use similar_asserts::assert_eq;
    use std::array::from_fn;

    fn compress(p: &Extended) -> [u8; 32] {
        compress_with_zinv(p, p.z.invert())
    }

    /// 1鍵だけの倍算（テストでは本番と同じ mul_base_batch を N = 1 で通す）
    fn mul_base(s: &[u8; 32]) -> Extended {
        let mut out = [Extended::IDENTITY];
        Table::get().mul_base_batch(&[*s], &mut out);
        out[0]
    }

    /// 桁が範囲内（最上位以外は -2^(W-1)〜2^(W-1)-1、最上位は 0〜2^(W-1)）
    fn digits_in_range(d: &[i16; ROWS]) -> bool {
        let half = HALF as i16;
        d[..ROWS - 1].iter().all(|x| (-half..half).contains(x)) && (0..=half).contains(&d[ROWS - 1])
    }

    /// 桁から元の値に戻す（Σ d_i * 2^(W*i)）
    fn reconstruct(d: &[i16; ROWS]) -> num_bigint::BigInt {
        d.iter()
            .rev()
            .fold(num_bigint::BigInt::from(0), |acc, &x| (acc << W) + x)
    }

    /// 最上位以外の全桁（W ビットずつ）が `v` のスカラー
    #[fixture]
    fn repeated_digit(#[default(0)] v: u32) -> [u8; 32] {
        let s = (0..ROWS - 1).fold(num_bigint::BigInt::from(0), |acc, i| {
            acc + (num_bigint::BigInt::from(v) << (W * i))
        });
        to_le32(&s)
    }

    #[rstest]
    #[case::identity(
        Fe::ZERO,
        Fe::ONE,
        "0100000000000000000000000000000000000000000000000000000000000000"
    )]
    #[case::basepoint(
        BX,
        BY,
        "5866666666666666666666666666666666666666666666666666666666666666"
    )]
    // -B は y が同じで x の符号だけ違う → 最上位ビットが立つ
    #[case::neg_basepoint(BX.neg(), BY, "58666666666666666666666666666666666666666666666666666666666666e6")]
    fn compresses_to_known_value(#[case] x: Fe, #[case] y: Fe, #[case] want: &str) {
        assert_eq!(hex(&compress(&Extended::from_affine(x, y))), want);
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// 境界のスカラー: 桁分解して戻すと元の値になり、倍算は curve25519-dalek・RFC 8032 の参照実装と一致する
    #[rstest]
    #[case::zero([0u8; 32])]
    #[case::clamped_min({ let mut s = [0u8; 32]; s[31] = 0x40; s })] // 2^254（途中の桁はすべて 0）
    #[case::clamped_max(clamp_integer([0xff; 32]))] // 繰り上がりが最上位まで伝わる
    #[case::all_digits_half(repeated_digit(1 << (W - 1)))] // 各桁がちょうど繰り上がりの境界
    #[case::all_digits_half_minus_1(repeated_digit((1 << (W - 1)) - 1))] // 境界の1つ手前（繰り上がらない）
    #[case::all_digits_max(repeated_digit((1 << W) - 1))]
    #[case::ones({ let mut s = [0x01u8; 32]; s[0] = 0x08; s[31] = 0x41; s })]
    #[case::max_below_2_255({ let mut s = [0xffu8; 32]; s[31] = 0x7f; s })] // 全桁で繰り上がる
    fn edge_scalars(#[case] s: [u8; 32]) {
        let d = signed_digits(&s);
        assert_eq!(digits_in_range(&d), true, "{d:?}");
        assert_eq!(reconstruct(&d), from_le(&s), "桁分解");

        let got = compress(&mul_base(&s));
        let dalek = EdwardsPoint::mul_base(&curve25519_dalek::Scalar::from_bytes_mod_order(s));
        assert_eq!(got, dalek.compress().to_bytes(), "curve25519-dalek");
        assert_eq!(got, base_mul_encoded(&s), "RFC 8032 参照実装");
    }

    #[rstest]
    fn niels_to_extended_is_same_point(
        #[values(0, 3, ROWS - 1)] row: usize,
        #[values(0, 1, 2, COLS - 2, COLS - 1)] k: usize,
        #[values(false, true)] negate: bool,
    ) {
        let n = Table::get().row(row)[k];
        let n = if negate { n.neg() } else { n };
        assert_eq!(
            compress(&n.to_extended()),
            compress(&Extended::IDENTITY.add_niels(&n))
        );
    }

    // -----------------------------------------------------------------------
    // RFC 8032 Section 6 の参照実装・curve25519-dalek との比較（proptest）
    // -----------------------------------------------------------------------

    use crate::property::check;
    use proptest::prelude::*;

    fn fe_of(x: &num_bigint::BigInt) -> Fe {
        Fe::from_bytes(&to_le32(x))
    }

    /// 参照実装の点を、このモジュールの拡張座標（Z = 1）に
    fn extended_of(p: &rfc8032::Point) -> Extended {
        let (x, y) = to_affine(p);
        Extended::from_affine(fe_of(&x), fe_of(&y))
    }

    fn niels_of(p: &rfc8032::Point) -> Niels {
        let (x, y) = to_affine(p);
        Niels::from_affine(fe_of(&x), fe_of(&y))
    }

    /// テーブルの全エントリ（22 行 × 2049）を、参照実装で `k * 2^(W*row) * B` を計算して検算する
    #[rstest]
    fn table_matches_rfc_reference_exhaustively(
        #[values(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
        )]
        row: usize,
    ) {
        assert_eq!(ROWS, 22, "行を増減したら #[values] も合わせる");
        let base = point_mul(&(num_bigint::BigInt::from(1) << (W * row)), &rfc8032::G);
        let mut acc = [0, 1, 1, 0].map(num_bigint::BigInt::from); // 単位元
        for (k, entry) in Table::get().row(row).iter().enumerate() {
            let got = compress(&Extended::IDENTITY.add_niels(entry));
            assert_eq!(got, point_compress(&acc), "row {row}, k {k}");
            acc = point_add(&acc, &base);
        }
    }

    /// 倍算に渡すスカラー
    #[derive(Debug, Clone, Copy)]
    enum Scalars {
        /// RFC 8032 Section 5.1.5 手順2でクランプしたもの（鍵の計算で実際に使う形）
        Clamped,
        /// 2^255 未満の任意の値（クランプしていないものも含む）
        Below2_255,
    }

    impl Scalars {
        fn strategy(self) -> BoxedStrategy<[u8; 32]> {
            match self {
                Scalars::Clamped => any::<[u8; 32]>().prop_map(clamp_integer).boxed(),
                Scalars::Below2_255 => any::<[u8; 32]>()
                    .prop_map(|mut s| {
                        s[31] &= 0x7f;
                        s
                    })
                    .boxed(),
            }
        }
    }

    /// 符号付き 2^W 進の桁に分解して戻すと元の値（各桁は範囲内）
    #[rstest]
    fn signed_digits_reconstruct_any_scalar(
        #[values(Scalars::Clamped, Scalars::Below2_255)] scalars: Scalars,
    ) {
        check(1024, scalars.strategy(), |s| {
            let d = signed_digits(&s);
            prop_assert!(digits_in_range(&d), "{:?}", d);
            prop_assert_eq!(reconstruct(&d), from_le(&s));
            Ok(())
        });
    }

    /// 倍算と比べる相手
    #[derive(Debug, Clone, Copy)]
    enum Reference {
        /// curve25519-dalek の定数時間実装
        Dalek,
        /// RFC 8032 Section 6 の参照実装（Section 5.1.5 手順 3・4: [s]B とその符号化）
        Rfc8032,
    }

    impl Reference {
        fn mul_base(self, s: &[u8; 32]) -> [u8; 32] {
            match self {
                Reference::Dalek => {
                    EdwardsPoint::mul_base(&curve25519_dalek::Scalar::from_bytes_mod_order(*s))
                        .compress()
                        .to_bytes()
                }
                Reference::Rfc8032 => base_mul_encoded(s),
            }
        }
    }

    /// 任意のスカラーで、倍算が別の実装と一致する（参照実装は遅いので回数を減らす）
    #[rstest]
    #[case::dalek(Reference::Dalek, 1024)]
    #[case::rfc8032(Reference::Rfc8032, 128)]
    fn mul_base_matches(
        #[case] reference: Reference,
        #[case] cases: u32,
        #[values(Scalars::Clamped, Scalars::Below2_255)] scalars: Scalars,
    ) {
        check(cases, scalars.strategy(), |s| {
            prop_assert_eq!(compress(&mul_base(&s)), reference.mul_base(&s));
            Ok(())
        });
    }

    /// 1つのバッチに入れた鍵どうしは混ざらない。途中の桁が 0 で加算を飛ばす鍵（2^254 など）を
    /// 混ぜても、どの位置の鍵も curve25519-dalek と一致する
    #[rstest]
    fn mul_base_batch_keeps_keys_separate(
        #[values(Scalars::Clamped, Scalars::Below2_255)] scalars: Scalars,
    ) {
        let strategy = (vec(scalars.strategy(), 8), vec(any::<bool>(), 8));
        check(64, strategy, |(random, sparse)| {
            let mut sparse_scalar = [0u8; 32];
            sparse_scalar[31] = 0x40; // 2^254: 途中の桁がすべて 0
            let s: [[u8; 32]; 8] = from_fn(|k| if sparse[k] { sparse_scalar } else { random[k] });
            let mut out = [Extended::IDENTITY; 8];
            Table::get().mul_base_batch(&s, &mut out);
            for (k, (p, s)) in out.iter().zip(&s).enumerate() {
                prop_assert_eq!(compress(p), Reference::Dalek.mul_base(s), "#{}", k);
            }
            Ok(())
        });
    }

    /// 足す2点の関係
    #[derive(Debug, Clone, Copy)]
    enum Relation {
        /// 別々の点 [a]B と [b]B
        Independent,
        /// 同じ点（2倍の計算になる）
        Same,
        /// 逆元（[L-a]B = -[a]B。結果は単位元）
        Negated,
    }

    /// 点の加算の版
    #[derive(Debug, Clone, Copy)]
    enum Variant {
        /// 通常版（拡張座標 + アフィン、乗算7回）
        WithT,
        /// 最後の加算用（T を計算しない。圧縮に使う X, Y, Z は通常版と同じ）
        WithoutT,
    }

    /// 点の加算が RFC 8032 Section 5.1.4 の公式と同じ点を出す。2倍（P = Q）や逆元との加算（結果が単位元）も
    #[rstest]
    fn add_niels_matches_rfc_point_add(
        #[values(Relation::Independent, Relation::Same, Relation::Negated)] relation: Relation,
        #[values(Variant::WithT, Variant::WithoutT)] variant: Variant,
    ) {
        check(128, (1u64.., 1u64..), |(a, b)| {
            let p = point_mul(&a.into(), &rfc8032::G);
            let q = match relation {
                Relation::Independent => point_mul(&b.into(), &rfc8032::G),
                Relation::Same => p.clone(),
                Relation::Negated => point_mul(&(&*rfc8032::Q - a), &rfc8032::G),
            };
            let (p_ext, q_niels) = (extended_of(&p), niels_of(&q));
            let got = match variant {
                Variant::WithT => p_ext.add_niels(&q_niels),
                Variant::WithoutT => p_ext.add_niels_last(&q_niels),
            };
            let want = point_add(&p, &q);
            prop_assert_eq!(compress(&got), point_compress(&want));
            if let Relation::Negated = relation {
                prop_assert!(is_identity(&want));
            }
            Ok(())
        });
    }

    /// 1/Z を外から渡す圧縮は、Section 5.1.2 の符号化（y の32バイト + 最上位ビットに x の符号）
    #[rstest]
    fn compress_follows_section_5_1_2() {
        check(256, 1u64.., |a| {
            let p = point_mul(&a.into(), &rfc8032::G);
            let (x, y) = to_affine(&p);
            let mut want = to_le32(&y);
            want[31] |= u8::from(x.bit(0)) << 7;
            prop_assert_eq!(compress(&extended_of(&p)), want);
            Ok(())
        });
    }
}
