//! 定数時間の固定基点倍算（符号付き 2^W 進の櫛形テーブル）
//!
//! 計算は [`crate::service::derive::comb`] と同じ `s * B = Σ d_i * (2^(W*i) * 8B) + 2^254 * B` で、
//! 桁の幅 W を経路ごとに選ぶ（[`W_AVX2`]・[`W_AVX512F`]・[`W_IFMA`]）。
//! 秘密のスカラーで変わるのは桁の値だけで、次のことはどの鍵でも同じになる。
//! - 桁の分け方（[`digits`]）: 分岐は桁の位置だけで決まる
//! - 表引き: 行の全エントリを読み込み、桁の絶対値をレジスタの並べ替え命令（AVX-512 は `vpermt2q`、
//!   AVX2 は `vpermd`）の番号にして選ぶ。読むメモリの場所は鍵によらず、並べ替え命令の時間も番号によらない。
//!   桁が 0 のレーンと負の桁の符号反転はマスクで扱う
//! - 加算: 桁が 0 でも行の 0 番目のエントリ（最上位の行以外は単位元）を足す。加算の式は例外の無い完全な式
//! - 逆元: 決まった加算連鎖（Montgomery の一括逆元も、掛ける順番は鍵によらない）
//!
//! SIMD の経路（AVX2・AVX-512）は、それぞれのファイルの `*_ct` の関数。SIMD の無い CPU では
//! curve25519-dalek の定数時間の倍算を使う。

use crate::service::derive::comb::{entries, rows_cols, signed_digits_w};
use crate::service::derive::field::Fe;

/// 経路ごとの桁の幅。広げると加算（行数）は減るが、1行の点が倍になり表引きが重くなる。
/// AVX2 は 5〜8、AVX-512F は 6〜9 を測って一番速いものにした（AVX2 は 7、AVX-512F は 8 で、
/// どちらも 1 つ狭い幅との差は数 %）。IFMA は乗算が AVX-512F の約4分の1で表引きの重みが大きくなるので、
/// 1 つ狭くしている（IFMA の CPU では測っていない）
pub(crate) const W_AVX2: usize = 7;
pub(crate) const W_AVX512F: usize = 8;
pub(crate) const W_IFMA: usize = 7;

/// 幅 `w` の行数（ceil(251 / w)）
pub(crate) const fn rows(w: usize) -> usize {
    rows_cols(w).0
}

/// 幅 `w` の1行のエントリ数（0..=2^(w-1)。0 番目は最上位の行以外は単位元）
pub(crate) const fn cols(w: usize) -> usize {
    rows_cols(w).1
}

/// クランプ済みのスカラーの符号付き 2^W 進の桁（`R` = [`rows`]`(W)`。
/// [`crate::service::derive::comb::signed_digits`] と同じ分け方）
#[inline(always)]
pub(crate) fn digits<const W: usize, const R: usize>(s: &[u8; 32]) -> [i16; R] {
    signed_digits_w::<W, R>(s)
}

/// 幅 `w` の表の全エントリ `[y+x, y-x, 2dxy]`（行ごとに [`cols`]`(w)` 個ずつ）
pub(crate) fn table(w: usize) -> Vec<[Fe; 3]> {
    entries(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::check;
    use crate::service::derive::comb::is_clamped;
    use curve25519_dalek::scalar::clamp_integer;
    use num_bigint::BigInt;
    use proptest::prelude::*;
    use rstest::rstest;

    /// 桁は -2^(W-1)〜2^(W-1)-1（最上位だけ 0〜2^(W-1)）で、Σ d_i·2^(W·i) が m（s = 8·(2^251 + m)）に戻る
    fn reconstruct<const W: usize, const R: usize>() {
        let edges = [[0u8; 32], [0xff; 32], [0x88; 32], [0x77; 32], [0x08; 32]].map(clamp_integer);
        let strategy = prop_oneof![
            any::<[u8; 32]>().prop_map(clamp_integer),
            proptest::sample::select(edges.to_vec()),
        ];
        let half = 1i16 << (W - 1);
        check(256, strategy, |s| {
            prop_assert!(is_clamped(&s));
            let d = digits::<W, R>(&s);
            for (i, &x) in d.iter().enumerate() {
                let ok = match i == R - 1 {
                    true => (0..=half).contains(&x),
                    false => (-half..half).contains(&x),
                };
                prop_assert!(ok, "{} 桁目 = {}", i, x);
            }
            let got = d
                .iter()
                .rev()
                .fold(BigInt::from(0), |acc, &x| (acc << W) + BigInt::from(x));
            let m = (BigInt::from_bytes_le(num_bigint::Sign::Plus, &s) >> 3u32)
                - (BigInt::from(1) << 251u32);
            prop_assert_eq!(got, m);
            Ok(())
        });
    }

    #[rstest]
    fn digits_reconstruct_scalar(#[values(4, 5, 6, 7, 8)] w: usize) {
        match w {
            4 => reconstruct::<4, { rows(4) }>(),
            5 => reconstruct::<5, { rows(5) }>(),
            6 => reconstruct::<6, { rows(6) }>(),
            7 => reconstruct::<7, { rows(7) }>(),
            8 => reconstruct::<8, { rows(8) }>(),
            _ => unreachable!("幅 {w} は測っていない"),
        }
    }

    /// 経路ごとの幅は、表引きが扱える範囲（AVX2 は 8 個の組が 16 まで、AVX-512 は 16 個の組が 16 まで）
    #[test]
    fn widths_in_range() {
        for (w, per_group) in [(W_AVX2, 8), (W_AVX512F, 16), (W_IFMA, 16)] {
            assert!((4..=8).contains(&w), "{w}");
            assert!((cols(w) - 1).div_ceil(per_group) <= 16, "{w}");
        }
    }

    /// 行数と列数（例: W = 6 で 42 行・33 列）
    #[rstest]
    #[case(4, 63, 9)]
    #[case(5, 51, 17)]
    #[case(6, 42, 33)]
    #[case(7, 36, 65)]
    #[case(8, 32, 129)]
    fn shape(#[case] w: usize, #[case] r: usize, #[case] c: usize) {
        assert_eq!((rows(w), cols(w)), (r, c));
        assert_eq!(table(w).len(), r * c);
    }
}
