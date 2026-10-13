//! 公開鍵の計算方法（どの高速化を使うか）
//!
//! CPU が対応しているかで決まる部分（[`Simd::effective`] など）と、表を作る
//! [`Backend::prepare`] は、計算そのものと一緒に [`crate::service::derive`] にある。

use crate::lang::t;

/// 公開鍵の計算方法。高速化を別々に切り替えられる（既定は全部あり）。
/// 全部なしにすると、ed25519-dalek で1鍵ずつ計算する高速化前と同じ計算になる
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Backend {
    /// 基点倍算
    pub mul: Mul,
    /// 圧縮（1/Z の計算）
    pub compress: Compress,
    /// 何鍵ずつ同時に計算するか（使ってよい SIMD の上限）
    pub simd: Simd,
}

/// 基点倍算 `s * B` の方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mul {
    /// 自前の櫛形テーブルを直接引く（可変時間なので探索専用）
    #[default]
    Table,
    /// curve25519-dalek の定数時間実装（タイミング攻撃に強いが遅い）
    ConstantTime,
}

/// 圧縮に必要な 1/Z の計算方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compress {
    /// [`BATCH`](crate::service::derive::BATCH) 鍵ぶんの逆元を1回にまとめる（Montgomery の方法）
    #[default]
    Batch,
    /// 1鍵ずつ逆元を計算する
    Each,
}

/// 使ってよい SIMD の上限。CPU が対応していなければ、対応している中で一番上のものに下がる
/// （上から Amx → Avx512Ifma → Avx512 → Avx2 → Off。x86_64 以外では常に Off）。
/// 既定は Avx512Ifma。実験的な Amx は、feature `amx` を有効にしてビルドしたときだけ使える
/// （無効なら Amx を上限にしても Avx512Ifma から下がる）
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Simd {
    /// 1鍵ずつ（高速化前）
    Off,
    /// AVX2 で4鍵ずつ（体演算は 25.5 ビット × 10 リム）
    Avx2,
    /// AVX-512F で8鍵ずつ（体演算は 25.5 ビット × 10 リム）
    Avx512,
    /// AVX-512 IFMA で8鍵ずつ（体演算は 51 ビット × 5 リムで、52 ビットの積和命令を使う）
    #[default]
    Avx512Ifma,
    /// 体の乗算を AMX のタイル演算（8 ビットの行列積）で（実験的。IFMA より遅い）。
    /// SHA-512 は AVX-512 で8つずつ
    Amx,
}

impl Mul {
    pub fn name(self) -> &'static str {
        match self {
            Mul::Table => t!("テーブル", "table"),
            Mul::ConstantTime => t!("定数時間", "constant time"),
        }
    }
}

impl Simd {
    /// 上から順に（Amx は feature `amx` が有効なときだけ）
    #[cfg(feature = "amx")]
    pub const ALL: [Simd; 5] = [
        Simd::Amx,
        Simd::Avx512Ifma,
        Simd::Avx512,
        Simd::Avx2,
        Simd::Off,
    ];
    /// 上から順に（Amx は feature `amx` が有効なときだけ）
    #[cfg(not(feature = "amx"))]
    pub const ALL: [Simd; 4] = [Simd::Avx512Ifma, Simd::Avx512, Simd::Avx2, Simd::Off];

    pub fn name(self) -> &'static str {
        match self {
            Simd::Amx => "AMX",
            Simd::Avx512Ifma => "AVX-512 IFMA",
            Simd::Avx512 => "AVX-512",
            Simd::Avx2 => "AVX2",
            Simd::Off => t!("なし", "none"),
        }
    }
}

impl Backend {
    /// 全部の高速化あり（既定）
    pub const FAST: Backend = Backend {
        mul: Mul::Table,
        compress: Compress::Batch,
        simd: Simd::Avx512Ifma,
    };
    /// 全部の高速化なし（ed25519-dalek で1鍵ずつ計算するのと同じ）
    pub const BASELINE: Backend = Backend {
        mul: Mul::ConstantTime,
        compress: Compress::Each,
        simd: Simd::Off,
    };
    /// すべての組み合わせ（最後が BASELINE）
    pub const ALL: [Backend; 4 * Simd::ALL.len()] = {
        const S: usize = Simd::ALL.len();
        let muls = [Mul::Table, Mul::ConstantTime];
        let compresses = [Compress::Batch, Compress::Each];
        let mut all = [Backend::FAST; 4 * S];
        let mut i = 0;
        while i < all.len() {
            all[i] = Backend {
                mul: muls[i / (2 * S)],
                compress: compresses[(i / S) % 2],
                simd: Simd::ALL[i % S],
            };
            i += 1;
        }
        all
    };
}
