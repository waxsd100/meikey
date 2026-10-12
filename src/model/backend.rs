//! 公開鍵の計算方法（どの高速化を使うか）
//!
//! CPU が対応しているかで決まる部分（[`Simd::effective`] など）と、表を作る
//! [`Backend::prepare`] は、計算そのものと一緒に [`crate::service::derive`] にある。

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
/// （上から Avx512Ifma → Avx512 → Avx2 → Off。x86_64 以外では常に Off）
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
}

impl Simd {
    /// 上から順に
    pub const ALL: [Simd; 4] = [Simd::Avx512Ifma, Simd::Avx512, Simd::Avx2, Simd::Off];

    pub fn name(self) -> &'static str {
        match self {
            Simd::Avx512Ifma => "AVX-512 IFMA",
            Simd::Avx512 => "AVX-512",
            Simd::Avx2 => "AVX2",
            Simd::Off => "なし",
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
    /// すべての組み合わせ（先頭が FAST、最後が BASELINE）
    pub const ALL: [Backend; 16] = {
        let muls = [Mul::Table, Mul::ConstantTime];
        let compresses = [Compress::Batch, Compress::Each];
        let mut all = [Backend::FAST; 16];
        let mut i = 0;
        while i < 16 {
            all[i] = Backend {
                mul: muls[i / 8],
                compress: compresses[(i / 4) % 2],
                simd: Simd::ALL[i % 4],
            };
            i += 1;
        }
        all
    };
}
