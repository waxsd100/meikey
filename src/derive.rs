//! シードから公開鍵をまとめて計算する
//!
//! ed25519 の公開鍵は `clamp(SHA-512(seed)[..32]) * B` を圧縮したもの。
//! 探索ではこれを [`BATCH`] 個ずつまとめて計算し、圧縮に必要な逆元を1回にまとめる。

use crate::comb::{self, Extended, Table};
use crate::field::Fe;
use curve25519_dalek::edwards::EdwardsPoint;
use curve25519_dalek::scalar::clamp_integer;
use sha2::{Digest, Sha512};

/// 一度にまとめて計算する鍵の数
pub const BATCH: usize = 256;

pub type Seeds = [[u8; 32]; BATCH];
pub type PublicKeys = [[u8; 32]; BATCH];

/// 公開鍵の計算方法。2つの高速化を別々に切り替えられる（既定は両方あり）。
/// 両方なしにすると、ed25519-dalek で1鍵ずつ計算する高速化前と同じ計算になる
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Backend {
    /// 基点倍算
    pub mul: Mul,
    /// 圧縮（1/Z の計算）
    pub compress: Compress,
}

/// 基点倍算 `s * B` の方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mul {
    /// 自前の 8 ビット幅テーブルを直接引く（加算32回。可変時間なので探索専用）
    #[default]
    Table,
    /// curve25519-dalek の定数時間実装（加算64回 + テーブルの全走査。タイミング攻撃に強いが遅い）
    ConstantTime,
}

/// 圧縮に必要な 1/Z の計算方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compress {
    /// [`BATCH`] 鍵ぶんの逆元を1回にまとめる（Montgomery の方法）
    #[default]
    Batch,
    /// 1鍵ずつ逆元を計算する
    Each,
}

impl Backend {
    /// 両方の高速化あり（既定）
    pub const FAST: Backend = Backend {
        mul: Mul::Table,
        compress: Compress::Batch,
    };
    /// 両方の高速化なし（ed25519-dalek で1鍵ずつ計算するのと同じ）
    pub const BASELINE: Backend = Backend {
        mul: Mul::ConstantTime,
        compress: Compress::Each,
    };
    /// すべての組み合わせ
    pub const ALL: [Backend; 4] = [
        Backend::FAST,
        Backend {
            mul: Mul::Table,
            compress: Compress::Each,
        },
        Backend {
            mul: Mul::ConstantTime,
            compress: Compress::Batch,
        },
        Backend::BASELINE,
    ];
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mul = match self.mul {
            Mul::Table => "テーブル",
            Mul::ConstantTime => "定数時間",
        };
        let compress = match self.compress {
            Compress::Batch => "まとめて",
            Compress::Each => "1鍵ずつ",
        };
        write!(f, "倍算={mul}・圧縮={compress}")
    }
}

/// スレッドごとに1つ持つ作業領域
pub struct Deriver {
    backend: Backend,
    points: Box<[Extended; BATCH]>,
    zs: Box<[Fe; BATCH]>,
    scratch: Vec<Fe>,
}

impl Deriver {
    pub fn new(backend: Backend) -> Self {
        if backend.mul == Mul::Table {
            Table::get(); // テーブル作成を探索開始前に済ませる
        }
        Self {
            backend,
            points: Box::new([Extended::IDENTITY; BATCH]),
            zs: Box::new([Fe::ZERO; BATCH]),
            scratch: Vec::with_capacity(BATCH),
        }
    }

    pub fn public_keys(&mut self, seeds: &Seeds, out: &mut PublicKeys) {
        match self.backend.mul {
            Mul::Table => self.table(seeds, out),
            Mul::ConstantTime => constant_time(seeds, out, self.backend.compress),
        }
    }

    fn table(&mut self, seeds: &Seeds, out: &mut PublicKeys) {
        let table = Table::get();
        for ((seed, p), z) in seeds.iter().zip(&mut self.points).zip(&mut self.zs) {
            *p = table.mul_base(&scalar(seed));
            *z = p.z;
        }
        match self.backend.compress {
            Compress::Batch => Fe::batch_invert(&mut self.zs[..], &mut self.scratch),
            Compress::Each => self.zs.iter_mut().for_each(|z| *z = z.invert()),
        }
        for ((p, z_inv), o) in self.points.iter().zip(&self.zs).zip(out) {
            *o = comb::compress_with_zinv(p, *z_inv);
        }
    }
}

fn constant_time(seeds: &Seeds, out: &mut PublicKeys, compress: Compress) {
    let point = |seed| EdwardsPoint::mul_base_clamped(scalar(seed));
    match compress {
        Compress::Batch => {
            let points: [EdwardsPoint; BATCH] = std::array::from_fn(|i| point(&seeds[i]));
            let compressed = EdwardsPoint::compress_batch::<BATCH>(&points);
            for (o, c) in out.iter_mut().zip(compressed.iter()) {
                *o = c.to_bytes();
            }
        }
        Compress::Each => {
            for (o, seed) in out.iter_mut().zip(seeds) {
                *o = point(seed).compress().to_bytes();
            }
        }
    }
}

/// RFC 8032: SHA-512(seed) の前半32バイトをクランプしたものが秘密スカラー
#[inline(always)]
fn scalar(seed: &[u8; 32]) -> [u8; 32] {
    let h = Sha512::digest(seed);
    clamp_integer(*h.first_chunk().expect("SHA-512 は64バイト"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VanityKey;
    use rand::rngs::ChaCha20Rng;
    use rand::{Rng, SeedableRng};
    use rstest::rstest;

    /// テストで流すバッチ
    #[derive(Debug, Clone, Copy)]
    enum Batch {
        /// 決定的な乱数のシード 256 個
        Random(u64),
        /// 同じシードを 256 個（一括逆元で全部同じ Z になる）
        Same([u8; 32]),
    }

    impl Batch {
        fn seeds(self) -> Seeds {
            match self {
                Batch::Random(rng_seed) => {
                    let mut rng = ChaCha20Rng::seed_from_u64(rng_seed);
                    let mut s: Seeds = [[0u8; 32]; BATCH];
                    s.iter_mut().for_each(|x| rng.fill_bytes(x));
                    s
                }
                Batch::Same(seed) => [seed; BATCH],
            }
        }
    }

    /// 1つの Deriver に続けてバッチを流しても、前のバッチの値が混ざらず全部 ed25519-dalek と一致する
    #[rstest]
    #[case::random_batches(&[Batch::Random(1), Batch::Random(2), Batch::Random(3)])]
    #[case::all_zero_seed(&[Batch::Same([0; 32])])]
    #[case::all_ff_seed(&[Batch::Same([0xff; 32])])]
    #[case::mixed(&[Batch::Same([0; 32]), Batch::Random(4), Batch::Same([0xff; 32])])]
    fn batches_match_ed25519_dalek(
        #[case] batches: &[Batch],
        #[values(Mul::Table, Mul::ConstantTime)] mul: Mul,
        #[values(Compress::Batch, Compress::Each)] compress: Compress,
    ) {
        let mut d = Deriver::new(Backend { mul, compress });
        let mut out: PublicKeys = [[0u8; 32]; BATCH];
        for (round, batch) in batches.iter().enumerate() {
            let seeds = batch.seeds();
            d.public_keys(&seeds, &mut out);
            for (i, (seed, pk)) in seeds.iter().zip(&out).enumerate() {
                assert_eq!(
                    pk,
                    VanityKey::from_seed(*seed).public_bytes(),
                    "{round} 回目の #{i}"
                );
            }
        }
    }

    /// --bench の表示。既定は両方の高速化あり、BASELINE は両方なし、ALL は4通りを1回ずつ
    #[rstest]
    #[case::fast(Backend::FAST, "倍算=テーブル・圧縮=まとめて")]
    #[case::no_batch(Backend { mul: Mul::Table, compress: Compress::Each }, "倍算=テーブル・圧縮=1鍵ずつ")]
    #[case::constant_time(Backend { mul: Mul::ConstantTime, compress: Compress::Batch }, "倍算=定数時間・圧縮=まとめて")]
    #[case::baseline(Backend::BASELINE, "倍算=定数時間・圧縮=1鍵ずつ")]
    fn backend_display(#[case] backend: Backend, #[case] want: &str) {
        assert_eq!(backend.to_string(), want);
        assert_eq!(Backend::ALL.iter().filter(|&&b| b == backend).count(), 1);
        assert_eq!(Backend::default(), Backend::FAST);
    }

    // ---- RFC 8032 Section 5.1.5 手順 1・2（SHA-512 とクランプ） ----

    use crate::property::check;
    use crate::rfc8032;
    use proptest::prelude::*;

    #[rstest]
    fn scalar_matches_rfc_secret_expand_for_vectors(
        #[values("1", "2", "3", "1024", "SHA(abc)")] name: &str,
    ) {
        let v = rfc8032::vector(name);
        assert_eq!(
            scalar(&v.secret),
            rfc8032::to_le32(&rfc8032::secret_expand(&v.secret).0)
        );
    }

    /// 任意のシードで、RFC の secret_expand と同じスカラーになる。手順 1・2 を1ビットずつも確かめる
    #[rstest]
    fn scalar_is_pruned_sha512_prefix() {
        check(4096, any::<[u8; 32]>(), |seed| {
            let s = scalar(&seed);
            prop_assert_eq!(s, rfc8032::to_le32(&rfc8032::secret_expand(&seed).0));

            // 手順2: 先頭オクテットの下位3ビットを 0、最終オクテットの最上位を 0、2番目を 1
            prop_assert_eq!(s[0] & 0b0000_0111, 0);
            prop_assert_eq!(s[31] & 0b1000_0000, 0);
            prop_assert_eq!(s[31] & 0b0100_0000, 0b0100_0000);
            // それ以外のビットは SHA-512 の前半 32 バイトそのまま（手順1）
            let h = Sha512::digest(seed);
            prop_assert_eq!(s[0] & !0b111, h[0] & !0b111);
            prop_assert_eq!(&s[1..31], &h[1..31]);
            prop_assert_eq!(s[31] & 0b0011_1111, h[31] & 0b0011_1111);
            Ok(())
        });
    }

    /// RFC 8032 への準拠（正解は tests/support/ の独立実装）。公開鍵を作るすべての経路が Section 7.1 の
    /// テストベクタどおりで、出力は Section 5.1.2 / 5.1.3 の正規の符号化・位数 L の点
    mod conformance {
        use super::*;

        fn derive(mul: Mul, compress: Compress, seeds: &Seeds) -> PublicKeys {
            let mut out: PublicKeys = [[0u8; 32]; BATCH];
            Deriver::new(Backend { mul, compress }).public_keys(seeds, &mut out);
            out
        }

        /// 残りの位置を埋めるシード（何でもよいが、テストごとに同じにする）
        fn filler_seeds() -> Seeds {
            std::array::from_fn(|i| [i as u8 ^ 0xa5; 32])
        }

        fn seeds_strategy() -> impl Strategy<Value = Seeds> {
            prop::collection::vec(any::<[u8; 32]>(), BATCH).prop_map(|v| v.try_into().unwrap())
        }

        // ===========================================================================
        // RFC 8032 Section 7.1: 自前の高速版を含む、公開鍵を作る全経路がテストベクタどおり
        // ===========================================================================

        /// Deriver のバッチの指定の位置に入れても、テストベクタどおり。
        /// 一括逆元は先頭・2番目・途中・末尾を通し、1鍵ずつの圧縮は先頭と末尾だけ
        #[rstest]
        #[case::table_batch_first(Mul::Table, Compress::Batch, 0)]
        #[case::table_batch_second(Mul::Table, Compress::Batch, 1)]
        #[case::table_batch_middle(Mul::Table, Compress::Batch, 128)]
        #[case::table_batch_last(Mul::Table, Compress::Batch, BATCH - 1)]
        #[case::table_each_first(Mul::Table, Compress::Each, 0)]
        #[case::table_each_last(Mul::Table, Compress::Each, BATCH - 1)]
        #[case::constant_time_batch_first(Mul::ConstantTime, Compress::Batch, 0)]
        #[case::constant_time_batch_last(Mul::ConstantTime, Compress::Batch, BATCH - 1)]
        #[case::constant_time_each_first(Mul::ConstantTime, Compress::Each, 0)]
        #[case::constant_time_each_last(Mul::ConstantTime, Compress::Each, BATCH - 1)]
        fn section_7_1_public_key(
            #[case] mul: Mul,
            #[case] compress: Compress,
            #[case] position: usize,
            #[values("1", "2", "3", "1024", "SHA(abc)")] name: &str,
        ) {
            let v = rfc8032::vector(name);
            let mut seeds = filler_seeds();
            seeds[position] = v.secret;
            assert_eq!(derive(mul, compress, &seeds)[position], v.public);
        }

        /// 書き出しに使う VanityKey（ed25519-dalek）もテストベクタどおり
        #[rstest]
        fn section_7_1_vanity_key(#[values("1", "2", "3", "1024", "SHA(abc)")] name: &str) {
            let v = rfc8032::vector(name);
            assert_eq!(*VanityKey::from_seed(v.secret).public_bytes(), v.public);
        }

        /// 5本を1つのバッチに入れても、それぞれ正しい（一括逆元が混ざらない）
        #[rstest]
        fn section_7_1_all_vectors_in_one_batch(
            #[values(Mul::Table, Mul::ConstantTime)] mul: Mul,
            #[values(Compress::Batch, Compress::Each)] compress: Compress,
        ) {
            let vectors = rfc8032::vectors();
            let mut seeds = filler_seeds();
            for (i, v) in vectors.iter().enumerate() {
                seeds[i * 50] = v.secret;
            }
            let out = derive(mul, compress, &seeds);
            for (i, v) in vectors.iter().enumerate() {
                assert_eq!(out[i * 50], v.public, "TEST {}", v.name);
            }
        }

        // ===========================================================================
        // RFC 8032 Section 5.1.2 / 5.1.3: 公開鍵は正規の符号化で、復号すると位数 L の点
        // ===========================================================================

        /// バッチ全体が ed25519-dalek と一致し、抜き出した1件は RFC 8032 の参照実装とも一致する。
        /// （1バッチ 256 鍵なので、4通りの組み合わせ × 16 バッチ = 16,384 鍵を確かめる）
        #[rstest]
        fn implementations_agree(
            #[values(Mul::Table, Mul::ConstantTime)] mul: Mul,
            #[values(Compress::Batch, Compress::Each)] compress: Compress,
        ) {
            check(16, (seeds_strategy(), 0..BATCH), |(seeds, pick)| {
                let got = derive(mul, compress, &seeds);
                for (i, (seed, pk)) in seeds.iter().zip(&got).enumerate() {
                    prop_assert_eq!(
                        *pk,
                        *VanityKey::from_seed(*seed).public_bytes(),
                        "#{} dalek",
                        i
                    );
                }
                prop_assert_eq!(
                    got[pick],
                    rfc8032::secret_to_public(&seeds[pick]),
                    "RFC 8032 参照実装"
                );
                Ok(())
            });
        }

        /// 出力は Section 5.1.3 で復号でき、再符号化すると同じバイト列（正規形）、
        /// しかも基点 B が生成する位数 L の部分群に入っている
        #[rstest]
        fn outputs_are_canonical_points_of_order_l(
            #[values(Mul::Table, Mul::ConstantTime)] mul: Mul,
            #[values(Compress::Batch, Compress::Each)] compress: Compress,
        ) {
            check(16, (any::<[u8; 32]>(), 0..BATCH), |(seed, position)| {
                let mut seeds = filler_seeds();
                seeds[position] = seed;
                let pk = derive(mul, compress, &seeds)[position];

                let point = rfc8032::point_decompress(&pk);
                prop_assert!(point.is_some(), "Section 5.1.3 で復号できない: {:02x?}", pk);
                let point = point.unwrap();
                prop_assert_eq!(rfc8032::point_compress(&point), pk, "正規の符号化でない");
                prop_assert!(
                    rfc8032::is_identity(&rfc8032::point_mul(&rfc8032::Q, &point)),
                    "[L]A ≠ 0"
                );
                Ok(())
            });
        }
    }
}
