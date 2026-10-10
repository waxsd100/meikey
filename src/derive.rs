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

/// 公開鍵の計算方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// 自前の可変時間実装（速い）
    #[default]
    Fast,
    /// curve25519-dalek の定数時間実装（タイミング攻撃に強いが遅い）
    ConstantTime,
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
        if backend == Backend::Fast {
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
        match self.backend {
            Backend::Fast => self.fast(seeds, out),
            Backend::ConstantTime => constant_time(seeds, out),
        }
    }

    fn fast(&mut self, seeds: &Seeds, out: &mut PublicKeys) {
        let table = Table::get();
        for ((seed, p), z) in seeds.iter().zip(&mut self.points).zip(&mut self.zs) {
            *p = table.mul_base(&scalar(seed));
            *z = p.z;
        }
        Fe::batch_invert(&mut self.zs[..], &mut self.scratch);
        for ((p, z_inv), o) in self.points.iter().zip(&self.zs).zip(out) {
            *o = comb::compress_with_zinv(p, *z_inv);
        }
    }
}

fn constant_time(seeds: &Seeds, out: &mut PublicKeys) {
    let points: [EdwardsPoint; BATCH] =
        std::array::from_fn(|i| EdwardsPoint::mul_base_clamped(scalar(&seeds[i])));
    let compressed = EdwardsPoint::compress_batch::<BATCH>(&points);
    for (o, c) in out.iter_mut().zip(compressed.iter()) {
        *o = c.to_bytes();
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
        #[values(Backend::Fast, Backend::ConstantTime)] backend: Backend,
    ) {
        let mut d = Deriver::new(backend);
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

        fn derive(backend: Backend, seeds: &Seeds) -> PublicKeys {
            let mut out: PublicKeys = [[0u8; 32]; BATCH];
            Deriver::new(backend).public_keys(seeds, &mut out);
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

        /// 公開鍵を作る経路
        #[derive(Debug, Clone, Copy)]
        enum Via {
            /// Deriver でバッチの指定の位置に入れる（一括逆元の先頭・途中・末尾を通す）
            Deriver(Backend, usize),
            /// 書き出しに使う VanityKey（ed25519-dalek）
            VanityKey,
        }

        #[rstest]
        #[case::fast_first(Via::Deriver(Backend::Fast, 0))]
        #[case::fast_second(Via::Deriver(Backend::Fast, 1))]
        #[case::fast_middle(Via::Deriver(Backend::Fast, 128))]
        #[case::fast_last(Via::Deriver(Backend::Fast, BATCH - 1))]
        #[case::constant_time_first(Via::Deriver(Backend::ConstantTime, 0))]
        #[case::constant_time_last(Via::Deriver(Backend::ConstantTime, BATCH - 1))]
        #[case::vanity_key(Via::VanityKey)]
        fn section_7_1_public_key(
            #[case] via: Via,
            #[values("1", "2", "3", "1024", "SHA(abc)")] name: &str,
        ) {
            let v = rfc8032::vector(name);
            let got = match via {
                Via::Deriver(backend, position) => {
                    let mut seeds = filler_seeds();
                    seeds[position] = v.secret;
                    derive(backend, &seeds)[position]
                }
                Via::VanityKey => *VanityKey::from_seed(v.secret).public_bytes(),
            };
            assert_eq!(got, v.public);
        }

        /// 5本を1つのバッチに入れても、それぞれ正しい（一括逆元が混ざらない）
        #[rstest]
        fn section_7_1_all_vectors_in_one_batch(
            #[values(Backend::Fast, Backend::ConstantTime)] backend: Backend,
        ) {
            let vectors = rfc8032::vectors();
            let mut seeds = filler_seeds();
            for (i, v) in vectors.iter().enumerate() {
                seeds[i * 50] = v.secret;
            }
            let out = derive(backend, &seeds);
            for (i, v) in vectors.iter().enumerate() {
                assert_eq!(out[i * 50], v.public, "TEST {}", v.name);
            }
        }

        // ===========================================================================
        // RFC 8032 Section 5.1.2 / 5.1.3: 公開鍵は正規の符号化で、復号すると位数 L の点
        // ===========================================================================

        /// バッチ全体が ed25519-dalek と一致し、抜き出した1件は RFC 8032 の参照実装とも一致する
        #[rstest]
        fn implementations_agree(#[values(Backend::Fast, Backend::ConstantTime)] backend: Backend) {
            check(32, (seeds_strategy(), 0..BATCH), |(seeds, pick)| {
                let got = derive(backend, &seeds);
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
            #[values(Backend::Fast, Backend::ConstantTime)] backend: Backend,
        ) {
            check(32, (any::<[u8; 32]>(), 0..BATCH), |(seed, position)| {
                let mut seeds = filler_seeds();
                seeds[position] = seed;
                let pk = derive(backend, &seeds)[position];

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
