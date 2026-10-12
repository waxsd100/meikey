//! シードから公開鍵をまとめて計算する
//!
//! ed25519 の公開鍵は `clamp(SHA-512(seed)[..32]) * B` を圧縮したもの。
//! 探索ではこれを [`BATCH`] 個ずつまとめて計算し、圧縮に必要な逆元を1回にまとめる。
//! x86_64 では CPU に合わせて、SHA-512 とテーブルの倍算を AVX-512 で8鍵ずつ（IFMA があれば
//! 52 ビットの積和で）、AVX2 で4鍵ずつ同時に計算する。
//!
//! 計算方法の型（[`Backend`] など）は [`crate::model::backend`] にあり、ここでは CPU が対応して
//! いるかで決まる部分（[`Simd::effective`]・[`Backend::prepare`]・表示）を足している。
//!
//! 中の部品（crate の外には公開しない）:
//! - `field` — GF(2^255 - 19) の体演算（51 ビット × 5 リム）
//! - `comb` — 櫛形テーブルによる固定基点倍算
//! - `avx2`・`avx512` — 倍算を4鍵・8鍵ずつ同時に
//! - `sha512x4`・`sha512x8` — SHA-512 を4シード・8シードずつ同時に

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;
pub(crate) mod comb;
pub(crate) mod field;
#[cfg(target_arch = "x86_64")]
pub(crate) mod sha512x4;
#[cfg(target_arch = "x86_64")]
pub(crate) mod sha512x8;

use crate::model::backend::{Backend, Compress, Mul, Simd};
#[cfg(target_arch = "x86_64")]
use avx2::{Avx2, Table4};
#[cfg(target_arch = "x86_64")]
use avx512::{Avx512, Ifma};
use comb::{Extended, Table};
use curve25519_dalek::edwards::EdwardsPoint;
use curve25519_dalek::scalar::clamp_integer;
use field::Fe;
use sha2::{Digest, Sha512};

/// 一度にまとめて計算する鍵の数
pub const BATCH: usize = 256;

pub type Seeds = [[u8; 32]; BATCH];
pub type PublicKeys = [[u8; 32]; BATCH];

impl Simd {
    /// この CPU で使えるか
    pub fn supported(self) -> bool {
        Lanes::detect(self).simd() == self
    }

    /// この上限で実際に使うもの
    pub fn effective(self) -> Simd {
        Lanes::detect(self).simd()
    }
}

/// 実際に使う SIMD と、CPU が対応していることの証明
#[derive(Debug, Clone, Copy)]
enum Lanes {
    #[cfg(target_arch = "x86_64")]
    Ifma(Ifma),
    #[cfg(target_arch = "x86_64")]
    Avx512(Avx512),
    #[cfg(target_arch = "x86_64")]
    Avx2(Avx2),
    Scalar,
}

impl Lanes {
    /// `limit` 以下で、CPU が対応している中で一番上のもの
    fn detect(limit: Simd) -> Lanes {
        #[cfg(target_arch = "x86_64")]
        {
            if limit >= Simd::Avx512Ifma
                && let Some(i) = Ifma::detect()
            {
                return Lanes::Ifma(i);
            }
            if limit >= Simd::Avx512
                && let Some(a) = Avx512::detect()
            {
                return Lanes::Avx512(a);
            }
            if limit >= Simd::Avx2
                && let Some(a) = Avx2::detect()
            {
                return Lanes::Avx2(a);
            }
        }
        let _ = limit;
        Lanes::Scalar
    }

    fn simd(self) -> Simd {
        match self {
            #[cfg(target_arch = "x86_64")]
            Lanes::Ifma(_) => Simd::Avx512Ifma,
            #[cfg(target_arch = "x86_64")]
            Lanes::Avx512(_) => Simd::Avx512,
            #[cfg(target_arch = "x86_64")]
            Lanes::Avx2(_) => Simd::Avx2,
            Lanes::Scalar => Simd::Off,
        }
    }
}

impl Backend {
    /// 探索で使う表を作っておく（探索の時間に含めないため。2回目からは何もしない）
    pub fn prepare(&self) {
        if self.mul != Mul::Table {
            return;
        }
        match Lanes::detect(self.simd) {
            #[cfg(target_arch = "x86_64")]
            Lanes::Ifma(i) => avx512::ifma::prepare(i),
            #[cfg(target_arch = "x86_64")]
            Lanes::Avx512(a) => avx512::f::prepare(a),
            #[cfg(target_arch = "x86_64")]
            Lanes::Avx2(a) => {
                Table4::get(a);
            }
            Lanes::Scalar => {
                Table::get();
            }
        }
    }
}

/// 実際に使う SIMD を表示する（上限より下がったときは理由も添える）
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
        let used = self.simd.effective();
        write!(f, "倍算={mul}・圧縮={compress}・SIMD={}", used.name())?;
        if used != self.simd {
            write!(f, "（{} は CPU が対応していません）", self.simd.name())?;
        }
        Ok(())
    }
}
/// スレッドごとに1つ持つ作業領域
pub struct Deriver {
    backend: Backend,
    lanes: Lanes,
    points: Box<[Extended; BATCH]>,
    zs: Box<[Fe; BATCH]>,
    scratch: Vec<Fe>,
}

impl Deriver {
    pub fn new(backend: Backend) -> Self {
        backend.prepare();
        Self {
            backend,
            lanes: Lanes::detect(backend.simd),
            points: Box::new([Extended::IDENTITY; BATCH]),
            zs: Box::new([Fe::ZERO; BATCH]),
            scratch: Vec::with_capacity(BATCH),
        }
    }

    pub fn public_keys(&mut self, seeds: &Seeds, out: &mut PublicKeys) {
        let scalars = self.scalars(seeds);
        match self.backend.mul {
            Mul::Table => self.table(&scalars, out),
            Mul::ConstantTime => constant_time(&scalars, out, self.backend.compress),
        }
    }

    /// 全シードの秘密スカラー
    fn scalars(&self, seeds: &Seeds) -> Seeds {
        let mut out: Seeds = [[0u8; 32]; BATCH];
        match self.lanes {
            #[cfg(target_arch = "x86_64")]
            Lanes::Ifma(i) => scalars_avx512(i.avx512(), seeds, &mut out),
            #[cfg(target_arch = "x86_64")]
            Lanes::Avx512(a) => scalars_avx512(a, seeds, &mut out),
            #[cfg(target_arch = "x86_64")]
            Lanes::Avx2(a) => {
                for (o, s) in out
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(seeds.as_chunks::<4>().0)
                {
                    *o = crate::service::derive::sha512x4::prefixes(a, s).map(clamp_integer);
                }
            }
            Lanes::Scalar => {
                for (o, s) in out.iter_mut().zip(seeds) {
                    *o = scalar(s);
                }
            }
        }
        out
    }

    fn table(&mut self, scalars: &Seeds, out: &mut PublicKeys) {
        match self.lanes {
            #[cfg(target_arch = "x86_64")]
            Lanes::Ifma(i) => avx512::ifma::mul_base_batch(i, scalars, &mut self.points),
            #[cfg(target_arch = "x86_64")]
            Lanes::Avx512(a) => avx512::f::mul_base_batch(a, scalars, &mut self.points),
            #[cfg(target_arch = "x86_64")]
            Lanes::Avx2(a) => Table4::get(a).mul_base_batch(a, scalars, &mut self.points),
            Lanes::Scalar => Table::get().mul_base_batch(scalars, &mut self.points),
        }
        for (p, z) in self.points.iter().zip(&mut self.zs) {
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

/// SHA-512 を AVX-512 で8シードずつ
#[cfg(target_arch = "x86_64")]
fn scalars_avx512(a: Avx512, seeds: &Seeds, out: &mut Seeds) {
    for (o, s) in out
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(seeds.as_chunks::<8>().0)
    {
        *o = crate::service::derive::sha512x8::prefixes(a, s).map(clamp_integer);
    }
}

fn constant_time(scalars: &Seeds, out: &mut PublicKeys, compress: Compress) {
    let point = |s: &[u8; 32]| EdwardsPoint::mul_base_clamped(*s);
    match compress {
        Compress::Batch => {
            let points: [EdwardsPoint; BATCH] = std::array::from_fn(|i| point(&scalars[i]));
            let compressed = EdwardsPoint::compress_batch::<BATCH>(&points);
            for (o, c) in out.iter_mut().zip(compressed.iter()) {
                *o = c.to_bytes();
            }
        }
        Compress::Each => {
            for (o, s) in out.iter_mut().zip(scalars) {
                *o = point(s).compress().to_bytes();
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
        #[values(Simd::Avx512Ifma, Simd::Avx512, Simd::Avx2, Simd::Off)] simd: Simd,
    ) {
        let mut d = Deriver::new(Backend {
            mul,
            compress,
            simd,
        });
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

    /// --bench の表示。SIMD は実際に使うものを表示し、上限より下がったときは理由も添える。
    /// ALL は16通りを1回ずつ、先頭が既定（全部の高速化あり）、最後が BASELINE（全部なし）
    #[rstest]
    #[case::fast(Backend::FAST, "倍算=テーブル・圧縮=まとめて・SIMD=AVX-512 IFMA")]
    #[case::no_batch(Backend { compress: Compress::Each, ..Backend::FAST }, "倍算=テーブル・圧縮=1鍵ずつ・SIMD=AVX-512 IFMA")]
    #[case::constant_time(Backend { mul: Mul::ConstantTime, ..Backend::FAST }, "倍算=定数時間・圧縮=まとめて・SIMD=AVX-512 IFMA")]
    #[case::avx512(Backend { simd: Simd::Avx512, ..Backend::FAST }, "倍算=テーブル・圧縮=まとめて・SIMD=AVX-512")]
    #[case::avx2(Backend { simd: Simd::Avx2, ..Backend::FAST }, "倍算=テーブル・圧縮=まとめて・SIMD=AVX2")]
    #[case::no_simd(Backend { simd: Simd::Off, ..Backend::FAST }, "倍算=テーブル・圧縮=まとめて・SIMD=なし")]
    #[case::baseline(Backend::BASELINE, "倍算=定数時間・圧縮=1鍵ずつ・SIMD=なし")]
    fn backend_display(#[case] backend: Backend, #[case] full: &str) {
        let got = backend.to_string();
        if backend.simd.supported() {
            assert_eq!(got, full);
        } else {
            let (head, _) = full.split_once("SIMD=").unwrap();
            let want = format!(
                "{head}SIMD={}（{} は CPU が対応していません）",
                backend.simd.effective().name(),
                backend.simd.name()
            );
            assert_eq!(got, want);
        }
        assert_eq!(Backend::ALL.iter().filter(|&&b| b == backend).count(), 1);
        assert_eq!(Backend::default(), Backend::FAST);
        assert_eq!(Backend::ALL[0], Backend::FAST);
        assert_eq!(Backend::ALL[15], Backend::BASELINE);
    }

    /// 各 SIMD を「使える」と判定するのは、CPU がその命令セットに対応しているときだけ
    /// （[`Simd::supported`] とは別に、命令セットの検出と直接比べる）
    #[cfg(target_arch = "x86_64")]
    #[rstest]
    fn simd_support_matches_cpu() {
        assert_eq!(Simd::Avx512Ifma.supported(), Ifma::detect().is_some());
        assert_eq!(Simd::Avx512.supported(), Avx512::detect().is_some());
        assert_eq!(Simd::Avx2.supported(), Avx2::detect().is_some());
    }

    /// SIMD の上限は、CPU が対応していなければ、対応している中で一番上のものに下がる
    #[rstest]
    fn simd_falls_back_to_best_supported(
        #[values(Simd::Avx512Ifma, Simd::Avx512, Simd::Avx2, Simd::Off)] limit: Simd,
    ) {
        let used = limit.effective();
        assert!(used <= limit && used.supported());
        assert_eq!(limit.supported(), used == limit);
        for s in Simd::ALL {
            if used < s && s <= limit {
                assert!(
                    !s.supported(),
                    "{s:?} に対応しているのに {used:?} に下がった"
                );
            }
        }
        assert!(Simd::Off.supported());
        #[cfg(not(target_arch = "x86_64"))]
        assert_eq!(used, Simd::Off);
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

        /// Deriver のバッチの指定の位置に入れても、テストベクタどおり。
        /// 一括逆元は先頭・2番目・途中・末尾を通し、1鍵ずつの圧縮は先頭と末尾だけ
        #[rstest]
        #[case::table_batch_first(Mul::Table, Compress::Batch, 0)]
        #[case::table_batch_second(Mul::Table, Compress::Batch, 1)]
        #[case::table_batch_middle(Mul::Table, Compress::Batch, 130)]
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
            #[values(Simd::Avx512Ifma, Simd::Avx512, Simd::Avx2, Simd::Off)] simd: Simd,
            #[values("1", "2", "3", "1024", "SHA(abc)")] name: &str,
        ) {
            let v = rfc8032::vector(name);
            let mut seeds = filler_seeds();
            seeds[position] = v.secret;
            assert_eq!(
                derive(
                    Backend {
                        mul,
                        compress,
                        simd
                    },
                    &seeds
                )[position],
                v.public
            );
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
            #[values(Simd::Avx512Ifma, Simd::Avx512, Simd::Avx2, Simd::Off)] simd: Simd,
        ) {
            let vectors = rfc8032::vectors();
            let mut seeds = filler_seeds();
            for (i, v) in vectors.iter().enumerate() {
                seeds[i * 50] = v.secret;
            }
            let out = derive(
                Backend {
                    mul,
                    compress,
                    simd,
                },
                &seeds,
            );
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
            #[values(Simd::Avx512Ifma, Simd::Avx512, Simd::Avx2, Simd::Off)] simd: Simd,
        ) {
            check(16, (seeds_strategy(), 0..BATCH), |(seeds, pick)| {
                let got = derive(
                    Backend {
                        mul,
                        compress,
                        simd,
                    },
                    &seeds,
                );
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
            #[values(Simd::Avx512Ifma, Simd::Avx512, Simd::Avx2, Simd::Off)] simd: Simd,
        ) {
            check(16, (any::<[u8; 32]>(), 0..BATCH), |(seed, position)| {
                let mut seeds = filler_seeds();
                seeds[position] = seed;
                let pk = derive(
                    Backend {
                        mul,
                        compress,
                        simd,
                    },
                    &seeds,
                )[position];

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
