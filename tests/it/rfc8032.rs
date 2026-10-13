//! 参照実装（tests/support/rfc8032.rs）そのものが RFC 8032 どおりかを確かめるテスト（[`tests`]）と、そのヘルパー
//!
//! ここが通っていることが、他のテストで参照実装を「正解」として使う前提になる。
//! 署名と検証のテストは ed25519-dalek にも同じものを通し、独立した2つの実装が RFC の値で揃うことも確かめる。

use crate::common::rfc8032::*;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use num_bigint::BigInt;
use rstest::fixture;

/// RFC 8032 Section 5.1 Table 1 の値（原文から機械的に抜き出したもの）
mod table1 {
    pub const D: &str =
        "37095705934669439343138083508754565189542113879843219016388785533085940283555";
    pub const BX: &str =
        "15112221349535400772501151409588531511454012693041857206046113283949847762202";
    pub const BY: &str =
        "46316835694926478169428394003475163141307993866256225615783033603165251855960";
    pub const L_MINUS_2_252: &str = "27742317777372353535851937790883648493";
}

fn big(s: &str) -> BigInt {
    s.parse().unwrap()
}

/// オイラーの基準: x が平方なら 1、非平方なら p - 1
fn euler(x: &BigInt) -> BigInt {
    x.modpow(&((&*P - 1) / 2), &P)
}

/// 先頭バイトだけが `k` の 32 バイト（リトルエンディアンの小さい整数）
#[fixture]
fn small(#[default(1)] k: u8) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[0] = k;
    b
}

/// Ed25519 の実装
#[derive(Debug, Clone, Copy)]
enum Impl {
    /// RFC 8032 Section 6 を写した参照実装
    Reference,
    /// ed25519-dalek
    Dalek,
}

impl Impl {
    fn public_key(self, secret: &[u8; 32]) -> [u8; 32] {
        match self {
            Impl::Reference => secret_to_public(secret),
            Impl::Dalek => SigningKey::from_bytes(secret).verifying_key().to_bytes(),
        }
    }

    fn sign(self, secret: &[u8; 32], message: &[u8]) -> [u8; 64] {
        match self {
            Impl::Reference => sign(secret, message),
            Impl::Dalek => SigningKey::from_bytes(secret).sign(message).to_bytes(),
        }
    }

    fn verify(self, public: &[u8; 32], message: &[u8], signature: &[u8; 64]) -> bool {
        match self {
            Impl::Reference => verify(public, message, signature),
            Impl::Dalek => VerifyingKey::from_bytes(public).is_ok_and(|key| {
                key.verify(message, &Signature::from_bytes(signature))
                    .is_ok()
            }),
        }
    }
}

/// 正しい署名に加える変更
#[derive(Debug, Clone, Copy)]
enum Tamper {
    MessageBit,
    RBit,
    SBit,
    /// 別のテストベクタの公開鍵で検証する
    OtherPublicKey,
    /// s に q を足す（同じ値を表すが s ≥ q。Section 5.1.7 手順1で拒否する可鍛性の対策）
    SPlusQ,
}

mod tests {
    use super::*;
    use rstest::rstest;
    use similar_asserts::assert_eq;

    // ---- Section 7.1 のテストベクタ ----

    #[rstest]
    fn data_file_has_all_section_7_1_vectors(vectors: Vec<Vector>) {
        let names: Vec<String> = vectors.into_iter().map(|v| v.name).collect();
        assert_eq!(names, ["1", "2", "3", "1024", "SHA(abc)"]);
    }

    /// 秘密鍵から RFC の PUBLIC KEY と SIGNATURE が出て、その署名を検証できる
    #[rstest]
    fn section_7_1(
        #[values(Impl::Reference, Impl::Dalek)] imp: Impl,
        #[values("1", "2", "3", "1024", "SHA(abc)")] name: &str,
    ) {
        let v = vector(name);
        assert_eq!(imp.public_key(&v.secret), v.public, "PUBLIC KEY");
        assert_eq!(imp.sign(&v.secret, &v.message), v.signature, "SIGNATURE");
        assert_eq!(
            imp.verify(&v.public, &v.message, &v.signature),
            true,
            "検証"
        );
    }

    /// どこを変えても検証は通らない（verify が何でも true を返していないことの確認）
    #[rstest]
    fn verify_rejects_tampering(
        #[values(
            Tamper::MessageBit,
            Tamper::RBit,
            Tamper::SBit,
            Tamper::OtherPublicKey,
            Tamper::SPlusQ
        )]
        tamper: Tamper,
        #[values(Impl::Reference, Impl::Dalek)] imp: Impl,
        #[values("1", "2", "3", "1024", "SHA(abc)")] name: &str,
    ) {
        let v = vector(name);
        let (mut public, mut message, mut sig) = (v.public, v.message.clone(), v.signature);
        match tamper {
            Tamper::MessageBit if message.is_empty() => message.push(0),
            Tamper::MessageBit => message[0] ^= 1,
            Tamper::RBit => sig[0] ^= 1,
            Tamper::SBit => sig[32] ^= 1,
            Tamper::OtherPublicKey => public = vector(if name == "1" { "2" } else { "1" }).public,
            Tamper::SPlusQ => {
                let s = from_le(&sig[32..]) + &*Q;
                sig[32..].copy_from_slice(&to_le32(&s));
            }
        }
        assert_eq!(imp.verify(&public, &message, &sig), false);
    }

    // ---- Section 5.1 の定数と曲線の性質 ----

    /// Table 1 の定数と、曲線・群の性質（Section 5.1.4 の「a = -1 は平方、d は非平方」）
    #[rstest]
    #[case::d(D.clone(), big(table1::D))]
    #[case::basepoint_x(to_affine(&G).0, big(table1::BX))]
    #[case::basepoint_y(to_affine(&G).1, big(table1::BY))]
    #[case::group_order(Q.clone(), (BigInt::from(1) << 252) + big(table1::L_MINUS_2_252))]
    #[case::basepoint_on_curve(
        { let (x, y) = to_affine(&G); pm(&(-(&x * &x) + &y * &y - 1 - &*D * &x * &x * &y * &y)) },
        BigInt::ZERO
    )]
    #[case::sqrt_m1(pm(&(&*MODP_SQRT_M1 * &*MODP_SQRT_M1)), &*P - 1)]
    #[case::d_is_non_square(euler(&D), &*P - 1)]
    #[case::minus_one_is_square(euler(&(&*P - 1)), BigInt::from(1))]
    fn section_5_1(#[case] got: BigInt, #[case] want: BigInt) {
        assert_eq!(got, want);
    }

    /// Section 5.1: 基点の位数は L（L 倍すると単位元）
    #[rstest]
    fn basepoint_has_order_l() {
        assert_eq!(is_identity(&point_mul(&Q, &G)), true);
    }

    /// Section 5.1.3: 復号できるもの・できないもの（y ≥ p、x = 0 なのに符号ビットが 1）
    #[rstest]
    #[case::basepoint(point_compress(&G), true)]
    #[case::identity(small(1), true)]
    #[case::y_is_p(to_le32(&P), false)]
    #[case::identity_with_sign_bit({ let mut b = small(1); b[31] |= 0x80; b }, false)]
    fn decompress(#[case] enc: [u8; 32], #[case] ok: bool) {
        assert_eq!(point_decompress(&enc).is_some(), ok);
    }
}
