//! RFC 8032 Section 6「Ed25519 Python Illustration」を、そのまま Rust に写した参照実装（テスト専用）
//!
//! 目的は「RFC のコードと1行ずつ対応させること」だけで、速度や定数時間性は一切考えていない。
//! 写し方の約束:
//! - Python の整数 → `BigInt`（負の数もそのまま扱う。Python と同じく途中では mod を取らない）
//! - Python の `x % p`（床除算の剰余）→ `mod_floor`（関数 `pm`）
//! - Python の `pow(x, e, p)` → `BigInt::modpow`（どちらも結果は 0 以上 p 未満）
//! - Python の `x // 8` など → 被除数が正の場合だけなので `/` と同じ
//!
//! RFC のコードを正しく写せていることは `tests/it/rfc8032.rs` のテストで、RFC 8032 Section 7.1 の
//! 全テストベクタ（公開鍵・署名・検証）と Section 5.1 Table 1 の定数で確かめる。
//!
//! meikey 本体（src/）のユニットテストと IT の両方から `#[path]` で読み込んでいる。

// 読み込む側（ユニットテスト・IT）によって使う関数が違い、未使用が出るかどうかも読み込む側で変わるので
// expect ではなく allow
#![allow(dead_code, reason = "読み込む側によって使う関数が違う")]

use num_bigint::{BigInt, Sign};
use num_integer::Integer;
use num_traits::{One, Zero};
use rstest::fixture;
use sha2::{Digest, Sha512};
use similar_asserts::assert_eq;
use std::sync::LazyLock;

/// 拡張座標 (X, Y, Z, T)、x = X/Z, y = Y/Z, x*y = T/Z
pub type Point = [BigInt; 4];

fn int(v: i64) -> BigInt {
    BigInt::from(v)
}

/// リトルエンディアンのバイト列 → 整数（Python の `int.from_bytes(s, "little")`）
pub fn from_le(s: &[u8]) -> BigInt {
    BigInt::from_bytes_le(Sign::Plus, s)
}

/// 整数 → 32 バイトのリトルエンディアン（Python の `int.to_bytes(n, 32, "little")`）
pub fn to_le32(n: &BigInt) -> [u8; 32] {
    assert_eq!(n.sign() == Sign::Minus, false, "to_bytes: 負の数");
    let (_, bytes) = n.to_bytes_le();
    assert_eq!(bytes.len() <= 32, true, "to_bytes: 32 バイトに収まらない");
    let mut out = [0u8; 32];
    out[..bytes.len()].copy_from_slice(&bytes);
    out
}

// ## First, some preliminaries that will be needed.

pub fn sha512(s: &[u8]) -> Vec<u8> {
    Sha512::digest(s).to_vec()
}

/// Base field Z_p: `p = 2**255 - 19`
pub static P: LazyLock<BigInt> = LazyLock::new(|| (int(1) << 255) - 19);

/// Python の `x % p`
pub fn pm(x: &BigInt) -> BigInt {
    x.mod_floor(&P)
}

pub fn modp_inv(x: &BigInt) -> BigInt {
    x.modpow(&(&*P - 2), &P)
}

/// Curve constant: `d = -121665 * modp_inv(121666) % p`
pub static D: LazyLock<BigInt> = LazyLock::new(|| pm(&(int(-121665) * modp_inv(&int(121666)))));

/// Group order: `q = 2**252 + 27742317777372353535851937790883648493`
pub static Q: LazyLock<BigInt> = LazyLock::new(|| {
    (int(1) << 252)
        + "27742317777372353535851937790883648493"
            .parse::<BigInt>()
            .unwrap()
});

pub fn sha512_modq(s: &[u8]) -> BigInt {
    from_le(&sha512(s)).mod_floor(&Q)
}

// ## Then follows functions to perform point operations.

pub fn point_add(p: &Point, q: &Point) -> Point {
    let a = pm(&((&p[1] - &p[0]) * (&q[1] - &q[0])));
    let b = pm(&((&p[1] + &p[0]) * (&q[1] + &q[0])));
    let c = pm(&(int(2) * &p[3] * &q[3] * &*D));
    let d = pm(&(int(2) * &p[2] * &q[2]));
    let (e, f, g, h) = (&b - &a, &d - &c, &d + &c, &b + &a);
    [&e * &f, &g * &h, &f * &g, &e * &h]
}

/// Computes Q = s * Q
pub fn point_mul(s: &BigInt, p: &Point) -> Point {
    let mut s = s.clone();
    let mut p = p.clone();
    let mut q: Point = [int(0), int(1), int(1), int(0)]; // Neutral element
    while s > BigInt::zero() {
        if s.is_odd() {
            q = point_add(&q, &p);
        }
        p = point_add(&p, &p);
        s >>= 1;
    }
    q
}

pub fn point_equal(p: &Point, q: &Point) -> bool {
    // x1 / z1 == x2 / z2  <==>  x1 * z2 == x2 * z1
    if !pm(&(&p[0] * &q[2] - &q[0] * &p[2])).is_zero() {
        return false;
    }
    if !pm(&(&p[1] * &q[2] - &q[1] * &p[2])).is_zero() {
        return false;
    }
    true
}

// ## Now follows functions for point compression.

/// Square root of -1: `modp_sqrt_m1 = pow(2, (p-1) // 4, p)`
pub static MODP_SQRT_M1: LazyLock<BigInt> = LazyLock::new(|| int(2).modpow(&((&*P - 1) / 4), &P));

/// Compute corresponding x-coordinate, with low bit corresponding to sign, or return None on failure
pub fn recover_x(y: &BigInt, sign: bool) -> Option<BigInt> {
    if y >= &*P {
        return None;
    }
    let x2: BigInt = (y * y - 1) * modp_inv(&(&*D * y * y + 1));
    if x2.is_zero() {
        return if sign { None } else { Some(int(0)) };
    }

    // Compute square root of x2
    let mut x = x2.modpow(&((&*P + 3) / 8), &P);
    if !pm(&(&x * &x - &x2)).is_zero() {
        x = pm(&(x * &*MODP_SQRT_M1));
    }
    if !pm(&(&x * &x - &x2)).is_zero() {
        return None;
    }

    if x.is_odd() != sign {
        x = &*P - x;
    }
    Some(x)
}

/// Base point
pub static G: LazyLock<Point> = LazyLock::new(|| {
    let g_y = pm(&(int(4) * modp_inv(&int(5))));
    let g_x = recover_x(&g_y, false).unwrap();
    let t = pm(&(&g_x * &g_y));
    [g_x, g_y, int(1), t]
});

pub fn point_compress(p: &Point) -> [u8; 32] {
    let zinv = modp_inv(&p[2]);
    let x = pm(&(&p[0] * &zinv));
    let y = pm(&(&p[1] * &zinv));
    to_le32(&(y | ((x & int(1)) << 255)))
}

pub fn point_decompress(s: &[u8]) -> Option<Point> {
    assert_eq!(s.len(), 32, "Invalid input length for decompression");
    let mut y = from_le(s);
    let sign = !(&y >> 255u32).is_zero();
    y &= (int(1) << 255) - 1;

    let x = recover_x(&y, sign)?;
    let t = pm(&(&x * &y));
    Some([x, y, int(1), t])
}

// ## These are functions for manipulating the private key.

pub fn secret_expand(secret: &[u8]) -> (BigInt, Vec<u8>) {
    assert_eq!(secret.len(), 32, "Bad size of private key");
    let h = sha512(secret);
    let mut a = from_le(&h[..32]);
    a &= (int(1) << 254) - 8;
    a |= int(1) << 254;
    (a, h[32..].to_vec())
}

pub fn secret_to_public(secret: &[u8]) -> [u8; 32] {
    let (a, _dummy) = secret_expand(secret);
    point_compress(&point_mul(&a, &G))
}

// ## The signature function works as below.

pub fn sign(secret: &[u8], msg: &[u8]) -> [u8; 64] {
    let (a, prefix) = secret_expand(secret);
    let pub_a = point_compress(&point_mul(&a, &G));
    let r = sha512_modq(&[prefix.as_slice(), msg].concat());
    let rr = point_mul(&r, &G);
    let rs = point_compress(&rr);
    let h = sha512_modq(&[rs.as_slice(), &pub_a, msg].concat());
    let s = (r + h * a).mod_floor(&Q);
    let mut out = [0u8; 64];
    out[..32].copy_from_slice(&rs);
    out[32..].copy_from_slice(&to_le32(&s));
    out
}

// ## And finally the verification function.

pub fn verify(public: &[u8], msg: &[u8], signature: &[u8]) -> bool {
    assert_eq!(public.len(), 32, "Bad public key length");
    // RFC のコードは `Exception("Bad signature length")` の前に raise が無く、実際には何もしない。
    // 原文どおり長さはここでは検査しない（下のスライスで 64 バイト未満なら panic する）
    let Some(pub_a) = point_decompress(public) else {
        return false;
    };
    let rs = &signature[..32];
    let Some(r) = point_decompress(rs) else {
        return false;
    };
    let s = from_le(&signature[32..]);
    if s >= *Q {
        return false;
    }
    let h = sha512_modq(&[rs, public, msg].concat());
    let sb = point_mul(&s, &G);
    let ha = point_mul(&h, &pub_a);
    point_equal(&sb, &point_add(&r, &ha))
}

// ---------------------------------------------------------------------------
// ここから下は RFC のコードには無い、テストで使う補助（RFC の関数の組み合わせだけで書く）
// ---------------------------------------------------------------------------

/// RFC 8032 Section 5.1.5 手順 3・4: スカラー s（リトルエンディアン）で [s]B を計算して符号化する
pub fn base_mul_encoded(scalar: &[u8; 32]) -> [u8; 32] {
    point_compress(&point_mul(&from_le(scalar), &G))
}

/// 点のアフィン座標 (x, y)（0 以上 p 未満）
pub fn to_affine(p: &Point) -> (BigInt, BigInt) {
    let zinv = modp_inv(&p[2]);
    (pm(&(&p[0] * &zinv)), pm(&(&p[1] * &zinv)))
}

pub fn is_identity(p: &Point) -> bool {
    point_equal(p, &[int(0), int(1), int(1), int(0)])
}

pub fn one() -> BigInt {
    BigInt::one()
}

/// RFC 8032 Section 7.1 のテストベクタ1本
#[derive(Debug, Clone)]
pub struct Vector {
    pub name: String,
    pub secret: [u8; 32],
    pub public: [u8; 32],
    pub message: Vec<u8>,
    pub signature: [u8; 64],
}

/// RFC 8032 Section 7.1 の全テストベクタ（tests/data/rfc8032-ed25519.txt）
#[fixture]
pub fn vectors() -> Vec<Vector> {
    const DATA: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/rfc8032-ed25519.txt"
    ));
    DATA.lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(':').collect();
            assert_eq!(f.len(), 5, "壊れた行: {l}");
            Vector {
                name: f[0].to_owned(),
                secret: hex(f[1]).try_into().unwrap(),
                public: hex(f[2]).try_into().unwrap(),
                message: hex(f[3]),
                signature: hex(f[4]).try_into().unwrap(),
            }
        })
        .collect()
}

/// RFC 8032 Section 7.1 のテストベクタを名前（"1", "2", "3", "1024", "SHA(abc)"）で取り出す
#[fixture]
pub fn vector(#[default("1")] name: &str) -> Vector {
    vectors()
        .into_iter()
        .find(|v| v.name == name)
        .unwrap_or_else(|| panic!("RFC 8032 に TEST {name} は無い"))
}

pub fn hex(s: &str) -> Vec<u8> {
    assert_eq!(s.len().is_multiple_of(2), true, "16進の桁数が奇数");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
