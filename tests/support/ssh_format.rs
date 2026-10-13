//! SSH 関連の形式を仕様書どおりに読み書きする、テスト専用の独立実装
//!
//! - RFC 4648 Section 4: base64（標準の文字表、`=` によるパディング）
//! - RFC 4251 Section 5: `uint32` / `string`
//! - RFC 8709 Section 4: `ssh-ed25519` 公開鍵 blob
//! - sshd(8) AUTHORIZED_KEYS FILE FORMAT: `.pub` の1行（`種類 base64 コメント`）と、
//!   オプション付きの行・コメント行・空行を含む authorized_keys ファイル
//! - OpenSSH PROTOCOL.key: 秘密鍵ファイル（openssh-key-v1）。暗号化されたもの（bcrypt と、OpenSSH の
//!   cipher.c にある暗号方式。chacha20-poly1305 は PROTOCOL.chacha20poly1305）も読み書きできる
//! - draft-ietf-sshm-ssh-agent Section 3.2.3: Ed25519 秘密鍵の中身（`ENC(A)`、`k || ENC(A)`）
//! - OpenSSH PROTOCOL.sshsig + RFC 8709 Section 6: `ssh-keygen -Y sign` の署名（armored 形式・blob・署名対象）
//!
//! meikey 本体と同じ部品（base64 クレートなど）は使わず、仕様の文章から直接書いている。
//! 本体と同じ部品で検証すると、同じ間違いを見逃すため。
//! 暗号化された鍵の暗号そのもの（bcrypt_pbkdf・AES・3DES・ChaCha20・Poly1305）だけは RustCrypto の
//! クレートを使う（meikey 本体はどれも使っていない）。並べ方は仕様から書いていて、
//! ssh-keygen が暗号化したファイルを1バイトも違わず作り直せることをテストで確かめている。

// 読み込む側（ユニットテスト・IT）によって使う関数が違い、未使用が出るかどうかも読み込む側で変わるので
// expect ではなく allow
#![allow(dead_code, reason = "読み込む側によって使う関数が違う")]

use std::iter::repeat_n;
use std::mem::take;

pub type Result<T> = std::result::Result<T, String>;

// ---------------------------------------------------------------------------
// RFC 4648 Section 4: Base 64 Encoding
// ---------------------------------------------------------------------------

/// RFC 4648 Table 1: The Base 64 Alphabet
pub const B64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// 24 ビットずつ 6 ビット×4 文字にし、足りない分は `=` で埋める
pub fn b64_encode(data: &[u8]) -> String {
    let mut out = String::new();
    for group in data.chunks(3) {
        let mut buf = [0u8; 3];
        buf[..group.len()].copy_from_slice(group);
        let n = u32::from(buf[0]) << 16 | u32::from(buf[1]) << 8 | u32::from(buf[2]);
        let chars = [18, 12, 6, 0].map(|shift| B64_ALPHABET[(n >> shift & 0x3f) as usize] as char);
        // 1 バイトなら 2 文字 + "=="、2 バイトなら 3 文字 + "="
        let keep = group.len() + 1;
        out.extend(&chars[..keep]);
        out.extend(repeat_n('=', 4 - keep));
    }
    out
}

/// 厳密な復号: 長さは 4 の倍数、文字表外の文字や途中の `=` は拒否、
/// 余りビットは 0（RFC 4648 Section 3.5 の canonical encoding）であること
pub fn b64_decode(s: &str) -> Result<Vec<u8>> {
    let s = s.as_bytes();
    if !s.len().is_multiple_of(4) {
        return Err(format!("長さ {} が 4 の倍数でない", s.len()));
    }
    let pad = s.iter().rev().take_while(|&&c| c == b'=').count();
    if pad > 2 {
        return Err("= が3個以上".into());
    }
    let value = |c: u8| {
        B64_ALPHABET
            .iter()
            .position(|&a| a == c)
            .map(|v| v as u32)
            .ok_or_else(|| format!("文字表に無い文字 {:?}", c as char))
    };
    let mut out = Vec::new();
    for (i, group) in s.chunks(4).enumerate() {
        let last = i == s.len() / 4 - 1;
        let pad_here = if last { pad } else { 0 };
        let mut n = 0u32;
        for (j, &c) in group.iter().enumerate() {
            let v = if j >= 4 - pad_here { 0 } else { value(c)? };
            n = n << 6 | v;
        }
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        let keep = 3 - pad_here;
        // 捨てるビットが 0 でなければ canonical ではない
        if bytes[keep..].iter().any(|&b| b != 0) {
            return Err("パディング前の余りビットが 0 でない".into());
        }
        out.extend_from_slice(&bytes[..keep]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// RFC 4251 Section 5: Data Type Representations Used in the SSH Protocols
// ---------------------------------------------------------------------------

/// `uint32`: 4 バイト、上位から（network byte order）
pub fn put_uint32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}

/// `string`: uint32 の長さ + その長さのバイト列（終端の NUL は付けない）
pub fn put_string(buf: &mut Vec<u8>, s: &[u8]) {
    put_uint32(buf, s.len().try_into().unwrap());
    buf.extend_from_slice(s);
}

/// RFC 4251 の型を前から順に読む
pub struct Reader<'a> {
    buf: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.buf.len() < n {
            return Err(format!(
                "{n} バイト読もうとしたが残り {} バイト",
                self.buf.len()
            ));
        }
        let (head, rest) = self.buf.split_at(n);
        self.buf = rest;
        Ok(head)
    }

    pub fn uint32(&mut self) -> Result<u32> {
        let b = self.bytes(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn string(&mut self) -> Result<&'a [u8]> {
        let len = self.uint32()? as usize;
        self.bytes(len)
    }

    pub fn rest(&mut self) -> &'a [u8] {
        take(&mut self.buf)
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

// ---------------------------------------------------------------------------
// RFC 8709 Section 4: Public Key Format
// ---------------------------------------------------------------------------

/// `string "ssh-ed25519"`, `string key`（key は RFC 8032 Section 5.1.5 の 32 バイト）
pub fn parse_public_blob(blob: &[u8]) -> Result<[u8; 32]> {
    let mut r = Reader::new(blob);
    let key_type = r.string()?;
    if key_type != b"ssh-ed25519" {
        return Err(format!(
            "鍵の種類が {:?}",
            String::from_utf8_lossy(key_type)
        ));
    }
    let key = r.string()?;
    let key: [u8; 32] = key
        .try_into()
        .map_err(|_| format!("公開鍵が {} バイト（32 バイトのはず）", key.len()))?;
    if !r.is_empty() {
        return Err("blob の後ろに余計なデータ".into());
    }
    Ok(key)
}

pub fn public_blob(key: &[u8; 32]) -> Vec<u8> {
    let mut b = Vec::new();
    put_string(&mut b, b"ssh-ed25519");
    put_string(&mut b, key);
    b
}

// ---------------------------------------------------------------------------
// sshd(8) AUTHORIZED_KEYS FILE FORMAT
// ---------------------------------------------------------------------------

/// `.pub` の1行（オプションなし）: `keytype base64-encoded-key [comment]`
#[derive(Debug, PartialEq, Eq)]
pub struct AuthorizedKey {
    pub key_type: String,
    pub public_key: [u8; 32],
    pub comment: Option<String>,
}

pub fn parse_authorized_key(line: &str) -> Result<AuthorizedKey> {
    if line.contains(['\n', '\r']) {
        return Err("1行に改行が含まれている".into());
    }
    let mut parts = line.splitn(3, ' ');
    let key_type = parts.next().unwrap_or_default().to_owned();
    let b64 = parts.next().ok_or("base64 部分が無い")?;
    let comment = parts.next().map(str::to_owned);
    let blob = b64_decode(b64)?;
    let public_key = parse_public_blob(&blob)?;
    // 行の種類と blob の中の種類は一致していなければならない
    if key_type != "ssh-ed25519" {
        return Err(format!("行の鍵の種類が {key_type:?}"));
    }
    Ok(AuthorizedKey {
        key_type,
        public_key,
        comment,
    })
}

/// sshd(8) に載っている鍵の種類（行の先頭がこれなら、オプションの欄は無い）
pub const KEY_TYPES: [&str; 8] = [
    "sk-ecdsa-sha2-nistp256@openssh.com",
    "ecdsa-sha2-nistp256",
    "ecdsa-sha2-nistp384",
    "ecdsa-sha2-nistp521",
    "sk-ssh-ed25519@openssh.com",
    "ssh-ed25519",
    "ssh-dss",
    "ssh-rsa",
];

/// オプション1つ: `name` または `name="value"`（value は `\"` を `"` に戻したもの）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyOption {
    pub name: String,
    pub value: Option<String>,
}

impl KeyOption {
    pub fn flag(name: &str) -> Self {
        Self {
            name: name.into(),
            value: None,
        }
    }

    pub fn value(name: &str, value: &str) -> Self {
        Self {
            name: name.into(),
            value: Some(value.into()),
        }
    }
}

/// authorized_keys の鍵の行: `[options] keytype base64-encoded-key [comment]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizedKeysLine {
    /// ファイルの何行目か（1 始まり。コメント行・空行も数える）
    pub line_number: usize,
    pub options: Vec<KeyOption>,
    pub key_type: String,
    /// base64 を復号した公開鍵 blob（RFC 4253 Section 6.6）
    pub blob: Vec<u8>,
    pub comment: Option<String>,
}

impl AuthorizedKeysLine {
    /// `ssh-ed25519` の行なら、RFC 8709 の 32 バイトの公開鍵
    pub fn ed25519(&self) -> Result<[u8; 32]> {
        parse_public_blob(&self.blob)
    }
}

/// authorized_keys ファイル全体: 空行と `#` で始まる行は飛ばし、残りの行を
/// オプション・鍵の種類・base64・コメントに分ける
pub fn parse_authorized_keys(text: &str) -> Result<Vec<AuthorizedKeysLine>> {
    let mut lines = Vec::new();
    for (i, line) in text.split('\n').enumerate() {
        let line_number = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = |e: String| format!("{line_number} 行目: {e}");
        let first = line.split(' ').next().unwrap_or_default();
        let (options, rest) = if KEY_TYPES.contains(&first) {
            (Vec::new(), line)
        } else {
            parse_key_options(line).map_err(at)?
        };
        let mut parts = rest.splitn(3, ' ');
        let key_type = parts.next().unwrap_or_default().to_owned();
        if !KEY_TYPES.contains(&key_type.as_str()) {
            return Err(at(format!("鍵の種類 {key_type:?} は sshd(8) に無い")));
        }
        let b64 = parts.next().ok_or_else(|| at("base64 部分が無い".into()))?;
        let blob = b64_decode(b64).map_err(at)?;
        // RFC 4253 Section 6.6: blob は鍵の種類の string で始まり、行の種類と一致する
        let inner = Reader::new(&blob).string().map_err(at)?;
        if inner != key_type.as_bytes() {
            return Err(at(format!(
                "行の種類 {key_type:?} と blob の種類 {:?} が違う",
                String::from_utf8_lossy(inner)
            )));
        }
        let entry = AuthorizedKeysLine {
            line_number,
            options,
            key_type,
            blob,
            comment: parts.next().map(str::to_owned),
        };
        if entry.key_type == "ssh-ed25519" {
            entry.ed25519().map_err(at)?;
        }
        lines.push(entry);
    }
    Ok(lines)
}

/// 行頭のオプションの欄を読み、残り（鍵の種類から後ろ）を返す。
/// sshd(8): カンマ区切り、ダブルクォートの中以外に空白は入らない、クォートの中の `"` は `\"` と書く
fn parse_key_options(line: &str) -> Result<(Vec<KeyOption>, &str)> {
    let mut options = Vec::new();
    let mut chars = line.char_indices().peekable();
    loop {
        let mut name = String::new();
        while let Some(&(_, c)) = chars.peek()
            && !matches!(c, ',' | ' ' | '=' | '"')
        {
            name.push(c);
            chars.next();
        }
        if name.is_empty() {
            return Err("オプションの名前が空".into());
        }
        let mut value = None;
        if chars.next_if(|&(_, c)| c == '=').is_some() {
            if chars.next_if(|&(_, c)| c == '"').is_none() {
                return Err(format!("{name}= の値がダブルクォートで囲まれていない"));
            }
            let mut v = String::new();
            loop {
                match chars.next() {
                    None => return Err(format!("{name} の値のクォートが閉じていない")),
                    Some((_, '"')) => break,
                    Some((_, '\\')) if chars.next_if(|&(_, c)| c == '"').is_some() => v.push('"'),
                    Some((_, c)) => v.push(c),
                }
            }
            value = Some(v);
        }
        options.push(KeyOption { name, value });
        match chars.next() {
            Some((_, ',')) => continue,
            Some((i, ' ')) => return Ok((options, &line[i + 1..])),
            Some((_, c)) => return Err(format!("オプションの後ろに {c:?}")),
            None => return Err("オプションの後ろに鍵が無い".into()),
        }
    }
}

/// [`parse_key_options`] の逆: オプションの欄の文字列（値は `"` を `\"` にしてクォートで囲む）
pub fn format_key_options(options: &[KeyOption]) -> String {
    options
        .iter()
        .map(|o| match &o.value {
            None => o.name.clone(),
            Some(v) => format!("{}=\"{}\"", o.name, v.replace('"', "\\\"")),
        })
        .collect::<Vec<_>>()
        .join(",")
}

// ---------------------------------------------------------------------------
// OpenSSH PROTOCOL.key + draft-ietf-sshm-ssh-agent Section 3.2.3
// ---------------------------------------------------------------------------

pub const PEM_BEGIN: &str = "-----BEGIN OPENSSH PRIVATE KEY-----\n";
pub const PEM_END: &str = "-----END OPENSSH PRIVATE KEY-----\n";
/// PROTOCOL.key の AUTH_MAGIC（"openssh-key-v1" と終端の NUL）
pub const AUTH_MAGIC: &[u8] = b"openssh-key-v1\0";
/// OpenSSH は base64 を 70 文字ごとに折り返す（最後の行だけ短くてよい）
pub const PEM_LINE_WIDTH: usize = 70;

/// 秘密鍵リストの暗号。値は OpenSSH の cipher.c の表（ブロック長・鍵の長さ・IV の長さ・認証タグの長さ）。
/// IV の長さは cipher_ivlen() と同じく、表に無ければブロック長（chacha20-poly1305 は IV を使わない）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cipher {
    pub name: &'static str,
    /// PROTOCOL.key 3. のパディングはこの倍数にそろえる
    pub block_size: usize,
    pub key_len: usize,
    pub iv_len: usize,
    /// AEAD の認証タグの長さ。タグは秘密鍵リストの string の外（直後）に付く（sshkey.c の sshkey_private_to_blob2）
    pub auth_len: usize,
}

const fn c(
    name: &'static str,
    block_size: usize,
    key_len: usize,
    iv_len: usize,
    auth_len: usize,
) -> Cipher {
    Cipher {
        name,
        block_size,
        key_len,
        iv_len,
        auth_len,
    }
}

pub const CIPHER_NONE: Cipher = c("none", 8, 0, 0, 0);

/// パスフレーズで暗号化するときに使える方式（`ssh -Q cipher` に出るもの全部）
pub const CIPHERS: [Cipher; 10] = [
    c("3des-cbc", 8, 24, 8, 0),
    c("aes128-cbc", 16, 16, 16, 0),
    c("aes192-cbc", 16, 24, 16, 0),
    c("aes256-cbc", 16, 32, 16, 0),
    c("aes128-ctr", 16, 16, 16, 0),
    c("aes192-ctr", 16, 24, 16, 0),
    c("aes256-ctr", 16, 32, 16, 0),
    c("aes128-gcm@openssh.com", 16, 16, 12, 16),
    c("aes256-gcm@openssh.com", 16, 32, 12, 16),
    c("chacha20-poly1305@openssh.com", 8, 64, 0, 16),
];

pub fn cipher(name: &[u8]) -> Result<Cipher> {
    std::iter::once(CIPHER_NONE)
        .chain(CIPHERS)
        .find(|c| c.name.as_bytes() == name)
        .ok_or_else(|| format!("未知の暗号 {:?}", String::from_utf8_lossy(name)))
}

impl Cipher {
    /// KDF の出力（鍵 || IV）で `data` をその場で暗号化し、認証タグ（AEAD でなければ空）を返す
    pub fn encrypt(self, key_iv: &[u8], data: &mut [u8]) -> Vec<u8> {
        self.crypt(key_iv, data, None).expect("暗号化は失敗しない")
    }

    /// その場で復号する。AEAD なら認証タグを確かめ、合わなければ `Err`
    pub fn decrypt(self, key_iv: &[u8], data: &mut [u8], tag: &[u8]) -> Result<()> {
        self.crypt(key_iv, data, Some(tag)).map(drop)
    }

    /// `tag` が None なら暗号化、Some なら復号
    fn crypt(self, key_iv: &[u8], data: &mut [u8], tag: Option<&[u8]>) -> Result<Vec<u8>> {
        use aes::cipher::{KeyIvInit, StreamCipher};
        use aes::{Aes128, Aes192, Aes256};
        if key_iv.len() != self.key_len + self.iv_len {
            return Err(format!("鍵 + IV が {} バイト", key_iv.len()));
        }
        if !data.len().is_multiple_of(self.block_size) {
            return Err(format!(
                "{} バイトはブロック長 {} の倍数でない",
                data.len(),
                self.block_size
            ));
        }
        let (key, iv) = key_iv.split_at(self.key_len);
        let decrypt = tag.is_some();
        match self.name {
            "none" => {}
            "3des-cbc" => cbc_crypt::<des::TdesEde3>(key, iv, data, decrypt),
            "aes128-cbc" => cbc_crypt::<Aes128>(key, iv, data, decrypt),
            "aes192-cbc" => cbc_crypt::<Aes192>(key, iv, data, decrypt),
            "aes256-cbc" => cbc_crypt::<Aes256>(key, iv, data, decrypt),
            // CTR は暗号化も復号も同じ（鍵ストリームとの XOR）。カウンタは IV 全体を 128 ビットのビッグエンディアンとして増やす
            "aes128-ctr" => {
                ctr::Ctr128BE::<Aes128>::new(key.into(), iv.into()).apply_keystream(data)
            }
            "aes192-ctr" => {
                ctr::Ctr128BE::<Aes192>::new(key.into(), iv.into()).apply_keystream(data)
            }
            "aes256-ctr" => {
                ctr::Ctr128BE::<Aes256>::new(key.into(), iv.into()).apply_keystream(data)
            }
            "aes128-gcm@openssh.com" => return gcm_crypt::<Aes128>(key, iv, data, tag),
            "aes256-gcm@openssh.com" => return gcm_crypt::<Aes256>(key, iv, data, tag),
            "chacha20-poly1305@openssh.com" => return chacha_poly_crypt(key, data, tag),
            other => return Err(format!("未対応の暗号 {other}")),
        }
        Ok(Vec::new())
    }
}

fn cbc_crypt<C>(key: &[u8], iv: &[u8], data: &mut [u8], decrypt: bool)
where
    C: aes::cipher::BlockCipher
        + aes::cipher::BlockEncryptMut
        + aes::cipher::BlockDecryptMut
        + aes::cipher::KeyInit,
{
    use aes::cipher::KeyIvInit;
    use aes::cipher::block_padding::NoPadding;
    use aes::cipher::{BlockDecryptMut, BlockEncryptMut};
    let len = data.len();
    if decrypt {
        cbc::Decryptor::<C>::new_from_slices(key, iv)
            .unwrap()
            .decrypt_padded_mut::<NoPadding>(data)
            .unwrap();
    } else {
        cbc::Encryptor::<C>::new_from_slices(key, iv)
            .unwrap()
            .encrypt_padded_mut::<NoPadding>(data, len)
            .unwrap();
    }
}

/// AES-GCM（RFC 5647 の OpenSSH 版）。鍵ファイルでは IV（12 バイト）をそのままナンスにし、追加データは無い
fn gcm_crypt<C>(key: &[u8], iv: &[u8], data: &mut [u8], tag: Option<&[u8]>) -> Result<Vec<u8>>
where
    C: aes::cipher::BlockCipher
        + aes::cipher::BlockSizeUser<BlockSize = aes::cipher::consts::U16>
        + aes::cipher::BlockEncrypt
        + aes::cipher::KeyInit,
{
    use aes_gcm::aead::{AeadInPlace, KeyInit};
    let gcm = aes_gcm::AesGcm::<C, aes::cipher::consts::U12>::new_from_slice(key).unwrap();
    let nonce = aes_gcm::Nonce::<aes::cipher::consts::U12>::from_slice(iv);
    match tag {
        None => Ok(gcm
            .encrypt_in_place_detached(nonce, b"", data)
            .unwrap()
            .to_vec()),
        Some(tag) => gcm
            .decrypt_in_place_detached(nonce, b"", data, tag.into())
            .map(|()| Vec::new())
            .map_err(|_| "AES-GCM の認証タグが合わない".into()),
    }
}

/// PROTOCOL.chacha20poly1305: 鍵の前半 32 バイトが本文用（K_2）。ナンスは通番（鍵ファイルでは 0）の
/// 64 ビットのビッグエンディアン。ブロック 0 の鍵ストリームの先頭 32 バイトが Poly1305 の鍵で、
/// 本文はブロック 1 から暗号化し、タグは暗号文（鍵ファイルでは長さの欄が無いので本文だけ）の Poly1305
fn chacha_poly_crypt(key: &[u8], data: &mut [u8], tag: Option<&[u8]>) -> Result<Vec<u8>> {
    use chacha20::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
    use poly1305::universal_hash::KeyInit;
    let main_key = &key[..32];
    let nonce = 0u64.to_be_bytes();
    let mut chacha = chacha20::ChaCha20Legacy::new(main_key.into(), &nonce.into());
    let mut poly_key = [0u8; 32];
    chacha.apply_keystream(&mut poly_key);
    let mac = |ciphertext: &[u8]| {
        poly1305::Poly1305::new(&poly_key.into())
            .compute_unpadded(ciphertext)
            .to_vec()
    };
    if let Some(tag) = tag
        && mac(data) != tag
    {
        return Err("chacha20-poly1305 の認証タグが合わない".into());
    }
    chacha.seek(64u64);
    chacha.apply_keystream(data);
    Ok(if tag.is_none() { mac(data) } else { Vec::new() })
}

/// PROTOCOL.key 2. の kdfoptions（`string salt`, `uint32 rounds` を連結して string にしたものの中身）
pub fn bcrypt_kdfoptions(salt: &[u8], rounds: u32) -> Vec<u8> {
    let mut o = Vec::new();
    put_string(&mut o, salt);
    put_uint32(&mut o, rounds);
    o
}

/// kdfoptions を salt と rounds に分ける
pub fn parse_bcrypt_kdfoptions(kdfoptions: &[u8]) -> Result<(Vec<u8>, u32)> {
    let mut r = Reader::new(kdfoptions);
    let salt = r.string()?.to_vec();
    let rounds = r.uint32()?;
    if !r.is_empty() {
        return Err("kdfoptions の後ろに余計なデータ".into());
    }
    Ok((salt, rounds))
}

/// 秘密鍵ファイルを仕様の各フィールドに分解したもの（暗号化されていれば復号した中身）
#[derive(Debug, Clone)]
pub struct PrivateKeyFile {
    /// base64 の各行の長さ
    pub line_lengths: Vec<usize>,
    pub ciphername: Vec<u8>,
    pub kdfname: Vec<u8>,
    pub kdfoptions: Vec<u8>,
    pub public_keys: Vec<Vec<u8>>,
    pub checkint1: u32,
    pub checkint2: u32,
    pub keys: Vec<Ed25519PrivateKey>,
    pub padding: Vec<u8>,
    /// 暗号化前の秘密鍵リスト全体の長さ（ブロック長の倍数のはず）
    pub private_list_len: usize,
    /// AEAD の認証タグ（秘密鍵リストの string の直後）
    pub tag: Vec<u8>,
}

/// draft-ietf-sshm-ssh-agent 3.2.3: `string "ssh-ed25519"`, `string ENC(A)`, `string k || ENC(A)`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ed25519PrivateKey {
    pub enc_a: [u8; 32],
    pub k: [u8; 32],
    pub k_enc_a_tail: [u8; 32],
    pub comment: Vec<u8>,
}

impl Ed25519PrivateKey {
    /// draft 3.2.3 のとおりに書いた中身（鍵の種類から `k || ENC(A)` まで。コメントは含まない）
    pub fn agent_encoding(&self) -> Vec<u8> {
        let mut b = Vec::new();
        put_string(&mut b, b"ssh-ed25519");
        put_string(&mut b, &self.enc_a);
        put_string(&mut b, &[self.k, self.k_enc_a_tail].concat());
        b
    }
}

/// 暗号化されていない秘密鍵ファイルを読む（暗号化されていれば `Err`）
pub fn parse_private_key_file(pem: &str) -> Result<PrivateKeyFile> {
    let file = decrypt_private_key_file(pem, b"")?;
    if file.ciphername != b"none" {
        return Err(format!(
            "暗号化されている（{}）",
            String::from_utf8_lossy(&file.ciphername)
        ));
    }
    Ok(file)
}

/// 秘密鍵ファイルを読み、暗号化されていれば `passphrase` で復号する（PROTOCOL.key 4.: bcrypt で鍵 || IV を作る）。
/// パスフレーズが違うと、AEAD ならタグが合わず、そうでなければ2つの checkint が一致しないので `Err`
pub fn decrypt_private_key_file(pem: &str, passphrase: &[u8]) -> Result<PrivateKeyFile> {
    let body = pem
        .strip_circumfix(PEM_BEGIN, PEM_END)
        .ok_or("BEGIN / END 行が無い")?;
    let lines: Vec<&str> = body.split_terminator('\n').collect();
    let line_lengths = lines.iter().map(|l| l.len()).collect();
    let data = b64_decode(&lines.concat())?;

    let mut r = Reader::new(&data);
    if r.bytes(AUTH_MAGIC.len())? != AUTH_MAGIC {
        return Err("AUTH_MAGIC が違う".into());
    }
    let ciphername = r.string()?.to_vec();
    let kdfname = r.string()?.to_vec();
    let kdfoptions = r.string()?.to_vec();
    let n = r.uint32()?;
    let public_keys = (0..n)
        .map(|_| r.string().map(<[u8]>::to_vec))
        .collect::<Result<Vec<_>>>()?;
    let mut private_list = r.string()?.to_vec();
    let cipher = cipher(&ciphername)?;
    let tag = r.bytes(cipher.auth_len)?.to_vec();
    if !r.is_empty() {
        return Err("秘密鍵リストの後ろに余計なデータ".into());
    }

    let encrypted = cipher != CIPHER_NONE;
    if encrypted {
        if kdfname != b"bcrypt" {
            return Err(format!(
                "暗号化されているのに kdfname が {:?}",
                String::from_utf8_lossy(&kdfname)
            ));
        }
        let (salt, rounds) = parse_bcrypt_kdfoptions(&kdfoptions)?;
        let mut key_iv = vec![0u8; cipher.key_len + cipher.iv_len];
        bcrypt_pbkdf::bcrypt_pbkdf(passphrase, &salt, rounds, &mut key_iv)
            .map_err(|e| format!("bcrypt_pbkdf: {e}"))?;
        cipher.decrypt(&key_iv, &mut private_list, &tag)?;
    }

    let mut p = Reader::new(&private_list);
    let checkint1 = p.uint32()?;
    let checkint2 = p.uint32()?;
    // PROTOCOL.key 3.: 復号できたかは2つの checkint が同じかで確かめる
    if encrypted && checkint1 != checkint2 {
        return Err("checkint が一致しない（パスフレーズが違う）".into());
    }
    let mut keys = Vec::new();
    for _ in 0..n {
        let key_type = p.string()?;
        if key_type != b"ssh-ed25519" {
            return Err(format!(
                "秘密鍵の種類が {:?}",
                String::from_utf8_lossy(key_type)
            ));
        }
        let enc_a: [u8; 32] = p
            .string()?
            .try_into()
            .map_err(|_| "ENC(A) が 32 バイトでない")?;
        let k_enc_a = p.string()?;
        if k_enc_a.len() != 64 {
            return Err(format!(
                "k || ENC(A) が {} バイト（64 のはず）",
                k_enc_a.len()
            ));
        }
        let comment = p.string()?.to_vec();
        keys.push(Ed25519PrivateKey {
            enc_a,
            k: *k_enc_a.first_chunk().expect("64 バイト"),
            k_enc_a_tail: *k_enc_a.last_chunk().expect("64 バイト"),
            comment,
        });
    }
    let padding = p.rest().to_vec();
    Ok(PrivateKeyFile {
        line_lengths,
        ciphername,
        kdfname,
        kdfoptions,
        public_keys,
        checkint1,
        checkint2,
        keys,
        padding,
        private_list_len: private_list.len(),
        tag,
    })
}

impl PrivateKeyFile {
    /// 1つの Ed25519 鍵のファイル。パディングは `cipher` のブロック長にそろえる最小のもの。
    /// 暗号化するなら `kdfoptions` は [`bcrypt_kdfoptions`]（暗号化しないなら空）
    pub fn new(cipher: Cipher, kdfoptions: Vec<u8>, key: Ed25519PrivateKey, checkint: u32) -> Self {
        // checkint 2つ（8）+ 鍵の中身 + コメント（4 + 長さ）
        let len = 8 + key.agent_encoding().len() + 4 + key.comment.len();
        let pad = len.next_multiple_of(cipher.block_size) - len;
        Self {
            line_lengths: Vec::new(),
            ciphername: cipher.name.into(),
            kdfname: if cipher == CIPHER_NONE {
                "none"
            } else {
                "bcrypt"
            }
            .into(),
            kdfoptions,
            public_keys: vec![public_blob(&key.enc_a)],
            checkint1: checkint,
            checkint2: checkint,
            keys: vec![key],
            padding: (1..=pad as u8).collect(),
            private_list_len: len + pad,
            tag: Vec::new(),
        }
    }

    /// 各フィールドをそのまま並べて PEM にする（[`decrypt_private_key_file`] の逆）。
    /// 暗号化するなら `passphrase` と kdfoptions の salt・rounds で鍵を作る。
    /// フィールドを壊してから書けば、仕様に違反したファイルも作れる
    pub fn encode(&self, passphrase: &[u8]) -> String {
        let mut private = Vec::new();
        put_uint32(&mut private, self.checkint1);
        put_uint32(&mut private, self.checkint2);
        for key in &self.keys {
            private.extend(key.agent_encoding());
            put_string(&mut private, &key.comment);
        }
        private.extend(&self.padding);

        let cipher = cipher(&self.ciphername).unwrap();
        let mut tag = Vec::new();
        if cipher != CIPHER_NONE {
            let (salt, rounds) = parse_bcrypt_kdfoptions(&self.kdfoptions).unwrap();
            let mut key_iv = vec![0u8; cipher.key_len + cipher.iv_len];
            bcrypt_pbkdf::bcrypt_pbkdf(passphrase, &salt, rounds, &mut key_iv).unwrap();
            tag = cipher.encrypt(&key_iv, &mut private);
        }

        let mut data = AUTH_MAGIC.to_vec();
        put_string(&mut data, &self.ciphername);
        put_string(&mut data, &self.kdfname);
        put_string(&mut data, &self.kdfoptions);
        put_uint32(&mut data, self.public_keys.len().try_into().unwrap());
        for blob in &self.public_keys {
            put_string(&mut data, blob);
        }
        put_string(&mut data, &private);
        data.extend(tag);
        let b64 = b64_encode(&data);
        let lines: Vec<&str> = b64
            .as_bytes()
            .chunks(PEM_LINE_WIDTH)
            .map(|l| std::str::from_utf8(l).unwrap())
            .collect();
        format!("{PEM_BEGIN}{}\n{PEM_END}", lines.join("\n"))
    }

    /// 1鍵のファイルとして仕様どおりかをまとめて確認し、違反をすべて返す
    pub fn violations(&self) -> Vec<String> {
        let mut v = Vec::new();
        let mut check = |ok: bool, msg: &str| {
            if !ok {
                v.push(msg.to_owned());
            }
        };
        let (last, rest) = self.line_lengths.split_last().unwrap_or((&0, &[]));
        check(
            rest.iter().all(|&n| n == PEM_LINE_WIDTH),
            "最終行以外が 70 文字でない",
        );
        check(
            (1..=PEM_LINE_WIDTH).contains(last),
            "最終行の長さが 1〜70 でない",
        );
        let cipher = cipher(&self.ciphername).unwrap_or(CIPHER_NONE);
        if cipher == CIPHER_NONE {
            // PROTOCOL.key 5. No encryption
            check(self.ciphername == b"none", "ciphername が none でない");
            check(self.kdfname == b"none", "kdfname が none でない");
            check(self.kdfoptions.is_empty(), "kdfoptions が空でない");
        } else {
            // PROTOCOL.key 2. KDF options for kdfname "bcrypt"
            check(self.kdfname == b"bcrypt", "kdfname が bcrypt でない");
            check(
                parse_bcrypt_kdfoptions(&self.kdfoptions).is_ok_and(|(_, rounds)| rounds > 0),
                "kdfoptions が salt と rounds でない",
            );
        }
        check(self.tag.len() == cipher.auth_len, "認証タグの長さが違う");
        check(
            self.public_keys.len() == 1 && self.keys.len() == 1,
            "鍵が1つでない",
        );
        // PROTOCOL.key 3. 2つの checkint は同じ値
        check(self.checkint1 == self.checkint2, "checkint が一致しない");
        // PROTOCOL.key 3. パディングは 1, 2, 3, ... で暗号のブロック長の倍数に揃える
        check(
            self.padding
                .iter()
                .enumerate()
                .all(|(i, &b)| b as usize == i + 1),
            "パディングが 1, 2, 3, ... でない",
        );
        check(
            self.private_list_len.is_multiple_of(cipher.block_size),
            "ブロック長の倍数でない",
        );
        check(
            self.padding.len() < cipher.block_size,
            "パディングが最小でない",
        );
        if let (Some(blob), Some(key)) = (self.public_keys.first(), self.keys.first()) {
            // ヘッダの公開鍵（RFC 8709）と秘密鍵の中の ENC(A) は同じ鍵
            check(
                parse_public_blob(blob) == Ok(key.enc_a),
                "ヘッダの公開鍵と ENC(A) が違う",
            );
            // agent draft 3.2.3: 2つ目の値は k || ENC(A)
            check(
                key.k_enc_a_tail == key.enc_a,
                "k || ENC(A) の後半が ENC(A) でない",
            );
        }
        v
    }
}

// ---------------------------------------------------------------------------
// OpenSSH PROTOCOL.sshsig + RFC 8709 Section 6（ssh-keygen -Y sign の署名）
// ---------------------------------------------------------------------------

pub const SSHSIG_BEGIN: &str = "-----BEGIN SSH SIGNATURE-----\n";
pub const SSHSIG_END: &str = "-----END SSH SIGNATURE-----";
/// PROTOCOL.sshsig の MAGIC_PREAMBLE
pub const SSHSIG_MAGIC: &[u8; 6] = b"SSHSIG";

/// PROTOCOL.sshsig 2. Blob format
#[derive(Debug)]
pub struct SshSig {
    pub version: u32,
    pub public_key: [u8; 32],
    pub namespace: Vec<u8>,
    pub reserved: Vec<u8>,
    pub hash_algorithm: Vec<u8>,
    /// RFC 8709 Section 6 の中身: RFC 8032 Section 5.1.6 の 64 バイトの署名
    pub signature: [u8; 64],
}

pub fn parse_sshsig(armored: &str) -> Result<SshSig> {
    let body = armored
        .trim_end_matches('\n')
        .strip_circumfix(SSHSIG_BEGIN, SSHSIG_END)
        .ok_or("BEGIN / END 行が無い")?;
    let blob = b64_decode(&body.split_terminator('\n').collect::<String>())?;

    let mut r = Reader::new(&blob);
    if r.bytes(6)? != SSHSIG_MAGIC {
        return Err("MAGIC_PREAMBLE が SSHSIG でない".into());
    }
    let version = r.uint32()?;
    let public_key = parse_public_blob(r.string()?)?;
    let namespace = r.string()?.to_vec();
    let reserved = r.string()?.to_vec();
    let hash_algorithm = r.string()?.to_vec();
    // RFC 8709 Section 6: string "ssh-ed25519", string signature（64 バイト）
    let mut s = Reader::new(r.string()?);
    if s.string()? != b"ssh-ed25519" {
        return Err("署名の種類が ssh-ed25519 でない".into());
    }
    let signature: [u8; 64] = s
        .string()?
        .try_into()
        .map_err(|_| "署名が 64 バイトでない")?;
    if !s.is_empty() || !r.is_empty() {
        return Err("署名 blob の後ろに余計なデータ".into());
    }
    Ok(SshSig {
        version,
        public_key,
        namespace,
        reserved,
        hash_algorithm,
        signature,
    })
}

/// PROTOCOL.sshsig の base64 の折り返し幅。仕様の文章は「76 文字ごと（SHOULD）」だが、
/// 仕様の例と ssh-keygen の出力は 70 文字で、ssh-key クレートは 70 文字でないと読めない
/// （76 文字だと `Base64(InvalidEncoding)`）。実際に使われている 70 文字に合わせる
pub const SSHSIG_LINE_WIDTH: usize = 70;

/// `parse_sshsig` の逆: PROTOCOL.sshsig 2. の blob を組み立て、1. の armored 形式にする
pub fn armor_sshsig(sig: &SshSig) -> String {
    let mut blob = SSHSIG_MAGIC.to_vec();
    put_uint32(&mut blob, sig.version);
    put_string(&mut blob, &public_blob(&sig.public_key));
    put_string(&mut blob, &sig.namespace);
    put_string(&mut blob, &sig.reserved);
    put_string(&mut blob, &sig.hash_algorithm);
    // RFC 8709 Section 6: string "ssh-ed25519", string signature
    let mut signature = Vec::new();
    put_string(&mut signature, b"ssh-ed25519");
    put_string(&mut signature, &sig.signature);
    put_string(&mut blob, &signature);

    let b64 = b64_encode(&blob);
    let lines: Vec<&str> = b64
        .as_bytes()
        .chunks(SSHSIG_LINE_WIDTH)
        .map(|l| std::str::from_utf8(l).unwrap())
        .collect();
    format!("{SSHSIG_BEGIN}{}\n{SSHSIG_END}\n", lines.join("\n"))
}

/// PROTOCOL.sshsig 3. Signed Data（これが Ed25519 で署名されるメッセージになる）
pub fn sshsig_signed_data(
    namespace: &[u8],
    reserved: &[u8],
    hash_algorithm: &[u8],
    message: &[u8],
) -> Vec<u8> {
    use sha2::Digest;
    let digest = match hash_algorithm {
        b"sha512" => sha2::Sha512::digest(message).to_vec(),
        b"sha256" => sha2::Sha256::digest(message).to_vec(),
        other => panic!("未対応のハッシュ {:?}", String::from_utf8_lossy(other)),
    };
    let mut d = SSHSIG_MAGIC.to_vec();
    put_string(&mut d, namespace);
    put_string(&mut d, reserved);
    put_string(&mut d, hash_algorithm);
    put_string(&mut d, &digest);
    d
}

/// OpenSSH の SHA256 フィンガープリント: "SHA256:" + base64(SHA-256(blob))（末尾の = は付けない）
pub fn sha256_fingerprint(blob: &[u8]) -> String {
    use sha2::Digest;
    format!(
        "SHA256:{}",
        b64_encode(&sha2::Sha256::digest(blob)).trim_end_matches('=')
    )
}
