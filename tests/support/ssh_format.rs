//! SSH 関連の形式を仕様書どおりに読み書きする、テスト専用の独立実装
//!
//! - RFC 4648 Section 4: base64（標準の文字表、`=` によるパディング）
//! - RFC 4251 Section 5: `uint32` / `string`
//! - RFC 8709 Section 4: `ssh-ed25519` 公開鍵 blob
//! - sshd(8) AUTHORIZED_KEYS FILE FORMAT: `.pub` の1行（`種類 base64 コメント`）
//! - OpenSSH PROTOCOL.key: 秘密鍵ファイル（openssh-key-v1）
//! - draft-ietf-sshm-ssh-agent Section 3.2.3: Ed25519 秘密鍵の中身（`ENC(A)`、`k || ENC(A)`）
//! - OpenSSH PROTOCOL.sshsig + RFC 8709 Section 6: `ssh-keygen -Y sign` の署名（armored 形式・blob・署名対象）
//!
//! meikey 本体と同じ部品（base64 クレートなど）は使わず、仕様の文章から直接書いている。
//! 本体と同じ部品で検証すると、同じ間違いを見逃すため。

#![allow(dead_code)] // 読み込む側によって使う関数が違う

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
        out.extend(std::iter::repeat_n('=', 4 - keep));
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
        std::mem::take(&mut self.buf)
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
// sshd(8) AUTHORIZED_KEYS FILE FORMAT（オプションなしの行）
// ---------------------------------------------------------------------------

/// `.pub` の1行: `keytype base64-encoded-key [comment]`
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

// ---------------------------------------------------------------------------
// OpenSSH PROTOCOL.key + draft-ietf-sshm-ssh-agent Section 3.2.3
// ---------------------------------------------------------------------------

pub const PEM_BEGIN: &str = "-----BEGIN OPENSSH PRIVATE KEY-----\n";
pub const PEM_END: &str = "-----END OPENSSH PRIVATE KEY-----\n";
/// PROTOCOL.key の AUTH_MAGIC（"openssh-key-v1" と終端の NUL）
pub const AUTH_MAGIC: &[u8] = b"openssh-key-v1\0";
/// cipher "none" のブロック長（OpenSSH の cipher.c）
pub const NONE_BLOCK_SIZE: usize = 8;

/// 秘密鍵ファイルを仕様の各フィールドに分解したもの
#[derive(Debug)]
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
}

/// draft-ietf-sshm-ssh-agent 3.2.3: `string "ssh-ed25519"`, `string ENC(A)`, `string k || ENC(A)`
#[derive(Debug)]
pub struct Ed25519PrivateKey {
    pub enc_a: [u8; 32],
    pub k: [u8; 32],
    pub k_enc_a_tail: [u8; 32],
    pub comment: Vec<u8>,
}

pub fn parse_private_key_file(pem: &str) -> Result<PrivateKeyFile> {
    let body = pem
        .strip_prefix(PEM_BEGIN)
        .and_then(|s| s.strip_suffix(PEM_END))
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
    let private_list = r.string()?;
    if !r.is_empty() {
        return Err("秘密鍵リストの後ろに余計なデータ".into());
    }

    let mut p = Reader::new(private_list);
    let checkint1 = p.uint32()?;
    let checkint2 = p.uint32()?;
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
            k: k_enc_a[..32].try_into().unwrap(),
            k_enc_a_tail: k_enc_a[32..].try_into().unwrap(),
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
    })
}

impl PrivateKeyFile {
    /// 暗号化なしの1鍵ファイルとして仕様どおりかをまとめて確認し、違反をすべて返す
    pub fn violations(&self) -> Vec<String> {
        let mut v = Vec::new();
        let mut check = |ok: bool, msg: &str| {
            if !ok {
                v.push(msg.to_owned());
            }
        };
        // OpenSSH は base64 を 70 文字ごとに折り返す（最後の行だけ短くてよい）
        let (last, rest) = self.line_lengths.split_last().unwrap_or((&0, &[]));
        check(rest.iter().all(|&n| n == 70), "最終行以外が 70 文字でない");
        check((1..=70).contains(last), "最終行の長さが 1〜70 でない");
        // PROTOCOL.key 5. No encryption
        check(self.ciphername == b"none", "ciphername が none でない");
        check(self.kdfname == b"none", "kdfname が none でない");
        check(self.kdfoptions.is_empty(), "kdfoptions が空でない");
        check(
            self.public_keys.len() == 1 && self.keys.len() == 1,
            "鍵が1つでない",
        );
        // PROTOCOL.key 3. 2つの checkint は同じ値
        check(self.checkint1 == self.checkint2, "checkint が一致しない");
        // PROTOCOL.key 3. パディングは 1, 2, 3, ... でブロック長の倍数に揃える
        check(
            self.padding
                .iter()
                .enumerate()
                .all(|(i, &b)| b as usize == i + 1),
            "パディングが 1, 2, 3, ... でない",
        );
        check(
            self.private_list_len.is_multiple_of(NONE_BLOCK_SIZE),
            "ブロック長の倍数でない",
        );
        check(
            self.padding.len() < NONE_BLOCK_SIZE,
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
        .strip_prefix(SSHSIG_BEGIN)
        .and_then(|s| s.trim_end_matches('\n').strip_suffix(SSHSIG_END))
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
