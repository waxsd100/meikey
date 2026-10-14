//! ed25519 鍵と、公開鍵の base64・authorized_keys の1行（秘密鍵のファイルへの書き出しは
//! [`crate::repository::key_file`]）

use crate::lang::tf;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use ed25519_dalek::SigningKey;

pub(crate) const KEY_TYPE: &[u8] = b"ssh-ed25519";

/// 公開鍵 blob（`string "ssh-ed25519"` + `string pubkey`）の長さ
pub(crate) const BLOB_LEN: usize = 4 + KEY_TYPE.len() + 4 + 32;

/// 公開鍵 blob の先頭（`string "ssh-ed25519"` + 公開鍵の長さ）。どの鍵でも同じ
const BLOB_HEADER: [u8; BLOB_LEN - 32] = {
    let mut h = [0u8; BLOB_LEN - 32];
    h[3] = KEY_TYPE.len() as u8;
    let mut i = 0;
    while i < KEY_TYPE.len() {
        h[4 + i] = KEY_TYPE[i];
        i += 1;
    }
    h[BLOB_LEN - 32 - 1] = 32;
    h
};

/// 公開鍵 blob を base64 にした長さ（51バイト → 68文字、パディングなし）
pub const PUBLIC_B64_LEN: usize = BLOB_LEN.div_ceil(3) * 4;

/// 公開鍵の base64 のうち、どの鍵でも同じになる先頭部分
pub const FIXED_PREFIX: &str = "AAAAC3NzaC1lZDI1NTE5AAAAI";

/// 公開鍵の base64（`ssh-ed25519 ` の後ろの部分）
pub type PublicB64 = [u8; PUBLIC_B64_LEN];

/// blob の先頭18バイト（3の倍数なので base64 の区切りと揃う）は鍵によらず同じ
const HEAD_BYTES: usize = 18;
const HEAD_B64: &[u8; HEAD_BYTES / 3 * 4] = b"AAAAC3NzaC1lZDI1NTE5AAAA";

/// 32バイトの公開鍵から base64 を作る（探索ループ用、アロケーションなし）。
/// 固定部分はコピーし、鍵ごとに変わる後半33バイトだけをエンコードする。
#[inline]
pub fn encode_public_b64(public: &[u8; 32], out: &mut PublicB64) {
    let mut tail = [0u8; BLOB_LEN - HEAD_BYTES];
    tail[..BLOB_HEADER.len() - HEAD_BYTES].copy_from_slice(&BLOB_HEADER[HEAD_BYTES..]);
    tail[BLOB_HEADER.len() - HEAD_BYTES..].copy_from_slice(public);
    out[..HEAD_B64.len()].copy_from_slice(HEAD_B64);
    let written = STANDARD
        .encode_slice(tail, &mut out[HEAD_B64.len()..])
        .expect("バッファ長は定数で一致している");
    debug_assert_eq!(HEAD_B64.len() + written, PUBLIC_B64_LEN);
}

/// コメントに使えない文字が入っている
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{}", tf!(
    "コメントに使えない文字: {:?}（改行・NUL は不可）",
    "invalid character in the comment: {:?} (no line breaks or NUL)",
    .0
))]
pub struct InvalidComment(pub char);

/// 鍵のコメントとして使えるか確かめる。
///
/// `.pub` は1鍵1行（sshd(8) AUTHORIZED_KEYS FILE FORMAT）なので、改行が入ると行が割れて
/// authorized_keys に貼れなくなる。NUL は C の文字列として読まれると途中で切れる。
pub fn check_comment(comment: &str) -> Result<(), InvalidComment> {
    match comment.chars().find(|c| matches!(c, '\n' | '\r' | '\0')) {
        Some(c) => Err(InvalidComment(c)),
        None => Ok(()),
    }
}

/// 32バイトのシードから作る ed25519 鍵
#[derive(Clone, PartialEq, Eq)]
pub struct VanityKey {
    seed: [u8; 32],
    public: [u8; 32],
}

// 使い終わった秘密鍵（シード）をメモリに残さない
impl Drop for VanityKey {
    fn drop(&mut self) {
        zeroize::Zeroize::zeroize(&mut self.seed);
    }
}

impl std::fmt::Debug for VanityKey {
    // 秘密鍵をログに出さない
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VanityKey")
            .field("public", &self.public_openssh(""))
            .finish_non_exhaustive()
    }
}

impl VanityKey {
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let public = SigningKey::from_bytes(&seed).verifying_key().to_bytes();
        Self { seed, public }
    }

    /// 秘密鍵（シード）。秘密鍵のファイルを書くときにだけ使う
    pub(crate) fn seed(&self) -> &[u8; 32] {
        &self.seed
    }

    pub(crate) fn public_bytes(&self) -> &[u8; 32] {
        &self.public
    }

    /// 公開鍵 blob（`string "ssh-ed25519"` + `string pubkey`）
    pub(crate) fn public_blob(&self) -> [u8; BLOB_LEN] {
        let mut blob = [0u8; BLOB_LEN];
        blob[..BLOB_HEADER.len()].copy_from_slice(&BLOB_HEADER);
        blob[BLOB_HEADER.len()..].copy_from_slice(&self.public);
        blob
    }

    pub fn write_public_b64(&self, out: &mut PublicB64) {
        encode_public_b64(&self.public, out);
    }

    pub fn public_b64(&self) -> PublicB64 {
        let mut out = [0u8; PUBLIC_B64_LEN];
        self.write_public_b64(&mut out);
        out
    }

    /// `authorized_keys` に書く1行（`ssh-ed25519 AAAA... comment`）。
    /// コメントは [`check_comment`] を通ったものを渡すこと
    pub fn public_openssh(&self, comment: &str) -> String {
        let b64 = self.public_b64();
        let b64 = std::str::from_utf8(&b64).expect("base64 は ASCII");
        match comment {
            "" => format!("ssh-ed25519 {b64}"),
            c => format!("ssh-ed25519 {b64} {c}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::check;
    use crate::rfc8032::vector;
    use crate::ssh_format::b64_decode;
    use crate::ssh_format::b64_encode;
    use crate::ssh_format::parse_authorized_key;
    use crate::ssh_format::parse_public_blob;
    use crate::ssh_format::public_blob;
    use proptest::prelude::*;
    use rstest::rstest;
    use similar_asserts::assert_eq;
    use std::str::from_utf8;

    // 公開鍵の行・base64 が仕様どおりかは、下の `conformance` の proptest で独立実装と突き合わせている。
    // ここではこのモジュールの細部だけを確かめる。

    /// 公開鍵の1行の完成形（RFC 8032 TEST 1 の鍵）。ssh-key クレートでも同じ鍵として読める
    #[rstest]
    #[case::no_comment(
        "",
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINdamAGCsQq31Uv+08lkBzoO4XLz2qYjJa8CGmj3B1Ea"
    )]
    #[case::comment(
        "wakokara",
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINdamAGCsQq31Uv+08lkBzoO4XLz2qYjJa8CGmj3B1Ea wakokara"
    )]
    fn public_openssh_line(#[case] comment: &str, #[case] want: &str) {
        let key = VanityKey::from_seed(vector("1").secret);
        let line = key.public_openssh(comment);
        assert_eq!(line, want);
        let parsed = ssh_key::PublicKey::from_openssh(&line).unwrap();
        assert_eq!(parsed.comment(), comment);
        assert_eq!(&parsed.key_data().ed25519().unwrap().0, key.public_bytes());
    }

    /// 公開鍵 blob の先頭（RFC 4251 Section 5 の string 2つ: RFC 8709 Section 4）
    #[rstest]
    #[case::type_len(0..4, &[0, 0, 0, 11])]
    #[case::type_name(4..15, b"ssh-ed25519")]
    #[case::key_len(15..19, &[0, 0, 0, 32])]
    fn blob_header_fields(#[case] range: std::ops::Range<usize>, #[case] want: &[u8]) {
        assert_eq!(&BLOB_HEADER[range], want);
    }

    /// コメントの検査。使えない文字なら、その文字とエラーメッセージ
    #[rstest]
    #[case::empty("", None)]
    #[case::plain("wakokara@example", None)]
    #[case::spaces_and_tabs("my key\tfor work", None)]
    #[case::japanese("和光から", None)]
    #[case::lf("wako\nkara", Some(('\n', "コメントに使えない文字: '\\n'（改行・NUL は不可）")))]
    #[case::crlf("wako\r\nkara", Some(('\r', "コメントに使えない文字: '\\r'（改行・NUL は不可）")))]
    #[case::nul("wako\0kara", Some(('\0', "コメントに使えない文字: '\\0'（改行・NUL は不可）")))]
    fn check_comment_cases(#[case] comment: &str, #[case] bad: Option<(char, &str)>) {
        match bad {
            None => assert_eq!(check_comment(comment), Ok(())),
            Some((c, message)) => {
                let e = check_comment(comment).unwrap_err();
                assert_eq!(e, InvalidComment(c));
                assert_eq!(e.to_string(), message);
            }
        }
    }

    /// Debug 表示は公開鍵だけで、どんな鍵でもシード（秘密鍵）が出ない
    #[rstest]
    fn debug_never_shows_seed() {
        check(1024, any::<[u8; 32]>(), |seed| {
            let dbg = format!("{:?}", VanityKey::from_seed(seed));
            let hex: String = seed.iter().map(|b| format!("{b:02x}")).collect();
            let seed_debug = format!("{seed:?}");
            prop_assert!(dbg.contains("ssh-ed25519"));
            prop_assert!(!dbg.contains("seed"));
            prop_assert!(!dbg.contains(&hex));
            prop_assert!(!dbg.contains(&seed_debug));
            Ok(())
        });
    }

    /// 検査を通るコメントなら `.pub` は必ず1行の正しい authorized_keys 形式になる。
    /// 改行か NUL を含むものは必ず拒否し、最初に出てくるその文字を返す
    #[rstest]
    #[case::without_line_breaks("[^\r\n\u{0}]{0,40}", true)]
    #[case::with_line_break_or_nul("[^\r\n\u{0}]{0,20}[\r\n\u{0}][^\r\n\u{0}]{0,20}", false)]
    fn comment_check_keeps_pub_on_one_line(#[case] comments: &'static str, #[case] accepted: bool) {
        check(1024, (comments, any::<[u8; 32]>()), |(comment, seed)| {
            let line = VanityKey::from_seed(seed).public_openssh(&comment);
            match check_comment(&comment) {
                Ok(()) => {
                    prop_assert!(accepted, "{:?} を受け付けた", comment);
                    let parsed = parse_authorized_key(&line).map_err(TestCaseError::fail)?;
                    prop_assert_eq!(parsed.comment.unwrap_or_default(), comment);
                }
                Err(InvalidComment(c)) => {
                    prop_assert!(!accepted, "{:?} を拒否した", comment);
                    let first = comment.chars().find(|c| matches!(c, '\n' | '\r' | '\0'));
                    prop_assert_eq!(Some(c), first);
                }
            }
            Ok(())
        });
    }

    /// RFC・仕様への準拠（正解は tests/support/ の独立実装）: 公開鍵の1行（RFC 8709・RFC 4648・
    /// authorized_keys）。秘密鍵ファイルは [`crate::repository::key_file`] のテスト
    mod conformance {
        use super::*;
        use similar_asserts::assert_eq;

        // ===========================================================================
        // RFC 8709 / RFC 4648 / authorized_keys: 公開鍵の1行
        // ===========================================================================

        /// `.pub` の1行が仕様どおり。base64 部分は RFC 4648 の独立実装の出力と1文字も違わない
        #[rstest]
        #[case::no_comment("")]
        #[case::comment("[^\r\n\u{0}]{1,80}")] // 改行・NUL 以外の任意の Unicode
        fn public_line_follows_spec(#[case] comments: &'static str) {
            check(256, (any::<[u8; 32]>(), comments), |(seed, comment)| {
                let key = VanityKey::from_seed(seed);
                let line = key.public_openssh(&comment);
                let parsed = parse_authorized_key(&line).map_err(TestCaseError::fail)?;
                prop_assert_eq!(&parsed.public_key, key.public_bytes());
                prop_assert_eq!(
                    parsed.comment,
                    (!comment.is_empty()).then(|| comment.clone())
                );
                let blob = public_blob(key.public_bytes());
                prop_assert_eq!(line.split(' ').nth(1).unwrap(), b64_encode(&blob));
                Ok(())
            });
        }

        /// 探索ループ用の `encode_public_b64` も、RFC 4648 + RFC 8709 の独立実装と一致する
        #[rstest]
        fn hot_loop_encoding_follows_spec() {
            check(256, any::<[u8; 32]>(), |public| {
                let mut b64 = [0u8; PUBLIC_B64_LEN];
                encode_public_b64(&public, &mut b64);
                let want = b64_encode(&public_blob(&public));
                prop_assert_eq!(from_utf8(&b64).unwrap(), want.as_str());
                prop_assert_eq!(
                    b64_decode(&want).map(|b| parse_public_blob(&b)),
                    Ok(Ok(public))
                );
                Ok(())
            });
        }

        /// 固定部分と「26文字目は A〜P」は、RFC 4648 のビット配置から導ける
        #[rstest]
        #[case::zero(0x00, 'A')]
        #[case::max_of_first_nibble(0x0f, 'A')]
        #[case::second(0x10, 'B')]
        #[case::max(0xff, 'P')]
        fn char_26_comes_from_high_nibble_of_first_key_byte(#[case] first: u8, #[case] want: char) {
            let mut public = [0x5a; 32];
            public[0] = first;
            let b64 = b64_encode(&public_blob(&public));
            assert_eq!(&b64[..FIXED_PREFIX.len()], FIXED_PREFIX);
            // blob の 18 バイト目（鍵長 0x20 の最下位バイト）の下位2ビット = 0 と、鍵の先頭バイトの上位4ビット
            assert_eq!(b64.as_bytes()[FIXED_PREFIX.len()] as char, want);
        }
    }
}
