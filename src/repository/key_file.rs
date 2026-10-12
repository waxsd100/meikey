//! 鍵を OpenSSH 形式のファイルに保存する
//!
//! 秘密鍵は暗号化なしの openssh-key-v1（PROTOCOL.key）、公開鍵は authorized_keys の1行（`.pub`）。

use crate::model::key::{KEY_TYPE, VanityKey};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// 鍵の保存先。controller はこの trait 越しに保存するので、テストでは mockall の
/// `MockKeyRepository` に差し替えて、ファイルに触れずに呼ばれ方を確かめられる
#[cfg_attr(test, mockall::automock)]
pub trait KeyRepository {
    /// 秘密鍵と公開鍵のうち、既にあるほうのパス（両方なければ None）
    fn existing(&self, private: &Path) -> io::Result<Option<PathBuf>>;

    /// 秘密鍵を `private`、公開鍵を `<private>.pub` に書く。
    /// `overwrite` でなければ、どちらかが既にあると書かずに失敗する（[`io::ErrorKind::AlreadyExists`]）
    fn save(
        &self,
        key: &VanityKey,
        comment: &str,
        private: &Path,
        overwrite: bool,
    ) -> io::Result<()>;
}

/// ファイルシステムに OpenSSH 形式で保存する [`KeyRepository`]（本番で使うもの）
#[derive(Debug, Clone, Copy, Default)]
pub struct KeyFiles;

impl KeyRepository for KeyFiles {
    fn existing(&self, private: &Path) -> io::Result<Option<PathBuf>> {
        existing(private)
    }

    fn save(
        &self,
        key: &VanityKey,
        comment: &str,
        private: &Path,
        overwrite: bool,
    ) -> io::Result<()> {
        save(key, comment, private, overwrite)
    }
}

/// 秘密鍵 `private` と組になる公開鍵のパス（`<private>.pub`）
pub fn pub_path(private: &Path) -> PathBuf {
    private.with_added_extension("pub")
}

/// 秘密鍵と公開鍵のうち、既にあるほうのパス（両方なければ None）
pub fn existing(private: &Path) -> io::Result<Option<PathBuf>> {
    for p in [private.to_path_buf(), pub_path(private)] {
        if p.try_exists()? {
            return Ok(Some(p));
        }
    }
    Ok(None)
}

/// 秘密鍵を `private`（Unix では 0600）、公開鍵を `<private>.pub` に書く。
/// `overwrite` でなければ、どちらかが既にあると書かずに失敗する（[`io::ErrorKind::AlreadyExists`]）
pub fn save(key: &VanityKey, comment: &str, private: &Path, overwrite: bool) -> io::Result<()> {
    // 片方だけ書いて止まらないよう、先に両方を確かめる（書くときも create_new なので、
    // 確かめた後に作られたファイルも上書きしない）
    if !overwrite && let Some(p) = existing(private)? {
        let message = format!("{} は既に存在します", p.display());
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, message));
    }
    write_file(
        private,
        &private_openssh(key, comment, rand::random()),
        0o600,
        overwrite,
    )?;
    write_file(
        &pub_path(private),
        &(key.public_openssh(comment) + "\n"),
        0o644,
        overwrite,
    )
}

/// ファイルを新しく作って書く。`overwrite` なら既存のファイルを先に消し、そうでなければ
/// 既存のファイルがあると失敗する（どちらも create_new）。
/// `mode` は Unix のパーミッション（秘密鍵は 0600 で所有者だけ読める）。既存のファイルに上書きすると
/// そのファイルのパーミッション（0644 など）が残ってしまうので、必ず新しいファイルとして作る
fn write_file(path: &Path, content: &str, mode: u32, overwrite: bool) -> io::Result<()> {
    if overwrite
        && let Err(e) = std::fs::remove_file(path)
        && e.kind() != io::ErrorKind::NotFound
    {
        return Err(e);
    }
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, mode);
    #[cfg(not(unix))]
    let _ = mode;
    opts.open(path)?.write_all(content.as_bytes())
}

/// 暗号化なしの OpenSSH 秘密鍵（openssh-key-v1）
///
/// `checkint` は復号確認用の乱数。テストで固定できるよう引数にしている。
pub fn private_openssh(key: &VanityKey, comment: &str, checkint: u32) -> String {
    let mut keypair = [0u8; 64];
    keypair[..32].copy_from_slice(key.seed());
    keypair[32..].copy_from_slice(key.public_bytes());

    let mut private = Vec::new();
    SshWriter::new(&mut private)
        .u32(checkint)
        .u32(checkint)
        .string(KEY_TYPE)
        .string(key.public_bytes().as_slice())
        .string(keypair.as_slice())
        .string(comment.as_bytes());
    // 8バイト境界まで 1, 2, 3, ... で埋める
    let pad = private.len().next_multiple_of(8) - private.len();
    private.extend(1..=pad as u8);

    let mut body = b"openssh-key-v1\0".to_vec();
    SshWriter::new(&mut body)
        .string(b"none") // cipher
        .string(b"none") // kdf
        .string(b"") // kdf options
        .u32(1) // 鍵の数
        .string(key.public_blob().as_slice())
        .string(private.as_slice());

    let b64 = STANDARD.encode(body);
    let lines: Vec<&str> = b64
        .as_bytes()
        .chunks(70)
        .map(|c| std::str::from_utf8(c).expect("base64 は ASCII"))
        .collect();
    format!(
        "-----BEGIN OPENSSH PRIVATE KEY-----\n{}\n-----END OPENSSH PRIVATE KEY-----\n",
        lines.join("\n")
    )
}

/// SSH のワイヤ形式（RFC 4251）で書き込むヘルパー
struct SshWriter<'a>(&'a mut Vec<u8>);

impl<'a> SshWriter<'a> {
    fn new(buf: &'a mut Vec<u8>) -> Self {
        Self(buf)
    }

    fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_be_bytes());
        self
    }

    fn string(&mut self, s: &[u8]) -> &mut Self {
        self.u32(u32::try_from(s.len()).expect("4GiB 未満"));
        self.0.extend_from_slice(s);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::check;
    use crate::{rfc8032, ssh_format};
    use proptest::prelude::*;
    use rstest::rstest;

    /// RFC 4251 Section 5 に載っている例そのもの
    #[rstest]
    #[case::uint32_example(None, Some(699_921_578), &[0x29, 0xb7, 0xf4, 0xaa])]
    #[case::string_example(Some(b"testing".as_slice()), None, b"\x00\x00\x00\x07testing")]
    #[case::empty_string(Some(b"".as_slice()), None, &[0, 0, 0, 0])]
    fn ssh_writer_matches_rfc_4251_examples(
        #[case] string: Option<&[u8]>,
        #[case] uint32: Option<u32>,
        #[case] want: &[u8],
    ) {
        let mut buf = Vec::new();
        let mut w = SshWriter::new(&mut buf);
        if let Some(s) = string {
            w.string(s);
        }
        if let Some(v) = uint32 {
            w.u32(v);
        }
        assert_eq!(buf, want);
    }

    #[derive(Debug, Clone)]
    enum Op {
        U32(u32),
        Str(Vec<u8>),
    }

    /// どんな並びで書いても、仕様書から書いた独立実装（RFC 4251 Section 5）と同じバイト列になる
    #[rstest]
    fn ssh_writer_matches_independent_rfc_4251() {
        let op = prop_oneof![
            any::<u32>().prop_map(Op::U32),
            prop::collection::vec(any::<u8>(), 0..300).prop_map(Op::Str),
        ];
        check(1024, prop::collection::vec(op, 0..12), |ops| {
            let (mut ours, mut spec) = (Vec::new(), Vec::new());
            let mut w = SshWriter::new(&mut ours);
            for op in &ops {
                match op {
                    Op::U32(v) => {
                        w.u32(*v);
                        ssh_format::put_uint32(&mut spec, *v);
                    }
                    Op::Str(s) => {
                        w.string(s);
                        ssh_format::put_string(&mut spec, s);
                    }
                }
            }
            prop_assert_eq!(ours, spec);
            Ok(())
        });
    }

    /// 書いたファイルの中身: 秘密鍵は元の鍵に戻せて、`.pub` は authorized_keys の1行と改行
    fn assert_files_hold(key: &VanityKey, private: &Path) {
        let pem = std::fs::read_to_string(private).unwrap();
        let file = ssh_format::parse_private_key_file(&pem).unwrap();
        assert_eq!(&file.keys[0].k, key.seed());
        assert_eq!(file.keys[0].comment, b"c");
        let pub_line = std::fs::read_to_string(pub_path(private)).unwrap();
        assert_eq!(pub_line, key.public_openssh("c") + "\n");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(private).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "秘密鍵は所有者だけ読める");
        }
    }

    /// 既にあるファイル（秘密鍵・`.pub`）と `overwrite` の組み合わせ。上書きしないなら、どちらかが
    /// あれば何も書かずに失敗して元のファイルはそのまま。上書きするなら、古いファイルを消して作り直す
    /// （0644 の古いファイルを置き換えても、秘密鍵は 0600 になる）
    #[rstest]
    fn save_respects_existing_files(
        #[values(false, true)] private_exists: bool,
        #[values(false, true)] pub_exists: bool,
        #[values(false, true)] overwrite: bool,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let private = dir.path().join("id_ed25519_test");
        for (exists, p) in [
            (private_exists, private.clone()),
            (pub_exists, pub_path(&private)),
        ] {
            if exists {
                std::fs::write(&p, "old\n").unwrap();
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).unwrap();
                }
            }
        }
        let first = [
            (private_exists, &private),
            (pub_exists, &pub_path(&private)),
        ]
        .into_iter()
        .find(|(e, _)| *e)
        .map(|(_, p)| p.clone());
        // 本番の KeyFiles（trait 越し）で確かめる
        assert_eq!(KeyFiles.existing(&private).unwrap(), first);

        let key = VanityKey::from_seed([7; 32]);
        let result = KeyFiles.save(&key, "c", &private, overwrite);
        if first.is_some() && !overwrite {
            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::AlreadyExists);
            // 片方だけ書かれることはなく、元のファイルもそのまま
            for (exists, p) in [
                (private_exists, private.clone()),
                (pub_exists, pub_path(&private)),
            ] {
                match exists {
                    true => assert_eq!(std::fs::read_to_string(&p).unwrap(), "old\n"),
                    false => assert!(!p.exists(), "{} を書いた", p.display()),
                }
            }
        } else {
            result.unwrap();
            assert_files_hold(&key, &private);
        }
    }

    /// RFC・仕様への準拠（正解は tests/support/ の独立実装）: 秘密鍵ファイル（PROTOCOL.key・
    /// agent draft 3.2.3）と、RFC 8032 の鍵の書き出し
    mod conformance {
        use super::*;

        /// RFC の鍵を OpenSSH 形式で書き出すと、仕様どおりのファイルになり、中身は RFC の値そのもの
        #[rstest]
        fn section_7_1_as_openssh_files(
            #[values("1", "2", "3", "1024", "SHA(abc)")] name: &str,
            #[values("", "rfc8032", "日本語のコメント")] comment: &str,
        ) {
            let v = rfc8032::vector(name);
            let key = VanityKey::from_seed(v.secret);

            // 公開鍵の1行（authorized_keys 形式 → RFC 4648 base64 → RFC 8709 blob）
            let line = ssh_format::parse_authorized_key(&key.public_openssh(comment)).unwrap();
            assert_eq!(line.public_key, v.public);
            assert_eq!(
                line.comment.as_deref(),
                (!comment.is_empty()).then_some(comment)
            );

            // 秘密鍵ファイル（PROTOCOL.key + agent draft 3.2.3）
            let file =
                ssh_format::parse_private_key_file(&private_openssh(&key, comment, 0x1234_5678))
                    .unwrap();
            assert_eq!(file.violations(), Vec::<String>::new());
            let k = &file.keys[0];
            assert_eq!(k.k, v.secret, "k は RFC 8032 の秘密鍵そのもの");
            assert_eq!(k.enc_a, v.public, "ENC(A) は RFC 8032 の公開鍵");
            assert_eq!(k.comment, comment.as_bytes());
            assert_eq!((file.checkint1, file.checkint2), (0x1234_5678, 0x1234_5678));

            // ファイルから取り出した k で、RFC の署名がそのまま再現できる
            assert_eq!(rfc8032::sign(&k.k, &v.message), v.signature);
        }

        // ===========================================================================
        // PROTOCOL.key + agent draft 3.2.3: 秘密鍵ファイル
        // ===========================================================================

        /// 秘密鍵ファイルの各フィールドが仕様どおりで、別の実装（ssh-key クレート）でも同じ鍵として読める。
        /// コメント（任意の Unicode）のバイト数を 8 で割った余りを変えて、パディング 0〜7 バイトをすべて通す
        #[rstest]
        fn private_key_file_follows_spec(#[values(0, 1, 2, 3, 4, 5, 6, 7)] remainder: usize) {
            // 秘密鍵の部分は checkint 2つ（8）+ 鍵の種類（4 + 11）+ ENC(A)（4 + 32）+ k || ENC(A)（4 + 64）
            // + コメント（4 + 長さ）で 131 + 長さ バイト。これをブロック長 8 の倍数にそろえる
            let padding = (8 - (131 + remainder) % 8) % 8;
            let comments = "(?s).{0,100}".prop_map(move |c: String| {
                let fill = (remainder + 8 - c.len() % 8) % 8;
                c + &"x".repeat(fill)
            });
            check(
                32,
                (any::<[u8; 32]>(), comments, any::<u32>()),
                |(seed, comment, checkint)| {
                    let key = VanityKey::from_seed(seed);
                    let pem = private_openssh(&key, &comment, checkint);
                    let file =
                        ssh_format::parse_private_key_file(&pem).map_err(TestCaseError::fail)?;
                    prop_assert_eq!(file.violations(), Vec::<String>::new());
                    prop_assert_eq!(file.padding.len(), padding);
                    let k = &file.keys[0];
                    prop_assert_eq!(k.k, seed);
                    prop_assert_eq!(&k.enc_a, key.public_bytes());
                    prop_assert_eq!(&k.comment, comment.as_bytes());
                    prop_assert_eq!((file.checkint1, file.checkint2), (checkint, checkint));

                    let parsed = ssh_key::PrivateKey::from_openssh(&pem)
                        .map_err(|e| TestCaseError::fail(e.to_string()))?;
                    prop_assert_eq!(parsed.comment(), comment.as_str());
                    prop_assert_eq!(
                        parsed.key_data().ed25519().unwrap().private.to_bytes(),
                        seed
                    );
                    Ok(())
                },
            );
        }
    }
}
