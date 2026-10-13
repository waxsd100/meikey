//! 鍵を OpenSSH 形式のファイルに保存する
//!
//! 秘密鍵は暗号化なしの openssh-key-v1（PROTOCOL.key）、公開鍵は authorized_keys の1行（`.pub`）。
//! 秘密鍵は本人だけが読めるようにする（Unix は 0600、Windows は本人だけのアクセス権）。

use crate::lang::tf;
use crate::model::key::{KEY_TYPE, VanityKey};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use std::fs::{File, OpenOptions, remove_file, symlink_metadata};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// 鍵のファイルを調べる・書くときのエラー
#[derive(Debug, thiserror::Error)]
pub enum KeyFileError {
    /// 保存先にファイル名が無い（`dir/` など）
    #[error("{}", tf!("{}: ファイル名が無い", "{}: no file name", .0.display()))]
    NoFileName(PathBuf),
    /// 上書きしないのに、鍵のファイル（秘密鍵か `.pub`）が既にある
    #[error("{}", tf!("既に存在: {}", "already exists: {}", .0.display()))]
    AlreadyExists(PathBuf),
    /// ファイルかディレクトリの読み書きに失敗した（`path` はどのファイル・ディレクトリで失敗したか）
    #[error("{}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

/// 鍵の保存先。controller はこの trait 越しに保存するので、テストでは mockall の
/// `MockKeyRepository` に差し替えて、ファイルに触れずに呼ばれ方を確かめられる
#[cfg_attr(test, mockall::automock)]
pub trait KeyRepository {
    /// 秘密鍵と公開鍵のうち、既にあるほうのパス（両方なければ None）。リンク先の無いシンボリックリンクも
    /// 「ある」とする（その名前では新しく作れないため）
    fn existing(&self, private: &Path) -> Result<Option<PathBuf>, KeyFileError>;

    /// `private` を置くディレクトリに、実際にファイルを作って消せるか確かめる（探す前に、保存できないことに
    /// 気づくため）
    fn check_writable(&self, private: &Path) -> Result<(), KeyFileError>;

    /// 秘密鍵を `private`、公開鍵を `<private>.pub` に書く。
    /// `overwrite` でなければ、どちらかが既にあると書かずに失敗する（[`KeyFileError::AlreadyExists`]）
    fn save(
        &self,
        key: &VanityKey,
        comment: &str,
        private: &Path,
        overwrite: bool,
    ) -> Result<(), KeyFileError>;
}

/// ファイルシステムに OpenSSH 形式で保存する [`KeyRepository`]（本番で使うもの）
#[derive(Debug, Clone, Copy, Default)]
pub struct KeyFiles;

impl KeyRepository for KeyFiles {
    fn existing(&self, private: &Path) -> Result<Option<PathBuf>, KeyFileError> {
        existing(private)
    }

    fn check_writable(&self, private: &Path) -> Result<(), KeyFileError> {
        check_writable(private)
    }

    fn save(
        &self,
        key: &VanityKey,
        comment: &str,
        private: &Path,
        overwrite: bool,
    ) -> Result<(), KeyFileError> {
        save(key, comment, private, overwrite)
    }
}

/// 秘密鍵 `private` と組になる公開鍵のパス（`<private>.pub`）
pub fn pub_path(private: &Path) -> PathBuf {
    private.with_added_extension("pub")
}

/// どのファイルで失敗したか分かるように、`e` にそのパスを付ける
fn with_path(path: &Path, e: io::Error) -> KeyFileError {
    KeyFileError::Io {
        path: path.to_path_buf(),
        source: e,
    }
}

/// 秘密鍵と公開鍵のうち、既にあるほうのパス（両方なければ None）。リンクはたどらずに調べるので、
/// リンク先の無いシンボリックリンクも「ある」になる
pub fn existing(private: &Path) -> Result<Option<PathBuf>, KeyFileError> {
    for p in [private.to_path_buf(), pub_path(private)] {
        match symlink_metadata(&p) {
            Ok(_) => return Ok(Some(p)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(with_path(&p, e)),
        }
    }
    Ok(None)
}

/// `private` を置くディレクトリに、試しのファイルを作って消せるか確かめる
pub fn check_writable(private: &Path) -> Result<(), KeyFileError> {
    let dir = match private.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => Path::new("."),
    };
    let name = private
        .file_name()
        .ok_or_else(|| KeyFileError::NoFileName(private.to_path_buf()))?
        .to_string_lossy();
    let probe = dir.join(format!(".{name}.meikey-{:016x}.tmp", rand::random::<u64>()));
    create_new(&probe, 0o600).map_err(|e| with_path(dir, e))?;
    remove_file(&probe).map_err(|e| with_path(&probe, e))
}

/// 秘密鍵を `private`（本人だけが読める）、公開鍵を `<private>.pub` に書く。
/// `overwrite` でなければ、どちらかが既にあると書かずに失敗する（[`KeyFileError::AlreadyExists`]）
pub fn save(
    key: &VanityKey,
    comment: &str,
    private: &Path,
    overwrite: bool,
) -> Result<(), KeyFileError> {
    // 片方だけ書いて止まらないよう、先に両方を確かめる（書くときも create_new なので、
    // 確かめた後に作られたファイルも上書きしない）
    if !overwrite && let Some(p) = existing(private)? {
        return Err(KeyFileError::AlreadyExists(p));
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

/// ファイルを新しく作る（既にあれば失敗する）。`mode` は Unix のパーミッションで、0600 なら
/// Windows でも本人だけのアクセス権にする（中身を書く前なので、ほかの人に読める瞬間はない）
fn create_new(path: &Path, mode: u32) -> io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, mode);
    let file = opts.open(path)?;
    #[cfg(windows)]
    if mode & 0o077 == 0 {
        restrict_to_owner(path)?;
    }
    #[cfg(not(any(unix, windows)))]
    let _ = mode;
    Ok(file)
}

/// ファイルを新しく作って書き、ディスクに書き終わるまで待つ。`overwrite` なら既存のファイルを先に消し、
/// そうでなければ既存のファイルがあると失敗する（どちらも create_new）。
/// 既存のファイルに上書きすると、そのファイルのパーミッション（0644 など）が残ってしまうので、
/// 必ず新しいファイルとして作る
fn write_file(path: &Path, content: &str, mode: u32, overwrite: bool) -> Result<(), KeyFileError> {
    if overwrite
        && let Err(e) = remove_file(path)
        && e.kind() != io::ErrorKind::NotFound
    {
        return Err(with_path(path, e));
    }
    let write = || {
        let mut file = create_new(path, mode)?;
        file.write_all(content.as_bytes())?;
        file.sync_all()
    };
    write().map_err(|e| with_path(path, e))
}

/// ファイルのアクセス権を、このプロセスのユーザーのフルコントロールだけにする（親フォルダから継承した
/// アクセス権は外す）。Unix の 0600 に当たり、Windows の OpenSSH はほかのユーザーが読める秘密鍵を使わない
#[cfg(windows)]
fn restrict_to_owner(path: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use std::ptr::{null, null_mut};
    use windows_sys::Win32::Foundation::{CloseHandle, ERROR_SUCCESS, HANDLE, HLOCAL, LocalFree};
    use windows_sys::Win32::Security::Authorization::{
        EXPLICIT_ACCESS_W, NO_MULTIPLE_TRUSTEE, SE_FILE_OBJECT, SET_ACCESS, SetEntriesInAclW,
        SetNamedSecurityInfoW, TRUSTEE_IS_SID, TRUSTEE_IS_USER, TRUSTEE_W,
    };
    use windows_sys::Win32::Security::{
        ACL, DACL_SECURITY_INFORMATION, GetTokenInformation, NO_INHERITANCE,
        PROTECTED_DACL_SECURITY_INFORMATION, TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::Storage::FileSystem::FILE_ALL_ACCESS;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    // このプロセスのユーザーの SID（TOKEN_USER の後ろに続く）。u64 で取って境界をそろえる
    let mut token: HANDLE = null_mut();
    // SAFETY: GetCurrentProcess は自分のプロセスの擬似ハンドルを返す。token は書き込み先
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let mut len = 0u32;
    // SAFETY: 大きさを問い合わせるだけ（バッファなし）
    unsafe { GetTokenInformation(token, TokenUser, null_mut(), 0, &mut len) };
    let mut buf = vec![0u64; (len as usize).div_ceil(8)];
    // SAFETY: buf は len バイト以上あり、8 バイト境界（TOKEN_USER の境界）にそろっている
    let ok =
        unsafe { GetTokenInformation(token, TokenUser, buf.as_mut_ptr().cast(), len, &mut len) };
    // SAFETY: OpenProcessToken で開いたハンドル
    unsafe { CloseHandle(token) };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: GetTokenInformation(TokenUser) が buf の先頭に TOKEN_USER を書いた
    let sid = unsafe { (*buf.as_ptr().cast::<TOKEN_USER>()).User.Sid };

    let access = EXPLICIT_ACCESS_W {
        grfAccessPermissions: FILE_ALL_ACCESS,
        grfAccessMode: SET_ACCESS,
        grfInheritance: NO_INHERITANCE,
        Trustee: TRUSTEE_W {
            pMultipleTrustee: null_mut(),
            MultipleTrusteeOperation: NO_MULTIPLE_TRUSTEE,
            TrusteeForm: TRUSTEE_IS_SID,
            TrusteeType: TRUSTEE_IS_USER,
            ptstrName: sid.cast(),
        },
    };
    let mut acl: *mut ACL = null_mut();
    // SAFETY: access は1件、sid は buf が生きている間有効。acl は書き込み先（LocalFree で解放する）
    let e = unsafe { SetEntriesInAclW(1, &access, null(), &mut acl) };
    if e != ERROR_SUCCESS {
        return Err(io::Error::from_raw_os_error(e as i32));
    }
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    // SAFETY: wide は NUL で終わるパス、acl は SetEntriesInAclW が作ったもの。継承を止めて（PROTECTED）
    // この ACL だけにする
    let e = unsafe {
        SetNamedSecurityInfoW(
            wide.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            null_mut(),
            null_mut(),
            acl,
            null(),
        )
    };
    // SAFETY: SetEntriesInAclW が LocalAlloc で作ったもの
    unsafe { LocalFree(acl as HLOCAL) };
    if e != ERROR_SUCCESS {
        return Err(io::Error::from_raw_os_error(e as i32));
    }
    Ok(())
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
    use crate::rfc8032::sign;
    use crate::rfc8032::vector;
    use crate::ssh_format::parse_authorized_key;
    use crate::ssh_format::parse_private_key_file;
    use crate::ssh_format::put_string;
    use crate::ssh_format::put_uint32;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use rstest::{fixture, rstest};
    use similar_asserts::assert_eq;
    #[cfg(unix)]
    use std::fs::Permissions;
    #[cfg(unix)]
    use std::fs::metadata;
    use std::fs::read_to_string;
    #[cfg(unix)]
    use std::fs::set_permissions;
    use std::fs::write;
    use tempfile::{TempDir, tempdir};

    /// テストごとの一時ディレクトリ（drop で消える）
    #[fixture]
    fn dir() -> TempDir {
        tempdir().unwrap()
    }

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
            vec(any::<u8>(), 0..300).prop_map(Op::Str),
        ];
        check(1024, vec(op, 0..12), |ops| {
            let (mut ours, mut spec) = (Vec::new(), Vec::new());
            let mut w = SshWriter::new(&mut ours);
            for op in &ops {
                match op {
                    Op::U32(v) => {
                        w.u32(*v);
                        put_uint32(&mut spec, *v);
                    }
                    Op::Str(s) => {
                        w.string(s);
                        put_string(&mut spec, s);
                    }
                }
            }
            prop_assert_eq!(ours, spec);
            Ok(())
        });
    }

    /// 書いたファイルの中身: 秘密鍵は元の鍵に戻せて、`.pub` は authorized_keys の1行と改行
    fn assert_files_hold(key: &VanityKey, private: &Path) {
        let pem = read_to_string(private).unwrap();
        let file = parse_private_key_file(&pem).unwrap();
        assert_eq!(&file.keys[0].k, key.seed());
        assert_eq!(file.keys[0].comment, b"c");
        let pub_line = read_to_string(pub_path(private)).unwrap();
        assert_eq!(pub_line, key.public_openssh("c") + "\n");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = metadata(private).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "秘密鍵は所有者だけ読める");
        }
    }

    /// リンク先の無いシンボリックリンクも「既にある」（その名前では作れないので、探す前に止める）
    #[cfg(unix)]
    #[rstest]
    #[case::private("id_test")]
    #[case::public("id_test.pub")]
    fn dangling_symlink_counts_as_existing(#[case] link: &str, dir: TempDir) {
        let private = dir.path().join("id_test");
        std::os::unix::fs::symlink(dir.path().join("nowhere/target"), dir.path().join(link))
            .unwrap();
        assert_eq!(existing(&private).unwrap(), Some(dir.path().join(link)));
    }

    /// 書けるディレクトリなら通り、試しのファイルは残さない。親のディレクトリが無い・ファイルなら、
    /// そのパスを付けたエラー（[`KeyFileError::Io`]）になる
    #[rstest]
    #[case::writable("id_test", None)]
    #[case::no_parent("nodir/id_test", Some("nodir"))]
    #[case::parent_is_file("file/id_test", Some("file"))]
    fn check_writable_cases(#[case] rel: &str, #[case] error_path: Option<&str>, dir: TempDir) {
        write(dir.path().join("file"), "x").unwrap();
        let before: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
        let result = check_writable(&dir.path().join(rel));
        match error_path {
            None => assert_eq!(result.is_ok(), true, "{result:?}"),
            Some(p) => {
                let e = result.unwrap_err();
                let message = e.to_string();
                assert_eq!(
                    matches!(&e, KeyFileError::Io { path, .. } if *path == dir.path().join(p)),
                    true,
                    "{e:?}"
                );
                assert_eq!(
                    message.starts_with(&dir.path().join(p).display().to_string()),
                    true,
                    "{message}"
                );
            }
        }
        let after: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(after.len(), before.len(), "試しのファイルが残っている");
    }

    /// ファイル名の無い保存先（`..` など）は、ディレクトリに触れずにエラーにする
    #[rstest]
    fn check_writable_needs_a_file_name() {
        let e = check_writable(Path::new("..")).unwrap_err();
        assert_eq!(
            matches!(&e, KeyFileError::NoFileName(p) if p == Path::new("..")),
            true
        );
        assert_eq!(e.to_string(), "..: ファイル名が無い");
    }

    /// 相対パスのファイル名だけ（親が空）なら、今のディレクトリを確かめる
    #[rstest]
    fn check_writable_bare_name_uses_current_dir() {
        assert_eq!(
            check_writable(Path::new("id_meikey_check_test")).is_ok(),
            true
        );
    }

    /// 保存に失敗したら、どのファイルで失敗したかがエラーに出る
    #[rstest]
    fn save_error_names_the_path(dir: TempDir) {
        let private = dir.path().join("nodir/id_test");
        let e = save(&VanityKey::from_seed([3; 32]), "c", &private, false).unwrap_err();
        assert_eq!(
            matches!(&e, KeyFileError::Io { path, source }
                if *path == private && source.kind() == io::ErrorKind::NotFound),
            true,
            "{e:?}"
        );
        assert_eq!(
            e.to_string().contains(&private.display().to_string()),
            true,
            "{e}"
        );
    }

    /// 既にあるファイル（秘密鍵・`.pub`）と `overwrite` の組み合わせ。上書きしないなら、どちらかが
    /// あれば何も書かずに失敗して元のファイルはそのまま。上書きするなら、古いファイルを消して作り直す
    /// （0644 の古いファイルを置き換えても、秘密鍵は 0600 になる）
    #[rstest]
    fn save_respects_existing_files(
        #[values(false, true)] private_exists: bool,
        #[values(false, true)] pub_exists: bool,
        #[values(false, true)] overwrite: bool,
        dir: TempDir,
    ) {
        let private = dir.path().join("id_ed25519_test");
        for (exists, p) in [
            (private_exists, private.clone()),
            (pub_exists, pub_path(&private)),
        ] {
            if exists {
                write(&p, "old\n").unwrap();
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    set_permissions(&p, Permissions::from_mode(0o644)).unwrap();
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
            assert_eq!(
                matches!(result.unwrap_err(), KeyFileError::AlreadyExists(p) if first.as_ref() == Some(&p)),
                true
            );
            // 片方だけ書かれることはなく、元のファイルもそのまま
            for (exists, p) in [
                (private_exists, private.clone()),
                (pub_exists, pub_path(&private)),
            ] {
                match exists {
                    true => assert_eq!(read_to_string(&p).unwrap(), "old\n"),
                    false => assert_eq!(p.exists(), false, "{} を書いた", p.display()),
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
        use similar_asserts::assert_eq;

        /// RFC の鍵を OpenSSH 形式で書き出すと、仕様どおりのファイルになり、中身は RFC の値そのもの
        #[rstest]
        fn section_7_1_as_openssh_files(
            #[values("1", "2", "3", "1024", "SHA(abc)")] name: &str,
            #[values("", "rfc8032", "日本語のコメント")] comment: &str,
        ) {
            let v = vector(name);
            let key = VanityKey::from_seed(v.secret);

            // 公開鍵の1行（authorized_keys 形式 → RFC 4648 base64 → RFC 8709 blob）
            let line = parse_authorized_key(&key.public_openssh(comment)).unwrap();
            assert_eq!(line.public_key, v.public);
            assert_eq!(
                line.comment.as_deref(),
                (!comment.is_empty()).then_some(comment)
            );

            // 秘密鍵ファイル（PROTOCOL.key + agent draft 3.2.3）
            let file =
                parse_private_key_file(&private_openssh(&key, comment, 0x1234_5678)).unwrap();
            assert_eq!(file.violations(), Vec::<String>::new());
            let k = &file.keys[0];
            assert_eq!(k.k, v.secret, "k は RFC 8032 の秘密鍵そのもの");
            assert_eq!(k.enc_a, v.public, "ENC(A) は RFC 8032 の公開鍵");
            assert_eq!(k.comment, comment.as_bytes());
            assert_eq!((file.checkint1, file.checkint2), (0x1234_5678, 0x1234_5678));

            // ファイルから取り出した k で、RFC の署名がそのまま再現できる
            assert_eq!(sign(&k.k, &v.message), v.signature);
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
                    let file = parse_private_key_file(&pem).map_err(TestCaseError::fail)?;
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
