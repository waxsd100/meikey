//! meikey で作った鍵が「本当に SSH 鍵として使えるか」を確かめるテスト（[`tests`]）と、そのヘルパー
//!
//! 1. 鍵ファイルを ssh-key・仕様書から書いた独立実装・OpenSSH の ssh-keygen で読む（[`Reader`]）
//! 2. 署名する側（[`Signer`]）× 検証する側（[`Verifier`]）の総当たりで、
//!    正しい署名だけが通り、別の鍵・改ざん・別の名前空間では通らない（[`Check`]）
//! 3. 署名を RFC 8032 の参照実装と1ビット単位で突き合わせる
//! 4. OpenSSH が書く鍵ファイルと1バイトも違わない
//!
//! パスフレーズを付けた鍵は [`crate::encrypted`]、ssh-agent は [`crate::agent`]、
//! 実際のログインは [`crate::sshd`]、authorized_keys は [`crate::authorized_keys`] で確かめる。
//!
//! 鍵は meikey 本体で低セキュリティモード（`--low-security`）と既定（定数時間）の両方で作り、
//! テストバイナリの中で使い回す（[`KeyFiles::shared`]）。
//! OpenSSH を使うケースは、OpenSSH が無い環境ではスキップする。

use crate::common::rfc8032::{secret_to_public, sign, verify};
use crate::common::ssh_format::{
    SshSig as SpecSig, armor_sshsig, parse_authorized_key, parse_private_key_file, parse_sshsig,
    public_blob, sha256_fingerprint, sshsig_signed_data,
};
use crate::common::*;
use ssh_key::{HashAlg, LineEnding, PrivateKey, PublicKey, SshSig};
use std::fs::{File, read_to_string, write};
use std::path::Path;
use std::process::Command;
use tempfile::tempdir_in;

const NAMESPACE: &str = "file";
const MESSAGE: &[u8] = b"meikey integration test\n";

// ---------------------------------------------------------------------------
// 1. 鍵ファイルを読む
// ---------------------------------------------------------------------------

/// 鍵ファイルの読み方
#[derive(Debug, Clone, Copy)]
enum Reader {
    /// ssh-key クレートで読み、シードから ed25519-dalek で公開鍵を計算し直す
    SshKey,
    /// 仕様書から書いた独立実装（PROTOCOL.key・RFC 8709）で読み、シードから RFC 8032 の参照実装で公開鍵を計算し直す
    Spec,
    /// `ssh-keygen -y` で秘密鍵から公開鍵を復元し、フィンガープリントを比べる
    SshKeygen,
}

// ---------------------------------------------------------------------------
// 2. 署名する側 × 検証する側
// ---------------------------------------------------------------------------

/// 署名する側
#[derive(Debug, Clone, Copy)]
enum Signer {
    /// ssh-key クレート
    SshKey,
    /// `ssh-keygen -Y sign`
    OpenSsh,
    /// 秘密鍵ファイルのシードで RFC 8032 の参照実装が署名し、独立実装で SSHSIG にする
    Reference,
}

impl Signer {
    /// `message` に `NAMESPACE` で署名し、armored SSHSIG（`ssh-keygen -Y sign` の出力と同じ形式）で返す
    fn sign(self, files: &KeyFiles, dir: &Path, message: &[u8], hash: &str) -> String {
        match self {
            Signer::SshKey => {
                let hash = match hash {
                    "sha512" => HashAlg::Sha512,
                    "sha256" => HashAlg::Sha256,
                    other => panic!("未対応のハッシュ {other}"),
                };
                PrivateKey::from_openssh(&files.private)
                    .unwrap()
                    .sign(NAMESPACE, hash, message)
                    .unwrap()
                    .to_pem(LineEnding::LF)
                    .unwrap()
            }
            Signer::OpenSsh => {
                let work = tempdir_in(dir).unwrap();
                let (key, _) = files.write_to(work.path());
                let msg = work.path().join("message");
                write(&msg, message).unwrap();
                let mut cmd = Command::new("ssh-keygen");
                cmd.args(["-Y", "sign", "-n", NAMESPACE]);
                // SHA-512 は -Y sign の既定。-O hashalg を知らない古い OpenSSH でも動くよう、既定なら付けない
                if hash != "sha512" {
                    cmd.arg("-O").arg(format!("hashalg={hash}"));
                }
                run(cmd.arg("-f").arg(&key).arg(&msg));
                read_to_string(msg.with_added_extension("sig")).unwrap()
            }
            Signer::Reference => {
                let data = sshsig_signed_data(NAMESPACE.as_bytes(), b"", hash.as_bytes(), message);
                armor_sshsig(&SpecSig {
                    version: 1,
                    public_key: files.public(),
                    namespace: NAMESPACE.into(),
                    reserved: Vec::new(),
                    hash_algorithm: hash.into(),
                    signature: sign(&files.seed(), &data),
                })
            }
        }
    }
}

/// 検証する側
#[derive(Debug, Clone, Copy)]
enum Verifier {
    /// ssh-key クレート
    SshKey,
    /// `ssh-keygen -Y verify`（allowed_signers に公開鍵を1つだけ登録）
    OpenSsh,
    /// 独立実装で SSHSIG を分解し、PROTOCOL.sshsig 3. の署名対象を組み立てて RFC 8032 の参照実装で検証
    Reference,
}

impl Verifier {
    /// `armored` が、`pub_line` の鍵による、`namespace` での `message` への署名として正しいか
    fn verify(
        self,
        dir: &Path,
        armored: &str,
        pub_line: &str,
        namespace: &str,
        message: &[u8],
    ) -> bool {
        match self {
            Verifier::SshKey => PublicKey::from_openssh(pub_line)
                .unwrap()
                .verify(namespace, message, &SshSig::from_pem(armored).unwrap())
                .is_ok(),
            Verifier::OpenSsh => {
                let signers = write_allowed_signers(dir, COMMENT, &key_part(pub_line));
                let (sig, msg) = (dir.join("verify.sig"), dir.join("verify.msg"));
                write(&sig, armored).unwrap();
                write(&msg, message).unwrap();
                Command::new("ssh-keygen")
                    .args(["-Y", "verify", "-n", namespace, "-I", COMMENT, "-f"])
                    .arg(&signers)
                    .arg("-s")
                    .arg(&sig)
                    .stdin(File::open(&msg).unwrap())
                    .output()
                    .unwrap()
                    .status
                    .success()
            }
            Verifier::Reference => {
                let sig = parse_sshsig(armored).unwrap();
                let public = parse_authorized_key(pub_line).unwrap().public_key;
                // 署名対象は「検証する側が期待する」名前空間とメッセージで組み立てる
                let data = sshsig_signed_data(
                    namespace.as_bytes(),
                    &sig.reserved,
                    &sig.hash_algorithm,
                    message,
                );
                verify(&public, &data, &sig.signature)
                    && sig.public_key == public
                    && sig.namespace == namespace.as_bytes()
            }
        }
    }
}

/// 検証のしかた
#[derive(Debug, Clone, Copy)]
enum Check {
    /// 署名した鍵・同じ名前空間・同じメッセージ → 通る
    Genuine,
    /// 別の鍵（もう一方の方式で作った鍵）の `.pub` で検証 → 通らない
    OtherKey,
    /// メッセージを1ビット変える → 通らない
    TamperedMessage,
    /// 別の名前空間で検証 → 通らない
    OtherNamespace,
}

mod tests {
    use super::*;
    use crate::common::property::check;
    use crate::common::rfc8032::vector;
    use ed25519_dalek::SigningKey;
    use meikey::VanityKey;
    use meikey::repository::key_file::private_openssh;
    use proptest::prelude::*;
    use rstest::rstest;
    use similar_asserts::assert_eq;
    use tempfile::TempDir;

    // ---------------------------------------------------------------------------
    // 1. 鍵ファイルを読む
    // ---------------------------------------------------------------------------

    /// 秘密鍵と `.pub` が同じ鍵を表していて、秘密鍵の中の公開鍵もシードから計算したものと一致する
    #[rstest]
    fn key_files_work(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values(Reader::SshKey, Reader::Spec, Reader::SshKeygen)] reader: Reader,
        dir: TempDir,
    ) {
        let files = KeyFiles::shared(mode);
        match reader {
            Reader::SshKey => {
                let private = PrivateKey::from_openssh(&files.private).expect("秘密鍵として読める");
                let public = PublicKey::from_openssh(&files.pub_line).expect("公開鍵として読める");
                assert_eq!(private.is_encrypted(), false);
                assert_eq!(private.comment(), COMMENT);
                assert_eq!(public.comment(), COMMENT);
                assert_eq!(private.public_key().key_data(), public.key_data());

                let pair = private.key_data().ed25519().unwrap();
                let recomputed = SigningKey::from_bytes(&pair.private.to_bytes()).verifying_key();
                assert_eq!(recomputed.to_bytes(), pair.public.0);
            }
            Reader::Spec => {
                let file = parse_private_key_file(&files.private).unwrap();
                assert_eq!(file.violations(), Vec::<String>::new());
                let line = parse_authorized_key(&files.pub_line).unwrap();
                assert_eq!(line.comment.as_deref(), Some(COMMENT));

                let key = &file.keys[0];
                assert_eq!(key.comment, COMMENT.as_bytes());
                assert_eq!(key.enc_a, line.public_key, "秘密鍵の中の公開鍵 = .pub");
                assert_eq!(
                    secret_to_public(&key.k),
                    line.public_key,
                    "シードから計算した公開鍵 = .pub"
                );
            }
            Reader::SshKeygen => {
                require!("ssh-keygen");
                let (key, pub_path) = files.write_to(dir.path());
                let out = run(Command::new("ssh-keygen").arg("-y").arg("-f").arg(&key));
                assert_eq!(key_part(&stdout(&out)), files.key_part());

                // 秘密鍵・.pub・独立実装のフィンガープリントが一致する
                let fingerprint = |p: &Path| {
                    let out = stdout(&run(Command::new("ssh-keygen").arg("-l").arg("-f").arg(p)));
                    assert_eq!(out.contains("(ED25519)"), true, "{out}");
                    out.split_whitespace().nth(1).unwrap().to_owned()
                };
                let want = sha256_fingerprint(&public_blob(&files.public()));
                assert_eq!(fingerprint(&key), want);
                assert_eq!(fingerprint(&pub_path), want);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // 2. 署名する側 × 検証する側
    // ---------------------------------------------------------------------------

    /// どの実装で署名しても、どの実装でも検証でき、別の鍵・改ざん・別の名前空間では通らない
    #[rstest]
    fn signatures_verify_only_when_genuine(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values(Signer::SshKey, Signer::OpenSsh, Signer::Reference)] signer: Signer,
        #[values(Verifier::SshKey, Verifier::OpenSsh, Verifier::Reference)] verifier: Verifier,
        #[values(
            Check::Genuine,
            Check::OtherKey,
            Check::TamperedMessage,
            Check::OtherNamespace
        )]
        check: Check,
        dir: TempDir,
    ) {
        if matches!(signer, Signer::OpenSsh) || matches!(verifier, Verifier::OpenSsh) {
            require!("ssh-keygen");
        }
        let files = KeyFiles::shared(mode);
        let armored = signer.sign(files, dir.path(), MESSAGE, "sha512");

        let other = KeyFiles::shared(mode.other());
        let mut tampered = MESSAGE.to_vec();
        tampered[0] ^= 1;
        let (pub_line, namespace, message, valid) = match check {
            Check::Genuine => (&files.pub_line, NAMESPACE, MESSAGE, true),
            Check::OtherKey => {
                assert_eq!(
                    other.key_part() == files.key_part(),
                    false,
                    "毎回違う鍵になる"
                );
                (&other.pub_line, NAMESPACE, MESSAGE, false)
            }
            Check::TamperedMessage => (&files.pub_line, NAMESPACE, tampered.as_slice(), false),
            Check::OtherNamespace => (&files.pub_line, "other", MESSAGE, false),
        };
        assert_eq!(
            verifier.verify(dir.path(), &armored, pub_line, namespace, message),
            valid
        );
    }

    // ---------------------------------------------------------------------------
    // 3. 署名を RFC 8032 の参照実装と1ビット単位で突き合わせる
    //
    // Ed25519 の署名は決定的（同じ鍵・同じメッセージなら毎回同じ）なので、鍵ファイルのシードが
    // RFC 8032 どおりに使われていれば、参照実装の署名と完全に一致する。
    // 署名ファイルの分解と署名対象の組み立て（PROTOCOL.sshsig）も、独立実装で行う。
    // ---------------------------------------------------------------------------

    /// ssh-key と OpenSSH の署名ファイルは、仕様から組み立てた署名ファイル（中身は RFC 8032 の参照実装の署名）と
    /// 1バイトも違わない
    #[rstest]
    fn signature_file_is_exactly_rfc_8032(
        #[values(Signer::SshKey, Signer::OpenSsh)] signer: Signer,
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values("sha512", "sha256")] hash: &str,
        #[values(b"".as_slice(), MESSAGE, &[0xa5; 1023])] message: &[u8],
        dir: TempDir,
    ) {
        if matches!(signer, Signer::OpenSsh) {
            require!("ssh-keygen");
        }
        let files = KeyFiles::shared(mode);
        let armored = signer.sign(files, dir.path(), message, hash);

        // まず PROTOCOL.sshsig 2. のフィールドごとに比べる（違ったとき、どこが違うか分かるように）
        let sig = parse_sshsig(&armored).unwrap();
        assert_eq!(sig.version, 1);
        assert_eq!(sig.public_key, files.public(), "署名に入っている公開鍵");
        assert_eq!(sig.namespace, NAMESPACE.as_bytes());
        assert_eq!(sig.reserved, Vec::<u8>::new());
        assert_eq!(sig.hash_algorithm, hash.as_bytes());

        // 鍵ファイルのシード k で、署名対象を参照実装が署名したものと完全に一致する
        let data = sshsig_signed_data(NAMESPACE.as_bytes(), b"", hash.as_bytes(), message);
        assert_eq!(
            sig.signature,
            sign(&files.seed(), &data),
            "RFC 8032 の参照実装と署名が違う"
        );

        // ファイル全体（base64 の折り返しと改行まで）も同じ
        assert_eq!(
            armored,
            Signer::Reference.sign(files, dir.path(), message, hash)
        );
    }

    /// RFC 8032 Section 7.1 の鍵を meikey の形式で書き出すと、OpenSSH がそのまま RFC の鍵として扱う
    #[rstest]
    fn rfc_vector_keys_work_with_openssh(
        #[values("1", "2", "3", "1024", "SHA(abc)")] name: &str,
        dir: TempDir,
    ) {
        require!("ssh-keygen");
        let v = vector(name);
        let files = KeyFiles::from_key(&VanityKey::from_seed(v.secret), "rfc8032");
        let (key, pub_path) = files.write_to(dir.path());

        // ssh-keygen が秘密鍵から復元した公開鍵は、RFC の PUBLIC KEY
        let line = stdout(&run(Command::new("ssh-keygen")
            .arg("-y")
            .arg("-f")
            .arg(&key)));
        let restored = parse_authorized_key(line.trim_end_matches(['\n', '\r'])).unwrap();
        assert_eq!(restored.public_key, v.public);

        // フィンガープリントは SHA-256(RFC 8709 の公開鍵 blob)
        let fp = stdout(&run(Command::new("ssh-keygen")
            .arg("-l")
            .arg("-f")
            .arg(&pub_path)));
        let want = sha256_fingerprint(&public_blob(&v.public));
        assert_eq!(fp.split_whitespace().nth(1), Some(want.as_str()));

        // OpenSSH の署名 = 参照実装の署名（メッセージは RFC のもの）
        let sig =
            parse_sshsig(&Signer::OpenSsh.sign(&files, dir.path(), &v.message, "sha512")).unwrap();
        let data = sshsig_signed_data(NAMESPACE.as_bytes(), b"", b"sha512", &v.message);
        assert_eq!(sig.signature, sign(&v.secret, &data));
    }

    /// 任意のシード・コメントで書いた鍵ファイルを本物の OpenSSH に読ませると、公開鍵とコメントを再現し、
    /// フィンガープリントも独立実装と一致する。コメントの長さが変わるので、パディング 0〜7 バイトをすべて通る
    #[rstest]
    #[case::ascii("[a-zA-Z0-9 _.@+=-]{0,24}")]
    #[case::japanese("[a-z0-9 日本語ｶﾅ]{0,24}")]
    // タブ・改行以外の制御文字・DEL・先頭や末尾の空白（.pub のコメントは行末まで全部）
    #[case::spaces_and_controls("[ \t\u{1}-\u{8}\u{b}\u{c}\u{e}-\u{1f}\u{7f}a-z]{0,24}")]
    // 改行・NUL 以外の任意の Unicode（meikey が受け付けるコメント全部）
    #[case::any_unicode("[^\r\n\u{0}]{0,24}")]
    fn ssh_keygen_restores_public_key_and_comment(#[case] comments: &'static str) {
        require!("ssh-keygen");
        check(24, (any::<[u8; 32]>(), comments), |(seed, comment)| {
            let files = KeyFiles::from_key(&VanityKey::from_seed(seed), &comment);
            let dir = dir();
            let (private, pub_path) = files.write_to(dir.path());

            // 秘密鍵から復元した公開鍵は、.pub の公開鍵とも、シードから ed25519-dalek で計算した公開鍵とも同じ
            let out = run(Command::new("ssh-keygen").arg("-y").arg("-f").arg(&private));
            // 改行だけを取る（trim_end だとコメント末尾の空白まで消える）
            let line = stdout(&out);
            let restored = parse_authorized_key(line.trim_end_matches(['\n', '\r']))
                .map_err(TestCaseError::fail)?;
            let public = SigningKey::from_bytes(&seed).verifying_key().to_bytes();
            prop_assert_eq!(restored.public_key, files.public());
            prop_assert_eq!(restored.public_key, public);
            prop_assert_eq!(restored.comment.unwrap_or_default(), comment);

            // .pub のフィンガープリントも OpenSSH と独立実装で一致
            let fp = stdout(&run(Command::new("ssh-keygen")
                .arg("-l")
                .arg("-f")
                .arg(&pub_path)));
            let want = sha256_fingerprint(&public_blob(&public));
            prop_assert_eq!(fp.split_whitespace().nth(1), Some(want.as_str()));
            Ok(())
        });
    }

    // ---------------------------------------------------------------------------
    // 4. OpenSSH が書く鍵ファイルと1バイトも違わない
    // ---------------------------------------------------------------------------

    /// `ssh-keygen -t ed25519` が作った鍵と同じシード・コメント・checkint を meikey に渡すと、
    /// 秘密鍵も `.pub` も1バイトも違わない（`.pub` は、コメントが空のときに ssh-keygen が付ける最後の空白だけ除く）。コメント（改行・NUL 以外の任意の Unicode）のバイト数を
    /// 8 で割った余りを変えて、パディング 0〜7 バイトをすべて通す
    #[rstest]
    fn meikey_writes_the_same_bytes_as_ssh_keygen(
        #[values(0, 1, 2, 3, 4, 5, 6, 7)] remainder: usize,
    ) {
        require!("ssh-keygen");
        let comments = "[^\r\n\u{0}]{0,40}".prop_map(move |c: String| {
            let fill = (remainder + 8 - c.len() % 8) % 8;
            c + &"x".repeat(fill)
        });
        check(4, comments, |comment| {
            let dir = dir();
            let theirs = ssh_keygen_ed25519(dir.path(), "k", &comment, "", &[]);
            let file = parse_private_key_file(&theirs.private).map_err(TestCaseError::fail)?;
            let key = VanityKey::from_seed(file.keys[0].k);
            prop_assert_eq!(
                private_openssh(&key, &comment, file.checkint1),
                theirs.private
            );
            // ssh-keygen はコメントが空でも、鍵の後ろに区切りの空白を 1 つ書く（meikey は書かない）
            let want = match comment.is_empty() {
                true => theirs
                    .pub_line
                    .strip_suffix(' ')
                    .unwrap_or(&theirs.pub_line),
                false => &theirs.pub_line,
            };
            prop_assert_eq!(key.public_openssh(&comment), want);
            Ok(())
        });
    }

    /// meikey の鍵ファイルを OpenSSH に書き直させても（コメントの変更・空のパスフレーズでの保存し直し）、
    /// meikey が同じ内容で書くファイルと1バイトも違わない
    #[rstest]
    #[case::change_comment(&["-c", "-C", "新しい comment"], "新しい comment")]
    #[case::save_again_without_passphrase(&["-p", "-P", "", "-N", ""], COMMENT)]
    fn ssh_keygen_rewrites_meikey_files_identically(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[case] args: &[&str],
        #[case] comment: &str,
        dir: TempDir,
    ) {
        require!("ssh-keygen");
        let files = KeyFiles::shared(mode);
        let (key, pub_path) = files.write_to(dir.path());
        run(Command::new("ssh-keygen").args(args).arg("-f").arg(&key));

        let rewritten = read_to_string(&key).unwrap();
        let checkint = parse_private_key_file(&rewritten).unwrap().checkint1;
        let vanity = VanityKey::from_seed(files.seed());
        assert_eq!(private_openssh(&vanity, comment, checkint), rewritten);
        assert_eq!(
            read_to_string(&pub_path).unwrap(),
            vanity.public_openssh(comment) + "\n"
        );
    }

    /// meikey の `-C` に渡したコメントは、空白・制御文字・Unicode を含めて、OpenSSH が秘密鍵から
    /// 1バイトも違わず復元する（`ssh-keygen -y` の出力 = meikey の `.pub`）
    #[rstest]
    #[case::spaces_and_controls(" lead\ttab\u{1}ctl\u{1b}esc\u{7f}del  trail ")]
    #[case::unicode("日本語 🔑 e\u{301} \u{200b}\u{feff}")]
    #[case::only_spaces("   ")]
    #[case::empty("")]
    fn cli_comment_survives_openssh(#[case] comment: &str, dir: TempDir) {
        require!("ssh-keygen");
        let out = meikey_in(dir.path(), &[PATTERN, "-j", "2", "-C", comment, "-o", "k"]);
        assert_eq!(out.status.success(), true, "{}", stderr(&out));
        let key = dir.path().join("k");
        let file = parse_private_key_file(&read_to_string(&key).unwrap()).unwrap();
        assert_eq!(file.keys[0].comment, comment.as_bytes());

        let restored = stdout(&run(Command::new("ssh-keygen")
            .arg("-y")
            .arg("-f")
            .arg(&key)));
        assert_eq!(
            restored,
            read_to_string(key.with_added_extension("pub")).unwrap()
        );
    }
}
