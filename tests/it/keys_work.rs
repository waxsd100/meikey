//! meikey で作った鍵が「本当に SSH 鍵として使えるか」を確かめるテスト（[`tests`]）と、そのヘルパー
//!
//! 1. 鍵ファイルを ssh-key・仕様書から書いた独立実装・OpenSSH の ssh-keygen で読む（[`Reader`]）
//! 2. 署名する側（[`Signer`]）× 検証する側（[`Verifier`]）の総当たりで、
//!    正しい署名だけが通り、別の鍵・改ざん・別の名前空間では通らない（[`Check`]）
//! 3. 署名を RFC 8032 の参照実装と1ビット単位で突き合わせる
//! 4. 一時的な sshd を立てて実際にログインする（Unix で sshd があるときだけ。[`sshd`]）
//!
//! 鍵は meikey 本体で高速版と定数時間版（`--constant-time`）の両方で作り、
//! テストバイナリの中で使い回す（[`KeyFiles::shared`]）。
//! OpenSSH を使うケースは、OpenSSH が無い環境ではスキップする。

use crate::common::*;
use ssh_key::{HashAlg, LineEnding, PrivateKey, PublicKey, SshSig};
use std::path::Path;
use std::process::Command;

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
    /// README の手順（`ssh-keygen -p`）でパスフレーズを付けると、そのパスフレーズでだけ開ける
    Passphrase(&'static str),
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
                let work = tempfile::tempdir_in(dir).unwrap();
                let (key, _) = files.write_to(work.path());
                let msg = work.path().join("message");
                std::fs::write(&msg, message).unwrap();
                let mut cmd = Command::new("ssh-keygen");
                cmd.args(["-Y", "sign", "-n", NAMESPACE]);
                // SHA-512 は -Y sign の既定。-O hashalg を知らない古い OpenSSH でも動くよう、既定なら付けない
                if hash != "sha512" {
                    cmd.arg("-O").arg(format!("hashalg={hash}"));
                }
                run(cmd.arg("-f").arg(&key).arg(&msg));
                std::fs::read_to_string(msg.with_added_extension("sig")).unwrap()
            }
            Signer::Reference => {
                let data = ssh_format::sshsig_signed_data(
                    NAMESPACE.as_bytes(),
                    b"",
                    hash.as_bytes(),
                    message,
                );
                ssh_format::armor_sshsig(&ssh_format::SshSig {
                    version: 1,
                    public_key: files.public(),
                    namespace: NAMESPACE.into(),
                    reserved: Vec::new(),
                    hash_algorithm: hash.into(),
                    signature: rfc8032::sign(&files.seed(), &data),
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
                std::fs::write(&sig, armored).unwrap();
                std::fs::write(&msg, message).unwrap();
                Command::new("ssh-keygen")
                    .args(["-Y", "verify", "-n", namespace, "-I", COMMENT, "-f"])
                    .arg(&signers)
                    .arg("-s")
                    .arg(&sig)
                    .stdin(std::fs::File::open(&msg).unwrap())
                    .output()
                    .unwrap()
                    .status
                    .success()
            }
            Verifier::Reference => {
                let sig = ssh_format::parse_sshsig(armored).unwrap();
                let public = ssh_format::parse_authorized_key(pub_line)
                    .unwrap()
                    .public_key;
                // 署名対象は「検証する側が期待する」名前空間とメッセージで組み立てる
                let data = ssh_format::sshsig_signed_data(
                    namespace.as_bytes(),
                    &sig.reserved,
                    &sig.hash_algorithm,
                    message,
                );
                rfc8032::verify(&public, &data, &sig.signature)
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

// ---------------------------------------------------------------------------
// 4. 実際に SSH でログインする
// ---------------------------------------------------------------------------

/// テスト用に一時的な sshd を立てて、ssh でログインする（Unix だけ）
#[cfg(unix)]
pub mod sshd {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::path::PathBuf;
    use std::process::{Child, Stdio};
    use std::time::{Duration, Instant};

    /// テスト用に立てた sshd（drop で止める）
    pub struct Sshd {
        child: Child,
        port: u16,
    }

    impl Drop for Sshd {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    fn find_sshd() -> Option<PathBuf> {
        [
            "/usr/sbin/sshd",
            "/usr/local/sbin/sshd",
            "/opt/homebrew/sbin/sshd",
            "/usr/bin/sshd",
        ]
        .into_iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
    }

    fn free_port() -> u16 {
        TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    pub fn current_user() -> String {
        stdout(&run(Command::new("id").arg("-un")))
            .trim()
            .to_owned()
    }

    /// `authorized_keys` に `authorized` を登録した sshd を 127.0.0.1 で起動する。
    /// 起動できない環境（権限・設定）なら `Err` で理由を返す。
    pub fn start(dir: &Path, authorized: &[&str]) -> Result<Sshd, String> {
        let sshd = find_sshd().ok_or("sshd が見つかりません")?;
        let host_key = dir.join("host_ed25519");
        run(Command::new("ssh-keygen")
            .args(["-q", "-t", "ed25519", "-N", "", "-f"])
            .arg(&host_key));
        std::fs::write(dir.join("authorized_keys"), authorized.join("\n") + "\n").unwrap();

        let port = free_port();
        let config = dir.join("sshd_config");
        std::fs::write(
            &config,
            format!(
                "Port {port}\n\
                 ListenAddress 127.0.0.1\n\
                 HostKey {host}\n\
                 PidFile {dir}/sshd.pid\n\
                 AuthorizedKeysFile {dir}/authorized_keys\n\
                 PubkeyAuthentication yes\n\
                 PasswordAuthentication no\n\
                 KbdInteractiveAuthentication no\n\
                 PermitRootLogin prohibit-password\n\
                 StrictModes no\n\
                 UsePAM no\n\
                 LogLevel VERBOSE\n",
                host = host_key.display(),
                dir = dir.display(),
            ),
        )
        .unwrap();

        // 設定と権限のチェック（root で /run/sshd がない、などはここで分かる）
        let check = Command::new(&sshd)
            .arg("-t")
            .arg("-f")
            .arg(&config)
            .output()
            .unwrap();
        if !check.status.success() {
            return Err(format!("sshd -t が失敗: {}", stderr(&check).trim()));
        }

        let child = Command::new(&sshd)
            .arg("-D")
            .arg("-e")
            .arg("-f")
            .arg(&config)
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(dir.join("sshd.log")).unwrap())
            .spawn()
            .map_err(|e| e.to_string())?;
        let sshd = Sshd { child, port };

        let deadline = Instant::now() + Duration::from_secs(10);
        while TcpStream::connect(("127.0.0.1", port)).is_err() {
            if Instant::now() > deadline {
                return Err("sshd が起動しない".into());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(sshd)
    }

    pub fn ssh(
        sshd: &Sshd,
        key: &Path,
        dir: &Path,
        user: &str,
        command: &str,
    ) -> std::process::Output {
        Command::new("ssh")
            .arg("-F")
            .arg("/dev/null") // ユーザーの ~/.ssh/config を読まない
            .arg("-i")
            .arg(key)
            .arg("-p")
            .arg(sshd.port.to_string())
            .args([
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "IdentityAgent=none",
                "-o",
                "BatchMode=yes",
                "-o",
                "PasswordAuthentication=no",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=10",
            ])
            .arg("-o")
            .arg(format!(
                "UserKnownHostsFile={}",
                dir.join("known_hosts").display()
            ))
            .arg(format!("{user}@127.0.0.1"))
            .arg(command)
            .stdin(Stdio::null())
            .output()
            .unwrap()
    }

    pub fn log(dir: &Path) -> String {
        std::fs::read_to_string(dir.join("sshd.log")).unwrap_or_default()
    }
}

mod tests {
    use super::*;
    use crate::common::property::check;
    use proptest::prelude::*;
    use rstest::rstest;

    // ---------------------------------------------------------------------------
    // 1. 鍵ファイルを読む
    // ---------------------------------------------------------------------------

    /// 秘密鍵と `.pub` が同じ鍵を表していて、秘密鍵の中の公開鍵もシードから計算したものと一致する
    #[rstest]
    fn key_files_work(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values(
            Reader::SshKey,
            Reader::Spec,
            Reader::SshKeygen,
            Reader::Passphrase("x"),
            Reader::Passphrase("correct horse battery staple"),
            Reader::Passphrase("p@ss/w0rd+=")
        )]
        reader: Reader,
    ) {
        let files = KeyFiles::shared(mode);
        match reader {
            Reader::SshKey => {
                let private = PrivateKey::from_openssh(&files.private).expect("秘密鍵として読める");
                let public = PublicKey::from_openssh(&files.pub_line).expect("公開鍵として読める");
                assert!(!private.is_encrypted());
                assert_eq!(private.comment(), COMMENT);
                assert_eq!(public.comment(), COMMENT);
                assert_eq!(private.public_key().key_data(), public.key_data());

                let pair = private.key_data().ed25519().unwrap();
                let recomputed =
                    ed25519_dalek::SigningKey::from_bytes(&pair.private.to_bytes()).verifying_key();
                assert_eq!(recomputed.to_bytes(), pair.public.0);
            }
            Reader::Spec => {
                let file = ssh_format::parse_private_key_file(&files.private).unwrap();
                assert_eq!(file.violations(), Vec::<String>::new());
                let line = ssh_format::parse_authorized_key(&files.pub_line).unwrap();
                assert_eq!(line.comment.as_deref(), Some(COMMENT));

                let key = &file.keys[0];
                assert_eq!(key.comment, COMMENT.as_bytes());
                assert_eq!(key.enc_a, line.public_key, "秘密鍵の中の公開鍵 = .pub");
                assert_eq!(
                    rfc8032::secret_to_public(&key.k),
                    line.public_key,
                    "シードから計算した公開鍵 = .pub"
                );
            }
            Reader::SshKeygen => {
                require!("ssh-keygen");
                let dir = tempfile::tempdir().unwrap();
                let (key, pub_path) = files.write_to(dir.path());
                let out = run(Command::new("ssh-keygen").arg("-y").arg("-f").arg(&key));
                assert_eq!(key_part(&stdout(&out)), files.key_part());

                // 秘密鍵・.pub・独立実装のフィンガープリントが一致する
                let fingerprint = |p: &Path| {
                    let out = stdout(&run(Command::new("ssh-keygen").arg("-l").arg("-f").arg(p)));
                    assert!(out.contains("(ED25519)"), "{out}");
                    out.split_whitespace().nth(1).unwrap().to_owned()
                };
                let want =
                    ssh_format::sha256_fingerprint(&ssh_format::public_blob(&files.public()));
                assert_eq!(fingerprint(&key), want);
                assert_eq!(fingerprint(&pub_path), want);
            }
            Reader::Passphrase(passphrase) => {
                require!("ssh-keygen");
                let dir = tempfile::tempdir().unwrap();
                let (key, _) = files.write_to(dir.path());
                // -a 1: 鍵導出（bcrypt）の回数を最小にする。確かめたいのはパスフレーズの扱いで、bcrypt の強さではない
                run(Command::new("ssh-keygen")
                    .args(["-p", "-a", "1", "-P", "", "-N", passphrase, "-f"])
                    .arg(&key));
                assert!(PrivateKey::read_openssh_file(&key).unwrap().is_encrypted());

                let open = |p: &str| {
                    Command::new("ssh-keygen")
                        .args(["-y", "-P", p, "-f"])
                        .arg(&key)
                        .output()
                        .unwrap()
                };
                let out = open(passphrase);
                assert!(out.status.success(), "{}", stderr(&out));
                assert_eq!(key_part(&stdout(&out)), files.key_part());
                assert!(
                    !open("wrong").status.success(),
                    "違うパスフレーズでは開けない"
                );
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
    ) {
        if matches!(signer, Signer::OpenSsh) || matches!(verifier, Verifier::OpenSsh) {
            require!("ssh-keygen");
        }
        let files = KeyFiles::shared(mode);
        let dir = tempfile::tempdir().unwrap();
        let armored = signer.sign(files, dir.path(), MESSAGE, "sha512");

        let other = KeyFiles::shared(mode.other());
        let mut tampered = MESSAGE.to_vec();
        tampered[0] ^= 1;
        let (pub_line, namespace, message, valid) = match check {
            Check::Genuine => (&files.pub_line, NAMESPACE, MESSAGE, true),
            Check::OtherKey => {
                assert_ne!(other.key_part(), files.key_part(), "毎回違う鍵になる");
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
    ) {
        if matches!(signer, Signer::OpenSsh) {
            require!("ssh-keygen");
        }
        let files = KeyFiles::shared(mode);
        let dir = tempfile::tempdir().unwrap();
        let armored = signer.sign(files, dir.path(), message, hash);

        // まず PROTOCOL.sshsig 2. のフィールドごとに比べる（違ったとき、どこが違うか分かるように）
        let sig = ssh_format::parse_sshsig(&armored).unwrap();
        assert_eq!(sig.version, 1);
        assert_eq!(sig.public_key, files.public(), "署名に入っている公開鍵");
        assert_eq!(sig.namespace, NAMESPACE.as_bytes());
        assert!(sig.reserved.is_empty());
        assert_eq!(sig.hash_algorithm, hash.as_bytes());

        // 鍵ファイルのシード k で、署名対象を参照実装が署名したものと完全に一致する
        let data =
            ssh_format::sshsig_signed_data(NAMESPACE.as_bytes(), b"", hash.as_bytes(), message);
        assert_eq!(
            sig.signature,
            rfc8032::sign(&files.seed(), &data),
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
    fn rfc_vector_keys_work_with_openssh(#[values("1", "2", "3", "1024", "SHA(abc)")] name: &str) {
        require!("ssh-keygen");
        let v = rfc8032::vector(name);
        let files = KeyFiles::from_key(&meikey::VanityKey::from_seed(v.secret), "rfc8032");
        let dir = tempfile::tempdir().unwrap();
        let (key, pub_path) = files.write_to(dir.path());

        // ssh-keygen が秘密鍵から復元した公開鍵は、RFC の PUBLIC KEY
        let line = stdout(&run(Command::new("ssh-keygen")
            .arg("-y")
            .arg("-f")
            .arg(&key)));
        let restored =
            ssh_format::parse_authorized_key(line.strip_suffix('\n').unwrap_or(&line)).unwrap();
        assert_eq!(restored.public_key, v.public);

        // フィンガープリントは SHA-256(RFC 8709 の公開鍵 blob)
        let fp = stdout(&run(Command::new("ssh-keygen")
            .arg("-l")
            .arg("-f")
            .arg(&pub_path)));
        let want = ssh_format::sha256_fingerprint(&ssh_format::public_blob(&v.public));
        assert_eq!(fp.split_whitespace().nth(1), Some(want.as_str()));

        // OpenSSH の署名 = 参照実装の署名（メッセージは RFC のもの）
        let sig = ssh_format::parse_sshsig(&Signer::OpenSsh.sign(
            &files,
            dir.path(),
            &v.message,
            "sha512",
        ))
        .unwrap();
        let data = ssh_format::sshsig_signed_data(NAMESPACE.as_bytes(), b"", b"sha512", &v.message);
        assert_eq!(sig.signature, rfc8032::sign(&v.secret, &data));
    }

    /// 任意のシード・コメントで書いた鍵ファイルを本物の OpenSSH に読ませると、公開鍵とコメントを再現し、
    /// フィンガープリントも独立実装と一致する。コメントの長さが変わるので、パディング 0〜7 バイトをすべて通る
    #[rstest]
    #[case::ascii("[a-zA-Z0-9 _.@+=-]{0,24}")]
    #[case::japanese("[a-z0-9 日本語ｶﾅ]{0,24}")]
    fn ssh_keygen_restores_public_key_and_comment(#[case] comments: &'static str) {
        require!("ssh-keygen");
        check(24, (any::<[u8; 32]>(), comments), |(seed, comment)| {
            let files = KeyFiles::from_key(&meikey::VanityKey::from_seed(seed), &comment);
            let dir = tempfile::tempdir().unwrap();
            let (private, pub_path) = files.write_to(dir.path());

            // 秘密鍵から復元した公開鍵は、.pub の公開鍵とも、シードから ed25519-dalek で計算した公開鍵とも同じ
            let out = run(Command::new("ssh-keygen").arg("-y").arg("-f").arg(&private));
            // 改行だけを取る（trim_end だとコメント末尾の空白まで消える）
            let line = stdout(&out);
            let restored =
                ssh_format::parse_authorized_key(line.strip_suffix('\n').unwrap_or(&line))
                    .map_err(TestCaseError::fail)?;
            let public = ed25519_dalek::SigningKey::from_bytes(&seed)
                .verifying_key()
                .to_bytes();
            prop_assert_eq!(restored.public_key, files.public());
            prop_assert_eq!(restored.public_key, public);
            prop_assert_eq!(restored.comment.unwrap_or_default(), comment);

            // .pub のフィンガープリントも OpenSSH と独立実装で一致
            let fp = stdout(&run(Command::new("ssh-keygen")
                .arg("-l")
                .arg("-f")
                .arg(&pub_path)));
            let want = ssh_format::sha256_fingerprint(&ssh_format::public_blob(&public));
            prop_assert_eq!(fp.split_whitespace().nth(1), Some(want.as_str()));
            Ok(())
        });
    }

    // ---------------------------------------------------------------------------
    // 4. 実際に SSH でログインする
    // ---------------------------------------------------------------------------

    /// 登録した鍵ではログインでき（sshd のログにもその鍵での公開鍵認証が残る）、
    /// 登録していない鍵（もう一方の方式で作った鍵）では Permission denied になる
    #[cfg(unix)]
    #[rstest]
    fn sshd_accepts_only_the_registered_key(#[values(Mode::Fast, Mode::ConstantTime)] mode: Mode) {
        require!("ssh-keygen");
        require!("ssh");
        let (registered, other) = (KeyFiles::shared(mode), KeyFiles::shared(mode.other()));
        let dir = tempfile::tempdir().unwrap();
        let dir = dir.path();
        let Ok(server) =
            sshd::start(dir, &[&registered.pub_line]).inspect_err(|e| eprintln!("スキップ: {e}"))
        else {
            return;
        };
        let user = sshd::current_user();

        // 登録した鍵: ログインしてコマンドが動く
        let key_dir = dir.join("registered");
        std::fs::create_dir(&key_dir).unwrap();
        let (key, _) = registered.write_to(&key_dir);
        let out = sshd::ssh(&server, &key, dir, &user, "echo meikey-login-ok");
        assert!(
            out.status.success(),
            "ログインに失敗\nssh: {}\nsshd: {}",
            stderr(&out),
            sshd::log(dir)
        );
        assert_eq!(stdout(&out).trim(), "meikey-login-ok");
        // sshd 側でも、この鍵（SHA256 フィンガープリント）の公開鍵認証で通っている
        let fp = ssh_format::sha256_fingerprint(&ssh_format::public_blob(&registered.public()));
        let log = sshd::log(dir);
        assert!(
            log.contains("Accepted publickey") && log.contains(&fp),
            "sshd のログに {fp} での認証がない\n{log}"
        );

        // 登録していない鍵: 入れない（＝ 上のログインは本当にこの鍵で認証している）
        let other_dir = dir.join("other");
        std::fs::create_dir(&other_dir).unwrap();
        let (other_key, _) = other.write_to(&other_dir);
        let out = sshd::ssh(&server, &other_key, dir, &user, "echo should-not-run");
        assert!(!out.status.success());
        assert!(
            stderr(&out).contains("Permission denied"),
            "{}",
            stderr(&out)
        );
    }
}
