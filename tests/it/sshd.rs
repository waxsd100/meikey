//! Docker で sshd のコンテナ（[linuxserver/openssh-server]）を立てて、meikey の鍵で実際にログインする
//! （Unix で Docker が使えるときだけ。使えなければスキップする）
//!
//! 鍵ファイル・ssh-agent の鍵・パスフレーズを付けた鍵（全暗号方式）のどれでもログインでき
//! （sshd のログにもその鍵での公開鍵認証が残る）、登録していない鍵・違うパスフレーズでは入れない。
//! authorized_keys の書き方による違いは [`crate::authorized_keys`] で確かめる。
//!
//! [linuxserver/openssh-server]: https://docs.linuxserver.io/images/docker-openssh-server/

use crate::agent::Agent;
use crate::common::*;
use similar_asserts::assert_eq;
use std::io::Read;
use std::net::TcpStream;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};
use testcontainers::core::{ExecCommand, IntoContainerPort, WaitFor};
use testcontainers::runners::SyncRunner;
use testcontainers::{Container, GenericImage, ImageExt};

const IMAGE: &str = "lscr.io/linuxserver/openssh-server";
const TAG: &str = "10.3_p1-r1-ls239";
/// コンテナの中でログインするユーザー
pub const USER: &str = "meikey";
/// コンテナの中の sshd のポート
const PORT: u16 = 2222;
/// ログインして実行するコマンドと、その出力
pub const COMMAND: &str = "echo meikey-login-ok";
pub const COMMAND_OUTPUT: &str = "meikey-login-ok";

/// テスト用に立てた sshd のコンテナ（drop で止めて消す）
pub struct Sshd {
    container: Container<GenericImage>,
    host: String,
    port: u16,
}

/// `authorized_keys` をそのまま `~/.ssh/authorized_keys` に置いた sshd のコンテナを起動する。
/// Docker が使えない環境なら `Err` で理由を返す
pub fn start(authorized_keys: &str) -> Result<Sshd, String> {
    let container = GenericImage::new(IMAGE, TAG)
        .with_exposed_port(PORT.tcp())
        .with_wait_for(WaitFor::message_on_stdout(format!(
            "sshd is listening on port {PORT}"
        )))
        .with_env_var("USER_NAME", USER)
        .with_env_var("PASSWORD_ACCESS", "false")
        // 起動時の設定スクリプトは、このファイルがあれば中身に触れない（所有者とパーミッションだけ直す）
        .with_copy_to(
            "/config/.ssh/authorized_keys",
            authorized_keys.as_bytes().to_vec(),
        )
        .start()
        .map_err(|e| format!("sshd のコンテナを起動できない: {e}"))?;
    let host = container.get_host().map_err(|e| e.to_string())?.to_string();
    let port = container
        .get_host_port_ipv4(PORT.tcp())
        .map_err(|e| e.to_string())?;
    wait_for_banner(&host, port)?;
    Ok(Sshd {
        container,
        host,
        port,
    })
}

/// [`start`] して、Docker が使えなければ理由を出して None（テストはスキップする）
pub fn start_or_skip(authorized_keys: &str) -> Option<Sshd> {
    start(authorized_keys)
        .inspect_err(|e| eprintln!("スキップ: {e}"))
        .ok()
}

/// sshd が接続を受け付けて `SSH-` で始まるバナーを返すまで待つ
/// （コンテナが起動の完了を出力してから、sshd が受け付けるまでに少し間がある）
fn wait_for_banner(host: &str, port: u16) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let mut banner = [0u8; 4];
        let ready = TcpStream::connect((host, port))
            .and_then(|mut c| c.read_exact(&mut banner))
            .is_ok_and(|()| &banner == b"SSH-");
        if ready {
            return Ok(());
        }
        if Instant::now() > deadline {
            return Err("sshd のコンテナが接続を受け付けない".into());
        }
        sleep(Duration::from_millis(100));
    }
}

/// ログインに使う鍵
#[derive(Clone, Copy)]
pub enum Identity<'a> {
    /// 暗号化されていない秘密鍵ファイル（`ssh -i`）
    File(&'a Path),
    /// パスフレーズを付けた秘密鍵ファイル。パスフレーズは `askpass` が答える
    Encrypted { key: &'a Path, askpass: &'a Path },
    /// ssh-agent の中の鍵（`-i` には `.pub` を渡して、どの鍵を使うか決める）
    Agent { agent: &'a Agent, public: &'a Path },
}

/// `identity` の鍵だけを使って公開鍵認証でログインし、`command` を実行する
pub fn ssh(sshd: &Sshd, identity: Identity, dir: &Path, command: &str) -> Output {
    let mut cmd = Command::new("ssh");
    cmd.args(["-F", "none"]) // ユーザーの ~/.ssh/config を読まない
        .arg("-p")
        .arg(sshd.port.to_string())
        .args([
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "PreferredAuthentications=publickey",
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
        .stdin(Stdio::null());
    match identity {
        Identity::File(key) => {
            cmd.arg("-i").arg(key);
            cmd.args(["-o", "IdentityAgent=none", "-o", "BatchMode=yes"]);
        }
        Identity::Encrypted { key, askpass } => {
            cmd.arg("-i").arg(key);
            // BatchMode だとパスフレーズを聞かないので、askpass に聞く
            cmd.args(["-o", "IdentityAgent=none", "-o", "BatchMode=no"]);
            cmd.args(["-o", "NumberOfPasswordPrompts=1"]);
            with_askpass(&mut cmd, askpass);
        }
        Identity::Agent { agent, public } => {
            cmd.arg("-i").arg(public);
            cmd.arg("-o")
                .arg(format!("IdentityAgent={}", agent.sock().display()));
            cmd.args(["-o", "BatchMode=yes"]);
        }
    }
    cmd.arg(format!("{USER}@{}", sshd.host))
        .arg(command)
        .output()
        .unwrap()
}

/// コンテナの中の sshd のログ
pub fn log(sshd: &Sshd) -> String {
    let mut out = sshd
        .container
        .exec(ExecCommand::new(["cat", "/config/logs/openssh/current"]))
        .unwrap();
    String::from_utf8_lossy(&out.stdout_to_vec().unwrap()).into_owned()
}

/// ログインできて `command` の出力が `want` だった。sshd のログにも `pub_line` の鍵
/// （SHA256 フィンガープリント）での公開鍵認証が残っている
pub fn assert_logged_in(sshd: &Sshd, out: &Output, pub_line: &str, want: &str) {
    let log = log(sshd);
    assert_eq!(
        out.status.success(),
        true,
        "ログインに失敗\nssh: {}\nsshd: {log}",
        stderr(out)
    );
    assert_eq!(stdout(out).trim_end_matches('\n'), want);
    let fp = fingerprint(pub_line);
    assert_eq!(
        log.lines()
            .any(|l| l.contains("Accepted publickey") && l.contains(&fp)),
        true,
        "sshd のログに {fp} での認証がない\n{log}"
    );
}

/// ログインできなかった（公開鍵認証で断られた）
pub fn assert_denied(sshd: &Sshd, out: &Output) {
    assert_eq!(
        out.status.success(),
        false,
        "ログインできてしまった\nsshd: {}",
        log(sshd)
    );
    assert_eq!(
        stderr(out).contains("Permission denied"),
        true,
        "{}",
        stderr(out)
    );
}

mod tests {
    use super::*;
    use crate::agent::Load;
    use crate::encrypted::{PASSPHRASE, add_passphrase};
    use rstest::rstest;
    use similar_asserts::assert_eq;
    use std::fs::{create_dir, write};
    use tempfile::TempDir;

    /// ログインの仕方
    #[derive(Debug, Clone, Copy)]
    enum Way {
        /// 鍵ファイル
        File,
        /// ssh-agent（[`Load`] の方法で入れる）
        Agent(Load),
        /// この暗号方式でパスフレーズを付けた鍵ファイル
        Encrypted(&'static str),
    }

    /// 登録した鍵ではログインでき、登録していない鍵（もう一方の方式で作った鍵）では入れない
    /// （＝ ログインできたのは本当にこの鍵で認証しているから）
    #[rstest]
    fn sshd_accepts_only_the_registered_key(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        dir: TempDir,
    ) {
        require!("ssh");
        let (registered, other) = (KeyFiles::shared(mode), KeyFiles::shared(mode.other()));
        let Some(server) = start_or_skip(&format!("{}\n", registered.pub_line)) else {
            return;
        };
        let dir = dir.path();
        let (key_dir, other_dir) = (dir.join("registered"), dir.join("other"));
        create_dir(&key_dir).unwrap();
        create_dir(&other_dir).unwrap();

        let (key, _) = registered.write_to(&key_dir);
        let out = ssh(&server, Identity::File(&key), dir, COMMAND);
        assert_logged_in(&server, &out, &registered.pub_line, COMMAND_OUTPUT);

        let (other_key, _) = other.write_to(&other_dir);
        let out = ssh(&server, Identity::File(&other_key), dir, COMMAND);
        assert_denied(&server, &out);
    }

    /// 鍵ファイル・ssh-agent（ssh-add で入れても、プロトコルで直接入れても）・パスフレーズを付けた鍵
    /// （全暗号方式）のどれでもログインできる
    #[rstest]
    fn login_with_every_kind_of_identity(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values(
            Way::File,
            Way::Agent(Load::SshAdd),
            Way::Agent(Load::Protocol),
            Way::Encrypted("3des-cbc"),
            Way::Encrypted("aes128-cbc"),
            Way::Encrypted("aes192-cbc"),
            Way::Encrypted("aes256-cbc"),
            Way::Encrypted("aes128-ctr"),
            Way::Encrypted("aes192-ctr"),
            Way::Encrypted("aes256-ctr"),
            Way::Encrypted("aes128-gcm@openssh.com"),
            Way::Encrypted("aes256-gcm@openssh.com"),
            Way::Encrypted("chacha20-poly1305@openssh.com")
        )]
        way: Way,
        dir: TempDir,
    ) {
        require!("ssh");
        let files = KeyFiles::shared(mode);
        let Some(server) = start_or_skip(&format!("{}\n", files.pub_line)) else {
            return;
        };
        let dir = dir.path();
        let key_dir = dir.join("key");
        create_dir(&key_dir).unwrap();

        let out = match way {
            Way::File => {
                let (key, _) = files.write_to(&key_dir);
                ssh(&server, Identity::File(&key), dir, COMMAND)
            }
            Way::Agent(load) => {
                let agent = Agent::start();
                load.load(&agent, files, &key_dir);
                // 秘密鍵ファイルの無いディレクトリに .pub だけ置き、それで鍵を選ぶ
                let pub_dir = dir.join("pub");
                create_dir(&pub_dir).unwrap();
                let public = pub_dir.join("id_ed25519.pub");
                write(&public, format!("{}\n", files.pub_line)).unwrap();
                ssh(
                    &server,
                    Identity::Agent {
                        agent: &agent,
                        public: &public,
                    },
                    dir,
                    COMMAND,
                )
            }
            Way::Encrypted(cipher) => {
                let (key, _) = files.write_to(&key_dir);
                add_passphrase(&key, PASSPHRASE, &["-Z", cipher, "-a", "2"]);
                let askpass = write_askpass(dir, PASSPHRASE);
                ssh(
                    &server,
                    Identity::Encrypted {
                        key: &key,
                        askpass: &askpass,
                    },
                    dir,
                    COMMAND,
                )
            }
        };
        assert_logged_in(&server, &out, &files.pub_line, COMMAND_OUTPUT);
    }

    /// パスフレーズを付けた鍵は、違うパスフレーズではログインに使えない
    #[rstest]
    fn wrong_passphrase_cannot_log_in(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        dir: TempDir,
    ) {
        require!("ssh");
        let files = KeyFiles::shared(mode);
        let Some(server) = start_or_skip(&format!("{}\n", files.pub_line)) else {
            return;
        };
        let dir = dir.path();
        let key_dir = dir.join("key");
        create_dir(&key_dir).unwrap();
        let (key, _) = files.write_to(&key_dir);
        add_passphrase(&key, PASSPHRASE, &["-a", "2"]);
        let askpass = write_askpass(dir, "wrong");
        let out = ssh(
            &server,
            Identity::Encrypted {
                key: &key,
                askpass: &askpass,
            },
            dir,
            COMMAND,
        );
        assert_denied(&server, &out);
        assert_eq!(log(&server).contains("Accepted publickey"), false);
    }
}
