//! meikey の鍵を ssh-agent に入れて使えるか（Unix だけ。Windows の OpenSSH の ssh-agent は名前付きパイプで待ち受ける）
//!
//! 鍵の入れ方は2つ（[`Load`]）: `ssh-add` で鍵ファイルから入れるのと、draft-ietf-sshm-ssh-agent の
//! SSH_AGENTC_ADD_IDENTITY を独立実装（[`AgentClient`]）で組み立て、鍵ファイルの中身（draft 3.2.3 の形）を
//! そのまま送るの。どちらでも、一覧（`ssh-add -L`・SSH_AGENTC_REQUEST_IDENTITIES）は `.pub` と同じで、
//! 署名（SSH_AGENTC_SIGN_REQUEST・`ssh-keygen -Y sign`）は RFC 8032 の参照実装と1ビットも違わない。
//! ssh-agent の鍵でログインするのは [`crate::sshd`] で確かめる。

use crate::common::ssh_format::{
    Ed25519PrivateKey, Reader, parse_private_key_file, put_string, put_uint32,
};
use crate::common::*;
use similar_asserts::assert_eq;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};
use tempfile::{TempDir, tempdir};

// draft-ietf-sshm-ssh-agent のメッセージ番号
pub const SSH_AGENT_FAILURE: u8 = 5;
pub const SSH_AGENT_SUCCESS: u8 = 6;
pub const SSH_AGENTC_REQUEST_IDENTITIES: u8 = 11;
pub const SSH_AGENT_IDENTITIES_ANSWER: u8 = 12;
pub const SSH_AGENTC_SIGN_REQUEST: u8 = 13;
pub const SSH_AGENT_SIGN_RESPONSE: u8 = 14;
pub const SSH_AGENTC_ADD_IDENTITY: u8 = 17;
pub const SSH_AGENTC_REMOVE_IDENTITY: u8 = 18;

/// テスト用に起動した ssh-agent（drop で止める）
pub struct Agent {
    child: Child,
    sock: PathBuf,
    _dir: TempDir,
}

impl Agent {
    /// `ssh-agent -D -a <ソケット>` を起動し、ソケットにつながるまで待つ
    pub fn start() -> Agent {
        let dir = tempdir().unwrap();
        let sock = dir.path().join("agent.sock");
        let child = Command::new("ssh-agent")
            .arg("-D")
            .arg("-a")
            .arg(&sock)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while UnixStream::connect(&sock).is_err() {
            assert_eq!(Instant::now() < deadline, true, "ssh-agent が起動しない");
            sleep(Duration::from_millis(20));
        }
        Agent {
            child,
            sock,
            _dir: dir,
        }
    }

    pub fn sock(&self) -> &Path {
        &self.sock
    }

    /// この ssh-agent を使う OpenSSH のコマンド（`SSH_AUTH_SOCK` を設定する）
    pub fn command(&self, program: &str) -> Command {
        let mut cmd = Command::new(program);
        cmd.env("SSH_AUTH_SOCK", &self.sock).stdin(Stdio::null());
        cmd
    }

    pub fn client(&self) -> AgentClient {
        AgentClient(UnixStream::connect(&self.sock).unwrap())
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// draft-ietf-sshm-ssh-agent のメッセージを送る、仕様から書いたクライアント
pub struct AgentClient(UnixStream);

impl AgentClient {
    /// メッセージは `uint32 長さ`、`byte 種類`、中身。返事も同じ形
    fn request(&mut self, kind: u8, contents: &[u8]) -> (u8, Vec<u8>) {
        let mut msg = Vec::new();
        put_uint32(&mut msg, (contents.len() + 1).try_into().unwrap());
        msg.push(kind);
        msg.extend(contents);
        self.0.write_all(&msg).unwrap();
        let mut len = [0u8; 4];
        self.0.read_exact(&mut len).unwrap();
        let mut reply = vec![0u8; u32::from_be_bytes(len) as usize];
        self.0.read_exact(&mut reply).unwrap();
        (reply[0], reply[1..].to_vec())
    }

    /// SSH_AGENTC_REQUEST_IDENTITIES: 入っている鍵の（公開鍵 blob, コメント）の一覧
    pub fn identities(&mut self) -> Vec<(Vec<u8>, Vec<u8>)> {
        let (kind, body) = self.request(SSH_AGENTC_REQUEST_IDENTITIES, &[]);
        assert_eq!(kind, SSH_AGENT_IDENTITIES_ANSWER);
        let mut r = Reader::new(&body);
        let n = r.uint32().unwrap();
        let list = (0..n)
            .map(|_| (r.string().unwrap().to_vec(), r.string().unwrap().to_vec()))
            .collect();
        assert_eq!(r.is_empty(), true);
        list
    }

    /// SSH_AGENTC_SIGN_REQUEST（flags は 0）。署名は RFC 8709 Section 6 の形
    /// （`string "ssh-ed25519"`、`string 署名`）で返るので、その 64 バイトを返す。断られたら None
    pub fn sign(&mut self, blob: &[u8], data: &[u8]) -> Option<[u8; 64]> {
        let mut body = Vec::new();
        put_string(&mut body, blob);
        put_string(&mut body, data);
        put_uint32(&mut body, 0);
        let (kind, reply) = self.request(SSH_AGENTC_SIGN_REQUEST, &body);
        if kind == SSH_AGENT_FAILURE {
            return None;
        }
        assert_eq!(kind, SSH_AGENT_SIGN_RESPONSE);
        let mut r = Reader::new(&reply);
        let mut s = Reader::new(r.string().unwrap());
        assert_eq!(r.is_empty(), true);
        assert_eq!(s.string().unwrap(), b"ssh-ed25519");
        let signature = s.string().unwrap().try_into().unwrap();
        assert_eq!(s.is_empty(), true);
        Some(signature)
    }

    /// SSH_AGENTC_ADD_IDENTITY: 鍵の中身（draft 3.2.3）とコメント
    pub fn add(&mut self, key: &Ed25519PrivateKey) -> bool {
        let mut body = key.agent_encoding();
        put_string(&mut body, &key.comment);
        self.reply_is_success(SSH_AGENTC_ADD_IDENTITY, &body)
    }

    /// SSH_AGENTC_REMOVE_IDENTITY
    pub fn remove(&mut self, blob: &[u8]) -> bool {
        let mut body = Vec::new();
        put_string(&mut body, blob);
        self.reply_is_success(SSH_AGENTC_REMOVE_IDENTITY, &body)
    }

    fn reply_is_success(&mut self, kind: u8, body: &[u8]) -> bool {
        let (reply, rest) = self.request(kind, body);
        assert_eq!(rest, Vec::<u8>::new());
        match reply {
            SSH_AGENT_SUCCESS => true,
            SSH_AGENT_FAILURE => false,
            other => panic!("SUCCESS でも FAILURE でもない返事 {other}"),
        }
    }
}

/// 鍵を ssh-agent に入れる方法
#[derive(Debug, Clone, Copy)]
pub enum Load {
    /// `ssh-add` で鍵ファイルから
    SshAdd,
    /// 鍵ファイルを独立実装で読み、中身をそのまま SSH_AGENTC_ADD_IDENTITY で送る
    Protocol,
}

impl Load {
    /// `files` の鍵を `agent` に入れる（`ssh-add` に渡す鍵ファイルは `dir` に書く）
    pub fn load(self, agent: &Agent, files: &KeyFiles, dir: &Path) {
        match self {
            Load::SshAdd => {
                let (key, _) = files.write_to(dir);
                let out = run(agent.command("ssh-add").arg(&key));
                assert_eq!(
                    stderr(&out).contains(&format!("({COMMENT})")),
                    true,
                    "{}",
                    stderr(&out)
                );
            }
            Load::Protocol => {
                let file = parse_private_key_file(&files.private).unwrap();
                assert_eq!(agent.client().add(&file.keys[0]), true);
            }
        }
    }
}

mod tests {
    use super::*;
    use crate::common::rfc8032::sign;
    use crate::common::ssh_format::{SshSig, armor_sshsig, public_blob, sshsig_signed_data};
    use crate::encrypted::{PASSPHRASE, add_passphrase};
    use rstest::rstest;
    use similar_asserts::assert_eq;
    use std::fs::{create_dir, read_to_string, write};
    use tempfile::TempDir;

    /// どちらの入れ方でも、ssh-agent の一覧は `.pub` と同じで（公開鍵 blob とコメント）、
    /// `ssh-add -T` で ssh-agent がこの鍵で署名できることも確かめられる
    #[rstest]
    fn agent_lists_the_key_as_pub_file(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values(Load::SshAdd, Load::Protocol)] load: Load,
        dir: TempDir,
    ) {
        require!("ssh-agent");
        let files = KeyFiles::shared(mode);
        let agent = Agent::start();
        load.load(&agent, files, dir.path());

        assert_eq!(
            agent.client().identities(),
            vec![(public_blob(&files.public()), COMMENT.as_bytes().to_vec())]
        );
        let out = run(agent.command("ssh-add").arg("-L"));
        assert_eq!(stdout(&out), format!("{}\n", files.pub_line));
        let out = stdout(&run(agent.command("ssh-add").arg("-l")));
        assert_eq!(out.contains(&fingerprint(&files.pub_line)), true, "{out}");
        assert_eq!(out.contains("(ED25519)"), true, "{out}");

        let pub_dir = dir.path().join("pub");
        create_dir(&pub_dir).unwrap();
        let pub_path = pub_dir.join("id_ed25519.pub");
        write(&pub_path, format!("{}\n", files.pub_line)).unwrap();
        run(agent.command("ssh-add").arg("-T").arg(&pub_path));
    }

    /// ssh-agent の署名は RFC 8032 の参照実装と1ビットも違わない（ssh-agent に直接頼んでも、
    /// 秘密鍵ファイルの無いところで `ssh-keygen -Y sign` に `.pub` だけ渡して ssh-agent に署名させても）
    #[rstest]
    fn agent_signature_is_exactly_rfc_8032(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values(Load::SshAdd, Load::Protocol)] load: Load,
        #[values(b"".as_slice(), b"meikey agent test\n", &[0xa5; 1023])] message: &[u8],
        dir: TempDir,
    ) {
        require!("ssh-agent");
        let files = KeyFiles::shared(mode);
        let agent = Agent::start();
        load.load(&agent, files, dir.path());

        let blob = public_blob(&files.public());
        assert_eq!(
            agent.client().sign(&blob, message),
            Some(sign(&files.seed(), message))
        );

        let work = dir.path().join("sign");
        create_dir(&work).unwrap();
        let pub_path = work.join("id_ed25519.pub");
        write(&pub_path, format!("{}\n", files.pub_line)).unwrap();
        let msg = work.join("message");
        write(&msg, message).unwrap();
        run(agent
            .command("ssh-keygen")
            .args(["-Y", "sign", "-n", "file", "-f"])
            .arg(&pub_path)
            .arg(&msg));
        let data = sshsig_signed_data(b"file", b"", b"sha512", message);
        let want = armor_sshsig(&SshSig {
            version: 1,
            public_key: files.public(),
            namespace: b"file".to_vec(),
            reserved: Vec::new(),
            hash_algorithm: b"sha512".to_vec(),
            signature: sign(&files.seed(), &data),
        });
        assert_eq!(
            read_to_string(msg.with_added_extension("sig")).unwrap(),
            want
        );
    }

    /// 入っていない鍵（もう一方の方式で作った鍵）では署名せず、取り除いた鍵でももう署名しない
    #[rstest]
    fn agent_signs_only_with_loaded_keys(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values(Load::SshAdd, Load::Protocol)] load: Load,
        dir: TempDir,
    ) {
        require!("ssh-agent");
        let (files, other) = (KeyFiles::shared(mode), KeyFiles::shared(mode.other()));
        let agent = Agent::start();
        load.load(&agent, files, dir.path());
        let mut client = agent.client();

        let (blob, other_blob) = (public_blob(&files.public()), public_blob(&other.public()));
        assert_eq!(client.sign(&other_blob, b"m"), None);
        assert_eq!(client.sign(&blob, b"m").is_some(), true);
        assert_eq!(client.remove(&blob), true);
        assert_eq!(client.identities(), Vec::new());
        assert_eq!(client.sign(&blob, b"m"), None);
        assert_eq!(client.remove(&blob), false);
    }

    /// パスフレーズを付けた鍵も、ssh-add がパスフレーズを聞いて入れ、元の鍵として使える。
    /// 違うパスフレーズでは入らない
    #[rstest]
    fn encrypted_key_via_ssh_add(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values(
            "aes256-ctr",
            "aes256-gcm@openssh.com",
            "chacha20-poly1305@openssh.com"
        )]
        cipher_name: &str,
        #[values(PASSPHRASE, "wrong")] answer: &str,
        dir: TempDir,
    ) {
        require!("ssh-agent");
        let files = KeyFiles::shared(mode);
        let (key, _) = files.write_to(dir.path());
        add_passphrase(&key, PASSPHRASE, &["-Z", cipher_name, "-a", "2"]);
        let askpass_dir = dir.path().join("askpass");
        create_dir(&askpass_dir).unwrap();
        let askpass = write_askpass(&askpass_dir, answer);

        let agent = Agent::start();
        let out = with_askpass(agent.command("ssh-add").arg(&key), &askpass)
            .output()
            .unwrap();
        let mut client = agent.client();
        if answer == PASSPHRASE {
            assert_eq!(out.status.success(), true, "{}", stderr(&out));
            let blob = public_blob(&files.public());
            assert_eq!(
                client.identities(),
                vec![(blob.clone(), COMMENT.as_bytes().to_vec())]
            );
            assert_eq!(client.sign(&blob, b"m"), Some(sign(&files.seed(), b"m")));
        } else {
            assert_eq!(out.status.success(), false);
            assert_eq!(client.identities(), Vec::new());
        }
    }
}
