//! README の手順（`ssh-keygen -p`）でパスフレーズを付けた meikey の鍵が PROTOCOL.key どおりに暗号化され、
//! 独立実装・ssh-key・ssh-keygen のどれでも、そのパスフレーズでだけ元の鍵として開けるか
//!
//! パスフレーズを付けた鍵を ssh-agent に入れる・ログインに使うのは [`crate::agent`]・[`crate::sshd`] で確かめる。

use crate::common::ssh_format::{CIPHERS, decrypt_private_key_file, public_blob};
use crate::common::*;
use similar_asserts::assert_eq;
use ssh_key::PrivateKey;
use std::fs::read_to_string;
use std::path::Path;
use std::process::Command;

pub const PASSPHRASE: &str = "correct horse battery staple";

/// `ssh-keygen -p` で使える暗号方式（`ssh -Q cipher` に出るもの全部）の名前
pub fn cipher_names() -> impl Iterator<Item = &'static str> {
    CIPHERS.iter().map(|c| c.name)
}

/// 暗号化されていない秘密鍵 `key` に `ssh-keygen -p` でパスフレーズを付け、暗号化したファイルの中身を返す。
/// `args` は `-Z`（暗号方式）・`-a`（bcrypt の回数）など
pub fn add_passphrase(key: &Path, passphrase: &str, args: &[&str]) -> String {
    run(Command::new("ssh-keygen")
        .args(["-p", "-P", "", "-N", passphrase])
        .args(args)
        .arg("-f")
        .arg(key));
    read_to_string(key).unwrap()
}

/// 暗号化した秘密鍵 `key`（中身は `pem`）を3つの実装で開くと、どれでも元の鍵（`files`）になる
fn assert_opens_as(pem: &str, key: &Path, passphrase: &str, files: &KeyFiles) {
    // 独立実装: 仕様どおりで、中身は元の鍵（シード・公開鍵・コメント）
    let file = decrypt_private_key_file(pem, passphrase.as_bytes()).unwrap();
    assert_eq!(file.violations(), Vec::<String>::new());
    assert_eq!(file.public_keys, vec![public_blob(&files.public())]);
    let k = &file.keys[0];
    assert_eq!(k.k, files.seed());
    assert_eq!(k.enc_a, files.public());
    assert_eq!(k.comment, COMMENT.as_bytes());
    // 同じフィールドから作り直すと1バイトも違わない
    assert_eq!(file.encode(passphrase.as_bytes()), pem);

    // ssh-key
    let encrypted = PrivateKey::from_openssh(pem).unwrap();
    assert_eq!(encrypted.is_encrypted(), true);
    assert_eq!(
        encrypted.cipher().as_str().as_bytes(),
        file.ciphername.as_slice()
    );
    let decrypted = encrypted.decrypt(passphrase).unwrap();
    assert_eq!(decrypted.comment(), COMMENT);
    assert_eq!(
        decrypted.key_data().ed25519().unwrap().private.to_bytes(),
        files.seed()
    );

    // ssh-keygen: 秘密鍵から復元した公開鍵の行は .pub と同じ
    let out = run(Command::new("ssh-keygen")
        .args(["-y", "-P", passphrase, "-f"])
        .arg(key));
    assert_eq!(stdout(&out), format!("{}\n", files.pub_line));
}

/// 違うパスフレーズでは、3つの実装のどれでも開けない
fn assert_rejects(pem: &str, key: &Path, wrong: &str) {
    assert_eq!(
        decrypt_private_key_file(pem, wrong.as_bytes()).is_err(),
        true
    );
    let encrypted = PrivateKey::from_openssh(pem).unwrap();
    assert_eq!(encrypted.decrypt(wrong).is_err(), true);
    let out = Command::new("ssh-keygen")
        .args(["-y", "-P", wrong, "-f"])
        .arg(key)
        .output()
        .unwrap();
    assert_eq!(out.status.success(), false);
}

mod tests {
    use super::*;
    use crate::common::ssh_format::{cipher, parse_bcrypt_kdfoptions, parse_private_key_file};
    use meikey::VanityKey;
    use meikey::repository::key_file::private_openssh;
    use rstest::rstest;
    use similar_asserts::assert_eq;
    use tempfile::TempDir;

    /// どの暗号方式で暗号化しても、PROTOCOL.key どおり（bcrypt・salt・rounds・ブロック長・認証タグ）で、
    /// そのパスフレーズでだけ元の鍵として開ける
    #[rstest]
    fn encrypted_key_follows_spec(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values(
            "3des-cbc",
            "aes128-cbc",
            "aes192-cbc",
            "aes256-cbc",
            "aes128-ctr",
            "aes192-ctr",
            "aes256-ctr",
            "aes128-gcm@openssh.com",
            "aes256-gcm@openssh.com",
            "chacha20-poly1305@openssh.com"
        )]
        cipher_name: &str,
        dir: TempDir,
    ) {
        require!("ssh-keygen");
        assert_eq!(cipher_names().any(|c| c == cipher_name), true);
        let files = KeyFiles::shared(mode);
        let (key, _) = files.write_to(dir.path());
        let pem = add_passphrase(&key, PASSPHRASE, &["-Z", cipher_name, "-a", "2"]);

        let file = decrypt_private_key_file(&pem, PASSPHRASE.as_bytes()).unwrap();
        assert_eq!(file.ciphername, cipher_name.as_bytes());
        assert_eq!(file.kdfname, b"bcrypt");
        let (salt, rounds) = parse_bcrypt_kdfoptions(&file.kdfoptions).unwrap();
        assert_eq!(
            (salt.len(), rounds),
            (16, 2),
            "OpenSSH の salt は 16 バイト"
        );
        let c = cipher(cipher_name.as_bytes()).unwrap();
        assert_eq!(file.private_list_len % c.block_size, 0);
        assert_eq!(file.tag.len(), c.auth_len);
        assert_eq!(
            parse_private_key_file(&pem).is_err(),
            true,
            "暗号化されている"
        );

        assert_opens_as(&pem, &key, PASSPHRASE, files);
        assert_rejects(&pem, &key, "wrong");
        assert_rejects(&pem, &key, "");
    }

    /// README の手順そのもの（暗号方式・bcrypt の回数は OpenSSH の既定）
    #[rstest]
    fn readme_procedure(#[values(Mode::Fast, Mode::ConstantTime)] mode: Mode, dir: TempDir) {
        require!("ssh-keygen");
        let files = KeyFiles::shared(mode);
        let (key, _) = files.write_to(dir.path());
        let pem = add_passphrase(&key, PASSPHRASE, &[]);
        assert_opens_as(&pem, &key, PASSPHRASE, files);
        assert_rejects(&pem, &key, "wrong");
    }

    /// どんなパスフレーズでも、そのパスフレーズでだけ開ける（1文字違い・末尾の空白違いでは開けない）
    #[rstest]
    fn any_passphrase_opens_only_itself(
        #[values(
            "x",
            PASSPHRASE,
            r#"p@ss/w0rd+= "'\"#,
            "パスフレーズ 🔑",
            " leading and trailing ",
            "a".repeat(200).leak()
        )]
        passphrase: &str,
        dir: TempDir,
    ) {
        require!("ssh-keygen");
        let files = KeyFiles::shared(Mode::Fast);
        let (key, _) = files.write_to(dir.path());
        let pem = add_passphrase(&key, passphrase, &["-a", "2"]);
        assert_opens_as(&pem, &key, passphrase, files);

        let mut one_char_off = passphrase.to_owned();
        let last = one_char_off.pop().unwrap();
        one_char_off.push(if last == 'x' { 'y' } else { 'x' });
        assert_rejects(&pem, &key, &one_char_off);
        assert_rejects(&pem, &key, &format!("{passphrase} "));
    }

    /// bcrypt の回数（`-a`）は kdfoptions の rounds に入り、どの回数でも開ける
    #[rstest]
    fn bcrypt_rounds(#[values(1, 16, 64)] rounds: u32, dir: TempDir) {
        require!("ssh-keygen");
        let files = KeyFiles::shared(Mode::Fast);
        let (key, _) = files.write_to(dir.path());
        let pem = add_passphrase(&key, PASSPHRASE, &["-a", &rounds.to_string()]);
        let file = decrypt_private_key_file(&pem, PASSPHRASE.as_bytes()).unwrap();
        assert_eq!(parse_bcrypt_kdfoptions(&file.kdfoptions).unwrap().1, rounds);
        assert_opens_as(&pem, &key, PASSPHRASE, files);
    }

    /// パスフレーズを変えると、新しいパスフレーズでだけ開ける
    #[rstest]
    fn change_passphrase(#[values(Mode::Fast, Mode::ConstantTime)] mode: Mode, dir: TempDir) {
        require!("ssh-keygen");
        let files = KeyFiles::shared(mode);
        let (key, _) = files.write_to(dir.path());
        add_passphrase(&key, "old", &["-a", "2"]);
        run(Command::new("ssh-keygen")
            .args(["-p", "-a", "2", "-P", "old", "-N", "new", "-f"])
            .arg(&key));
        let pem = read_to_string(&key).unwrap();
        assert_opens_as(&pem, &key, "new", files);
        assert_rejects(&pem, &key, "old");
    }

    /// パスフレーズを外すと、meikey が書くのと1バイトも違わないファイルに戻る
    #[rstest]
    fn removing_passphrase_restores_meikey_format(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values(
            "aes256-ctr",
            "aes256-cbc",
            "aes256-gcm@openssh.com",
            "chacha20-poly1305@openssh.com"
        )]
        cipher_name: &str,
        dir: TempDir,
    ) {
        require!("ssh-keygen");
        let files = KeyFiles::shared(mode);
        let (key, _) = files.write_to(dir.path());
        add_passphrase(&key, PASSPHRASE, &["-Z", cipher_name, "-a", "2"]);
        run(Command::new("ssh-keygen")
            .args(["-p", "-P", PASSPHRASE, "-N", "", "-f"])
            .arg(&key));
        let pem = read_to_string(&key).unwrap();
        let checkint = parse_private_key_file(&pem).unwrap().checkint1;
        let vanity = VanityKey::from_seed(files.seed());
        assert_eq!(private_openssh(&vanity, COMMENT, checkint), pem);
    }
}
