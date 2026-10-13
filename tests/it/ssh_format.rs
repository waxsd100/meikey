//! 独立実装（tests/support/ssh_format.rs）そのものが仕様どおりか
//!
//! ほかのテストはこの独立実装を正解として使うので、甘い（壊れたものを通す）と meikey 本体の誤りを見逃す。
//! ここでは meikey を使わずに、次の3つを確かめる。
//!
//! 1. 仕様書に載っている例（RFC 4648 Section 10、sshd(8) の authorized_keys の例）
//! 2. 仕様に違反するように壊したものを、違反として見つける・読まずに `Err` にする
//! 3. OpenSSH の `ssh-keygen` が作った鍵（暗号化なし・全暗号方式）を違反なしで読み、
//!    同じフィールドから1バイトも違わないファイルを作り直せる

use crate::common::rfc8032::secret_to_public;
use crate::common::ssh_format::*;

/// テストで使う Ed25519 の鍵（シードから RFC 8032 の参照実装で公開鍵を計算する）
pub fn key(seed: u8, comment: &str) -> Ed25519PrivateKey {
    let k = [seed; 32];
    let enc_a = secret_to_public(&k);
    Ed25519PrivateKey {
        enc_a,
        k,
        k_enc_a_tail: enc_a,
        comment: comment.into(),
    }
}

/// `seed` の鍵の `.pub` の1行
pub fn pub_line(seed: u8, comment: &str) -> String {
    let blob = public_blob(&secret_to_public(&[seed; 32]));
    format!("ssh-ed25519 {} {comment}", b64_encode(&blob))
}

/// 復号した PEM の中身を `f` で書き換え、PEM に戻す（base64 の折り返しは 70 文字）
fn edit_raw(pem: &str, f: fn(&mut Vec<u8>)) -> String {
    let body = pem.strip_circumfix(PEM_BEGIN, PEM_END).unwrap();
    let mut data = b64_decode(&body.split_terminator('\n').collect::<String>()).unwrap();
    f(&mut data);
    let b64 = b64_encode(&data);
    let lines: Vec<&str> = b64
        .as_bytes()
        .chunks(PEM_LINE_WIDTH)
        .map(|l| std::str::from_utf8(l).unwrap())
        .collect();
    format!("{PEM_BEGIN}{}\n{PEM_END}", lines.join("\n"))
}

mod tests {
    use super::*;
    use crate::common::*;
    use rstest::rstest;
    use similar_asserts::assert_eq;
    use std::fs::read_to_string;
    use std::process::Command;
    use tempfile::TempDir;

    // ---------------------------------------------------------------------------
    // RFC 4648 / RFC 4251 / RFC 8709
    // ---------------------------------------------------------------------------

    /// RFC 4648 Section 10 の例
    #[rstest]
    #[case::empty("", "")]
    #[case::f("f", "Zg==")]
    #[case::fo("fo", "Zm8=")]
    #[case::foo("foo", "Zm9v")]
    #[case::foob("foob", "Zm9vYg==")]
    #[case::fooba("fooba", "Zm9vYmE=")]
    #[case::foobar("foobar", "Zm9vYmFy")]
    fn base64_rfc_4648_section_10(#[case] data: &str, #[case] encoded: &str) {
        assert_eq!(b64_encode(data.as_bytes()), encoded);
        assert_eq!(b64_decode(encoded), Ok(data.as_bytes().to_vec()));
    }

    /// RFC 4648 の base64 として正しくないもの（Section 3.5 の canonical でないものも）は読まない
    #[rstest]
    #[case::length_not_multiple_of_4("Zg=")]
    #[case::no_padding("Zg")]
    #[case::three_pads("Z===")]
    #[case::pad_in_the_middle("Zg==Zg==")]
    #[case::non_canonical_one_pad("Zm9=")]
    #[case::non_canonical_two_pads("Zh==")]
    #[case::url_safe_alphabet("Zm-v")]
    #[case::trailing_newline("Zm9v\n")]
    #[case::space("Zm 9")]
    fn base64_rejects(#[case] encoded: &str) {
        assert_eq!(b64_decode(encoded).is_err(), true, "{encoded:?}");
    }

    /// RFC 4251 Section 5: 長さが足りない uint32・string は読まない
    #[rstest]
    #[case::short_uint32(&[0, 0, 1])]
    #[case::string_longer_than_data(&[0, 0, 0, 5, b'a', b'b'])]
    #[case::huge_length(&[0xff, 0xff, 0xff, 0xff])]
    fn reader_rejects_truncated(#[case] data: &[u8]) {
        assert_eq!(Reader::new(data).string().is_err(), true);
    }

    /// RFC 8709 Section 4 の公開鍵 blob でないものは読まない
    #[rstest]
    #[case::other_type(|b: &mut Vec<u8>| b[4..15].copy_from_slice(b"ssh-ed25518"))]
    #[case::key_31_bytes(|b: &mut Vec<u8>| { b[18] = 31; b.pop(); })]
    #[case::key_33_bytes(|b: &mut Vec<u8>| { b[18] = 33; b.push(0); })]
    #[case::trailing_byte(|b: &mut Vec<u8>| b.push(0))]
    #[case::truncated(|b: &mut Vec<u8>| { b.pop(); })]
    fn public_blob_rejects(#[case] edit: fn(&mut Vec<u8>)) {
        let mut blob = public_blob(&secret_to_public(&[1; 32]));
        assert_eq!(parse_public_blob(&blob), Ok(secret_to_public(&[1; 32])));
        edit(&mut blob);
        assert_eq!(parse_public_blob(&blob).is_err(), true);
    }

    /// `.pub` の1行として正しくないものは読まない
    #[rstest]
    #[case::no_base64("ssh-ed25519")]
    #[case::line_type_differs_from_blob(&pub_line(1, "c").replacen("ssh-ed25519", "ssh-rsa", 1))]
    #[case::two_spaces(&pub_line(1, "c").replacen(' ', "  ", 1))]
    #[case::newline(&format!("{}\n", pub_line(1, "c")))]
    #[case::broken_base64(&pub_line(1, "c").replacen("AAAA", "AA-A", 1))]
    fn pub_line_rejects(#[case] line: &str) {
        assert_eq!(parse_authorized_key(line).is_err(), true, "{line:?}");
    }

    // ---------------------------------------------------------------------------
    // sshd(8) AUTHORIZED_KEYS FILE FORMAT
    // ---------------------------------------------------------------------------

    /// sshd(8) の「An example authorized_keys file」のオプション（鍵の部分は Ed25519 の鍵に置き換える）。
    /// コメント行・空行は飛ばし、オプションは名前と値（クォートを外したもの）に分かれる
    #[rstest]
    fn authorized_keys_sshd_8_example() {
        let key = pub_line(1, "jane@example.net");
        let lines = [
            "# Comments are allowed at start of line. Blank lines are allowed.",
            "# Plain key, no restrictions",
            &key,
            "# Forced command, disable PTY and all forwarding",
            &format!("restrict,command=\"dump /home\" {key}"),
            "",
            "# Restriction of ssh -L forwarding destinations",
            &format!("permitopen=\"192.0.2.1:80\",permitopen=\"192.0.2.2:25\" {key}"),
            "# Restriction of ssh -R forwarding listeners",
            &format!("permitlisten=\"localhost:8080\",permitlisten=\"[::1]:22000\" {key}"),
            "# Configuration for tunnel forwarding",
            &format!("tunnel=\"0\",command=\"sh /etc/netstart tun0\" {key}"),
            "# Override of restriction to allow PTY allocation",
            &format!("restrict,pty,command=\"nethack\" {key}"),
            "# Allow FIDO key without requiring touch",
            &format!("no-touch-required {key}"),
            "# Require user-verification (e.g. PIN or biometric) for FIDO key",
            &format!("verify-required {key}"),
            "# Trust CA key, allow touch-less FIDO if requested in certificate",
            &format!("cert-authority,no-touch-required,principals=\"user_a\" {key}"),
        ];
        let parsed = parse_authorized_keys(&(lines.join("\n") + "\n")).unwrap();

        let (f, v) = (KeyOption::flag, KeyOption::value);
        let want: [(usize, Vec<KeyOption>); 9] = [
            (3, vec![]),
            (5, vec![f("restrict"), v("command", "dump /home")]),
            (
                8,
                vec![
                    v("permitopen", "192.0.2.1:80"),
                    v("permitopen", "192.0.2.2:25"),
                ],
            ),
            (
                10,
                vec![
                    v("permitlisten", "localhost:8080"),
                    v("permitlisten", "[::1]:22000"),
                ],
            ),
            (
                12,
                vec![v("tunnel", "0"), v("command", "sh /etc/netstart tun0")],
            ),
            (14, vec![f("restrict"), f("pty"), v("command", "nethack")]),
            (16, vec![f("no-touch-required")]),
            (18, vec![f("verify-required")]),
            (
                20,
                vec![
                    f("cert-authority"),
                    f("no-touch-required"),
                    v("principals", "user_a"),
                ],
            ),
        ];
        assert_eq!(parsed.len(), want.len());
        for (line, (number, options)) in parsed.iter().zip(want) {
            assert_eq!(line.line_number, number);
            assert_eq!(line.options, options);
            assert_eq!(line.key_type, "ssh-ed25519");
            assert_eq!(line.ed25519(), Ok(secret_to_public(&[1; 32])));
            assert_eq!(line.comment.as_deref(), Some("jane@example.net"));
            // 書き戻すと元の行のオプションの欄になる
            let text = &lines[number - 1];
            let formatted = format_key_options(&line.options);
            assert_eq!(
                text.strip_suffix(key.as_str()).unwrap().trim_end(),
                formatted
            );
        }
    }

    /// sshd(8): クォートの中は空白・カンマを含められ、`"` は `\"` と書く
    #[rstest]
    #[case::escaped_quote(r#"command="echo \"hi\"""#, "echo \"hi\"")]
    #[case::comma_and_space(r#"command="a,b c""#, "a,b c")]
    #[case::other_backslash(r#"command="a\b""#, r"a\b")]
    #[case::empty_value(r#"command="""#, "")]
    fn authorized_keys_quoted_value(#[case] options: &str, #[case] value: &str) {
        let line = format!("{options} {}", pub_line(1, "c"));
        let parsed = parse_authorized_keys(&line).unwrap();
        assert_eq!(parsed[0].options, vec![KeyOption::value("command", value)]);
        assert_eq!(format_key_options(&parsed[0].options), options);
    }

    /// authorized_keys として正しくない行は読まない（`with_key` なら後ろに正しい鍵を付ける）
    #[rstest]
    #[case::unterminated_quote(r#"command="echo hi "#, true)]
    #[case::unquoted_value("command=echo", true)]
    #[case::space_in_options("restrict, pty", true)]
    #[case::empty_option("restrict,,pty", true)]
    #[case::options_without_key("restrict", false)]
    #[case::unknown_key_type_after_options("restrict ssh-foo AAAA", false)]
    fn authorized_keys_rejects(#[case] prefix: &str, #[case] with_key: bool) {
        let text = match with_key {
            true => format!("{prefix} {}", pub_line(1, "c")),
            false => prefix.to_owned(),
        };
        assert_eq!(parse_authorized_keys(&text).is_err(), true, "{text:?}");
    }

    /// 行の種類と blob の中の種類が違う行は読まない（オプションがあってもなくても）
    #[rstest]
    #[case::without_options("")]
    #[case::with_options("restrict ")]
    fn authorized_keys_rejects_type_mismatch(#[case] prefix: &str) {
        let line = pub_line(1, "c").replacen("ssh-ed25519", "ssh-rsa", 1);
        assert_eq!(
            parse_authorized_keys(&format!("{prefix}{line}")).is_err(),
            true
        );
    }

    // ---------------------------------------------------------------------------
    // PROTOCOL.key: 壊したファイル
    // ---------------------------------------------------------------------------

    /// 正しいファイル（コメント 1 バイトで、秘密鍵リストは 132 バイト + パディング 4 バイト）
    fn valid() -> PrivateKeyFile {
        let file = PrivateKeyFile::new(CIPHER_NONE, Vec::new(), key(1, "v"), 0x0102_0304);
        assert_eq!(file.padding, vec![1, 2, 3, 4]);
        file
    }

    /// 正しいファイルは違反なしで読め、書いたフィールドがそのまま戻る
    #[rstest]
    fn valid_file_round_trips() {
        let file = valid();
        let parsed = parse_private_key_file(&file.encode(b"")).unwrap();
        assert_eq!(parsed.violations(), Vec::<String>::new());
        assert_eq!(parsed.keys, file.keys);
        assert_eq!(
            (parsed.checkint1, parsed.checkint2),
            (0x0102_0304, 0x0102_0304)
        );
        assert_eq!(parsed.padding, file.padding);
        assert_eq!(parsed.encode(b""), file.encode(b""));
    }

    /// フィールドを1つ壊すと、その違反だけが見つかる
    #[rstest]
    #[case::checkint(|f: &mut PrivateKeyFile| f.checkint2 ^= 1, "checkint が一致しない")]
    #[case::padding_not_sequential(|f: &mut PrivateKeyFile| f.padding[0] = 0, "パディングが 1, 2, 3, ... でない")]
    #[case::padding_too_long(|f: &mut PrivateKeyFile| f.padding = (1..=12).collect(), "パディングが最小でない")]
    #[case::padding_too_short(|f: &mut PrivateKeyFile| { f.padding.pop(); }, "ブロック長の倍数でない")]
    #[case::k_enc_a_tail(|f: &mut PrivateKeyFile| f.keys[0].k_enc_a_tail[0] ^= 1, "k || ENC(A) の後半が ENC(A) でない")]
    #[case::header_public_key(|f: &mut PrivateKeyFile| f.public_keys[0] = public_blob(&[9; 32]), "ヘッダの公開鍵と ENC(A) が違う")]
    #[case::kdfname(|f: &mut PrivateKeyFile| f.kdfname = b"bcrypt".into(), "kdfname が none でない")]
    #[case::kdfoptions(|f: &mut PrivateKeyFile| f.kdfoptions = bcrypt_kdfoptions(&[0; 16], 16), "kdfoptions が空でない")]
    // 2つ目の鍵（コメント 5 バイト）は 128 バイトなので、ブロック長の倍数は崩れない
    #[case::two_keys(|f: &mut PrivateKeyFile| { f.keys.push(key(2, "other")); f.public_keys.push(public_blob(&key(2, "").enc_a)); }, "鍵が1つでない")]
    fn broken_field_is_a_violation(#[case] edit: fn(&mut PrivateKeyFile), #[case] want: &str) {
        let mut file = valid();
        edit(&mut file);
        let parsed = parse_private_key_file(&file.encode(b"")).unwrap();
        assert_eq!(parsed.violations(), vec![want.to_owned()]);
    }

    /// base64 の折り返しが 70 文字でない
    #[rstest]
    #[case::width_64(64, "最終行以外が 70 文字でない")]
    #[case::width_76(76, "最終行以外が 70 文字でない")]
    fn wrong_line_width_is_a_violation(#[case] width: usize, #[case] want: &str) {
        let pem = valid().encode(b"");
        let b64: String = pem
            .strip_circumfix(PEM_BEGIN, PEM_END)
            .unwrap()
            .replace('\n', "");
        let lines: Vec<&str> = b64
            .as_bytes()
            .chunks(width)
            .map(|l| std::str::from_utf8(l).unwrap())
            .collect();
        let pem = format!("{PEM_BEGIN}{}\n{PEM_END}", lines.join("\n"));
        assert_eq!(
            parse_private_key_file(&pem).unwrap().violations(),
            vec![want.to_owned()]
        );
    }

    /// 構造が壊れたファイルは読まない
    #[rstest]
    #[case::auth_magic(|d: &mut Vec<u8>| d[13] = b'2')]
    #[case::trailing_data(|d: &mut Vec<u8>| d.push(0))]
    #[case::truncated(|d: &mut Vec<u8>| { d.pop(); })]
    #[case::unknown_cipher(|d: &mut Vec<u8>| d[19..23].copy_from_slice(b"nope"))]
    fn broken_structure_is_rejected(#[case] edit: fn(&mut Vec<u8>)) {
        let pem = edit_raw(&valid().encode(b""), edit);
        assert_eq!(parse_private_key_file(&pem).is_err(), true);
    }

    /// BEGIN / END の行が無い・違う
    #[rstest]
    #[case::no_end(|p: &str| p.replace(PEM_END, ""))]
    #[case::no_final_newline(|p: &str| p.trim_end().to_owned())]
    #[case::other_label(|p: &str| p.replace("OPENSSH PRIVATE KEY", "PRIVATE KEY"))]
    fn broken_armor_is_rejected(#[case] edit: fn(&str) -> String) {
        let pem = edit(&valid().encode(b""));
        assert_eq!(parse_private_key_file(&pem).is_err(), true);
    }

    /// 暗号化されたファイルは、暗号化されていないファイルとしては読まない
    #[rstest]
    fn encrypted_file_is_not_read_as_unencrypted() {
        let cipher = cipher(b"aes256-ctr").unwrap();
        let file = PrivateKeyFile::new(cipher, bcrypt_kdfoptions(&[7; 16], 1), key(1, "c"), 5);
        let pem = file.encode(b"pass");
        assert_eq!(parse_private_key_file(&pem).is_err(), true);
        assert_eq!(
            decrypt_private_key_file(&pem, b"pass").unwrap().keys,
            file.keys
        );
    }

    // ---------------------------------------------------------------------------
    // PROTOCOL.key: OpenSSH が作ったファイル
    // ---------------------------------------------------------------------------

    /// `ssh-keygen -t ed25519` で作った鍵は違反なしで読め、中身は RFC 8032 どおりで、
    /// 同じフィールド（同じ salt・rounds・checkint）から1バイトも違わないファイルを作り直せる。
    /// パスフレーズが違えば読まない。コメントの長さを変えて、パディングをすべて通す
    #[rstest]
    fn ssh_keygen_keys_follow_spec(
        #[values(
            "none",
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
        let cipher = cipher(cipher_name.as_bytes()).unwrap();
        let passphrase = if cipher == CIPHER_NONE {
            ""
        } else {
            "パスフレーズ pass"
        };
        for len in 0..cipher.block_size {
            let comment = "c".repeat(len);
            let name = format!("k{len}");
            let files = ssh_keygen_ed25519(
                dir.path(),
                &name,
                &comment,
                passphrase,
                &["-Z", cipher_name, "-a", "2"],
            );

            let file = decrypt_private_key_file(&files.private, passphrase.as_bytes()).unwrap();
            assert_eq!(
                file.violations(),
                Vec::<String>::new(),
                "コメント {len} バイト"
            );
            assert_eq!(file.ciphername, cipher_name.as_bytes());
            assert_eq!(file.tag.len(), cipher.auth_len);
            if cipher != CIPHER_NONE {
                let (salt, rounds) = parse_bcrypt_kdfoptions(&file.kdfoptions).unwrap();
                assert_eq!((salt.len(), rounds), (16, 2));
            }
            let key = &file.keys[0];
            assert_eq!(
                secret_to_public(&key.k),
                key.enc_a,
                "ENC(A) はシードから RFC 8032 で計算した公開鍵"
            );
            assert_eq!(key.comment, comment.as_bytes());
            let line = parse_authorized_key(&files.pub_line).unwrap();
            assert_eq!(line.public_key, key.enc_a);
            assert_eq!(line.comment.unwrap_or_default(), comment);

            assert_eq!(
                file.encode(passphrase.as_bytes()),
                files.private,
                "作り直したファイル"
            );
            if cipher != CIPHER_NONE {
                assert_eq!(
                    decrypt_private_key_file(&files.private, b"wrong").is_err(),
                    true
                );
                assert_eq!(parse_private_key_file(&files.private).is_err(), true);
            }
        }
    }

    /// 独立実装で暗号化したファイル（ssh-keygen が作ったものではない）を ssh-keygen が開ける
    #[rstest]
    fn ssh_keygen_opens_files_encrypted_by_spec(
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
        let cipher = cipher(cipher_name.as_bytes()).unwrap();
        let file = PrivateKeyFile::new(
            cipher,
            bcrypt_kdfoptions(&[0xa5; 16], 3),
            key(4, "spec"),
            0xdead_beef,
        );
        let files = KeyFiles {
            private: file.encode(b"pw"),
            pub_line: pub_line(4, "spec"),
        };
        let (path, _) = files.write_to(dir.path());
        let out = run(Command::new("ssh-keygen")
            .args(["-y", "-P", "pw", "-f"])
            .arg(&path));
        assert_eq!(stdout(&out), format!("{}\n", files.pub_line));
        let wrong = Command::new("ssh-keygen")
            .args(["-y", "-P", "wrong", "-f"])
            .arg(&path)
            .output()
            .unwrap();
        assert_eq!(wrong.status.success(), false);
        assert_eq!(read_to_string(&path).unwrap(), files.private);
    }
}
