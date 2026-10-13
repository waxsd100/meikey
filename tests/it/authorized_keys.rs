//! meikey の `.pub` を authorized_keys（sshd(8) AUTHORIZED_KEYS FILE FORMAT）に書いて使えるか
//!
//! コメント行・空行・ほかの種類の鍵の行・ほかの meikey の鍵の行と並べ、行頭にオプション（[`Options`]）を
//! 付けても、独立実装・ssh-key・`ssh-keygen -l` が同じ鍵として読み、sshd がオプションどおりに扱う
//! （sshd は Unix で Docker が使えるときだけ）。`.pub` をそのまま連結したファイルも authorized_keys になる。

use crate::common::*;
use std::path::Path;

/// meikey の鍵の行の前に付けるオプション（sshd(8) のオプションの欄）
#[derive(Debug, Clone, Copy)]
pub enum Options {
    /// オプションなし（`.pub` の1行そのまま）
    None,
    Restrict,
    /// 機能を1つずつ止める
    NoEachFeature,
    /// 決まったコマンドを実行させる
    Command,
    /// クォートの中の `\"`
    EscapedQuote,
    /// 元のコマンドは `SSH_ORIGINAL_COMMAND` に入る
    OriginalCommand,
    /// restrict のあとで pty だけ許す（sshd(8) の例と同じ並び）
    RestrictPtyCommand,
    /// オプションの名前は大文字・小文字を区別しない
    Uppercase,
    /// どこからでも
    FromAny,
    /// 関係ないアドレスの否定と、どこからでも
    FromNegatedOther,
    /// ほかのアドレスからだけ（＝ テストの接続元からは入れない）
    FromOtherHost,
    ExpiryFuture,
    ExpiryPast,
}

impl Options {
    /// オプションの欄の文字列（`None` は空）
    pub fn text(self) -> &'static str {
        match self {
            Options::None => "",
            Options::Restrict => "restrict",
            Options::NoEachFeature => {
                "no-agent-forwarding,no-port-forwarding,no-pty,no-user-rc,no-X11-forwarding"
            }
            Options::Command => r#"command="echo forced""#,
            Options::EscapedQuote => r#"command="echo \"forced  quote\"""#,
            Options::OriginalCommand => r#"command="echo original: $SSH_ORIGINAL_COMMAND""#,
            Options::RestrictPtyCommand => r#"restrict,pty,command="echo multi""#,
            Options::Uppercase => r#"RESTRICT,COMMAND="echo upper""#,
            Options::FromAny => r#"from="*""#,
            Options::FromNegatedOther => r#"from="!192.0.2.1,*""#,
            Options::FromOtherHost => r#"from="192.0.2.1""#,
            Options::ExpiryFuture => r#"expiry-time="29991231""#,
            Options::ExpiryPast => r#"expiry-time="20000101""#,
        }
    }

    /// sshd で `echo meikey-login-ok` を実行したときの出力（None ならログインできない）。
    /// sshd のテストは Unix だけ
    #[cfg(unix)]
    pub fn outcome(self) -> Option<&'static str> {
        match self {
            Options::None
            | Options::Restrict
            | Options::NoEachFeature
            | Options::FromAny
            | Options::FromNegatedOther
            | Options::ExpiryFuture => Some("meikey-login-ok"),
            Options::Command => Some("forced"),
            Options::EscapedQuote => Some("forced  quote"),
            Options::OriginalCommand => Some("original: echo meikey-login-ok"),
            Options::RestrictPtyCommand => Some("multi"),
            Options::Uppercase => Some("upper"),
            Options::FromOtherHost | Options::ExpiryPast => None,
        }
    }

    /// オプションを付けた `pub_line` の行
    pub fn line(self, pub_line: &str) -> String {
        match self {
            Options::None => pub_line.to_owned(),
            _ => format!("{} {pub_line}", self.text()),
        }
    }
}

/// authorized_keys の中身: コメント行・空行・ほかの種類の鍵（ECDSA）の行・ほかの meikey の鍵の行のあとに、
/// オプションを付けた `pub_line` を置く。ほかの鍵の行は ssh-keygen で `dir` に作る
pub fn authorized_keys(
    dir: &Path,
    pub_line: &str,
    other_pub_line: &str,
    options: Options,
) -> String {
    let ecdsa = dir.join("ecdsa");
    run(std::process::Command::new("ssh-keygen")
        .args([
            "-q",
            "-t",
            "ecdsa",
            "-N",
            "",
            "-C",
            "someone@example.net",
            "-f",
        ])
        .arg(&ecdsa));
    let ecdsa_line = std::fs::read_to_string(ecdsa.with_added_extension("pub")).unwrap();
    [
        "# meikey test",
        "",
        ecdsa_line.trim_end(),
        "# 別の meikey の鍵",
        other_pub_line,
        &options.line(pub_line),
        "",
    ]
    .join("\n")
}

mod tests {
    use super::*;
    use crate::common::ssh_format::{format_key_options, parse_authorized_keys};
    use rstest::rstest;
    use similar_asserts::assert_eq;
    use ssh_key::{AuthorizedKeys, PublicKey};
    use std::fs::write;
    use tempfile::TempDir;

    /// ほかの行と並べてオプションを付けても、独立実装・ssh-key・`ssh-keygen -l` が同じ鍵として読む
    #[rstest]
    fn readers_find_the_key_among_other_lines(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values(
            Options::None,
            Options::Restrict,
            Options::NoEachFeature,
            Options::Command,
            Options::EscapedQuote,
            Options::OriginalCommand,
            Options::RestrictPtyCommand,
            Options::Uppercase,
            Options::FromAny,
            Options::FromNegatedOther,
            Options::FromOtherHost,
            Options::ExpiryFuture,
            Options::ExpiryPast
        )]
        options: Options,
        dir: TempDir,
    ) {
        require!("ssh-keygen");
        let (files, other) = (KeyFiles::shared(mode), KeyFiles::shared(mode.other()));
        let text = authorized_keys(dir.path(), &files.pub_line, &other.pub_line, options);

        // 独立実装: コメント行・空行を飛ばして3つの鍵。最後の行がオプション付きの meikey の鍵
        let lines = parse_authorized_keys(&text).unwrap();
        assert_eq!(
            lines.iter().map(|l| l.line_number).collect::<Vec<_>>(),
            vec![3, 5, 6]
        );
        assert_eq!(lines[1].ed25519(), Ok(other.public()));
        let ours = &lines[2];
        assert_eq!(ours.key_type, "ssh-ed25519");
        assert_eq!(ours.ed25519(), Ok(files.public()));
        assert_eq!(ours.comment.as_deref(), Some(COMMENT));
        assert_eq!(format_key_options(&ours.options), options.text());

        // ssh-key。ssh-key 0.6 はクォートの中の空白を受け付けない（sshd(8) は「ダブルクォートの中以外に
        // 空白は入らない」で、中の空白は使える）ので、空白を含むオプションでは使わない
        if !options.text().contains(' ') {
            let entries: Vec<_> = AuthorizedKeys::new(&text)
                .collect::<Result<_, _>>()
                .unwrap();
            assert_eq!(entries.len(), 3);
            let public = PublicKey::from_openssh(&files.pub_line).unwrap();
            assert_eq!(entries[2].public_key().key_data(), public.key_data());
            assert_eq!(entries[2].public_key().comment(), COMMENT);
            assert_eq!(entries[2].config_opts().as_str(), options.text());
        }

        // ssh-keygen -l: 鍵ごとに1行、ファイルの順に、フィンガープリントとコメント
        let path = dir.path().join("authorized_keys");
        write(&path, &text).unwrap();
        let out = stdout(&run(std::process::Command::new("ssh-keygen")
            .arg("-l")
            .arg("-f")
            .arg(&path)));
        let listed: Vec<Vec<&str>> = out.lines().map(|l| l.split(' ').collect()).collect();
        assert_eq!(listed.len(), 3, "{out}");
        assert_eq!(listed[1][1], fingerprint(&other.pub_line));
        assert_eq!(listed[2][1], fingerprint(&files.pub_line));
        assert_eq!(listed[2][2], COMMENT);
        assert_eq!(listed[2][3], "(ED25519)");
    }

    /// meikey の `.pub` ファイル（改行で終わる）をそのまま連結すると、authorized_keys として読める
    #[rstest]
    fn concatenated_pub_files_are_authorized_keys(dir: TempDir) {
        require!("ssh-keygen");
        let (a, b) = (
            KeyFiles::shared(Mode::Fast),
            KeyFiles::shared(Mode::ConstantTime),
        );
        let text = format!("{}\n{}\n", a.pub_line, b.pub_line);
        let lines = parse_authorized_keys(&text).unwrap();
        assert_eq!(
            lines
                .iter()
                .map(|l| l.ed25519().unwrap())
                .collect::<Vec<_>>(),
            vec![a.public(), b.public()]
        );
        assert_eq!(AuthorizedKeys::new(&text).count(), 2);
        let path = dir.path().join("authorized_keys");
        write(&path, &text).unwrap();
        let out = stdout(&run(std::process::Command::new("ssh-keygen")
            .arg("-l")
            .arg("-f")
            .arg(&path)));
        assert_eq!(out.lines().count(), 2, "{out}");
    }

    /// sshd がオプションどおりに扱う: 決まったコマンドを実行させる・接続元や期限で断る
    #[cfg(unix)]
    #[rstest]
    fn sshd_applies_options(
        #[values(Mode::Fast, Mode::ConstantTime)] mode: Mode,
        #[values(
            Options::None,
            Options::Restrict,
            Options::NoEachFeature,
            Options::Command,
            Options::EscapedQuote,
            Options::OriginalCommand,
            Options::RestrictPtyCommand,
            Options::Uppercase,
            Options::FromAny,
            Options::FromNegatedOther,
            Options::FromOtherHost,
            Options::ExpiryFuture,
            Options::ExpiryPast
        )]
        options: Options,
        dir: TempDir,
    ) {
        use crate::sshd::{self, COMMAND, Identity};
        require!("ssh");
        let (files, other) = (KeyFiles::shared(mode), KeyFiles::shared(mode.other()));
        let text = authorized_keys(dir.path(), &files.pub_line, &other.pub_line, options);
        let Some(server) = sshd::start_or_skip(&text) else {
            return;
        };
        let key_dir = dir.path().join("key");
        std::fs::create_dir(&key_dir).unwrap();
        let (key, _) = files.write_to(&key_dir);
        let out = sshd::ssh(&server, Identity::File(&key), dir.path(), COMMAND);
        match options.outcome() {
            Some(want) => sshd::assert_logged_in(&server, &out, &files.pub_line, want),
            None => sshd::assert_denied(&server, &out),
        }
    }

    /// `.pub` をそのまま連結した authorized_keys で、どちらの鍵でもログインできる
    #[cfg(unix)]
    #[rstest]
    fn sshd_accepts_concatenated_pub_files(dir: TempDir) {
        use crate::sshd::{self, COMMAND, COMMAND_OUTPUT, Identity};
        require!("ssh");
        let keys = [
            KeyFiles::shared(Mode::Fast),
            KeyFiles::shared(Mode::ConstantTime),
        ];
        let text = format!("{}\n{}\n", keys[0].pub_line, keys[1].pub_line);
        let Some(server) = sshd::start_or_skip(&text) else {
            return;
        };
        for (i, files) in keys.iter().enumerate() {
            let key_dir = dir.path().join(format!("key{i}"));
            std::fs::create_dir(&key_dir).unwrap();
            let (key, _) = files.write_to(&key_dir);
            let out = sshd::ssh(&server, Identity::File(&key), dir.path(), COMMAND);
            sshd::assert_logged_in(&server, &out, &files.pub_line, COMMAND_OUTPUT);
        }
    }
}
