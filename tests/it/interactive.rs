//! 引数なしで端末から起動したときの対話のテスト（[`tests`]）と、そのヘルパー（Unix だけ）
//!
//! 疑似端末で meikey を起動し、人がキーを押すように答えて、表示と書き出したファイルを確かめる。

use crate::common::{BIN, LOCALE_VARS};
use rexpect::process::WaitStatus;
use rexpect::session::{PtySession, spawn_command};
use std::path::Path;
use std::process::Command;

/// 1つの表示を待つ時間の上限
const TIMEOUT_MS: u64 = 30_000;
/// Enter・Esc のキー
const ENTER: &str = "\r";
const ESC: &str = "\x1b";

/// `dir` で、引数なしの meikey を疑似端末につないで起動する（日本語のロケールで）
fn spawn_in(dir: &Path) -> PtySession {
    spawn_with_locale(dir, "ja_JP.UTF-8")
}

/// `dir` で、引数なしの meikey を、ロケールを `locale` にして疑似端末につないで起動する
fn spawn_with_locale(dir: &Path, locale: &str) -> PtySession {
    let mut cmd = Command::new(BIN);
    for name in LOCALE_VARS {
        cmd.env_remove(name);
    }
    cmd.current_dir(dir)
        .env("TERM", "xterm-256color")
        .env("LC_ALL", locale);
    spawn_command(cmd, Some(TIMEOUT_MS)).expect("疑似端末で起動できる")
}

/// `prompt` が出るのを待って、`keys` を押す。rexpect は出力を1バイトずつ1文字として読むので、
/// 待つ文字列も UTF-8 のバイトを1文字ずつにしてから探す
fn answer(p: &mut PtySession, prompt: &str, keys: &str) {
    let bytes: String = prompt.bytes().map(char::from).collect();
    p.exp_string(&bytes)
        .unwrap_or_else(|e| panic!("{prompt} が出ない: {e}"));
    p.send(keys).unwrap();
    p.flush().unwrap();
}

/// 終わるのを待って、終了コード
fn exit_code(mut p: PtySession) -> i32 {
    p.exp_eof().unwrap();
    match p.process().wait().unwrap() {
        WaitStatus::Exited(_, code) => code,
        s => panic!("終了コードで終わっていない: {s:?}"),
    }
}

mod tests {
    use super::*;
    use crate::common::dir;
    use rstest::rstest;
    use similar_asserts::assert_eq;
    use std::fs::{read_dir, read_to_string};
    use tempfile::TempDir;

    /// 既定のまま答えていくと、引数で起動したときと同じく鍵を探して書き出す
    #[rstest]
    fn generates_key(dir: TempDir) {
        let mut p = spawn_in(dir.path());
        answer(&mut p, "何をする", ENTER);
        answer(&mut p, "探し方", ENTER);
        answer(&mut p, "公開鍵に入れたい文字列", &format!("12{ENTER}"));
        // 低セキュリティモードにしない（使うものは聞かれず CPU）
        answer(&mut p, "低セキュリティモードにする", ENTER);
        answer(&mut p, "秘密鍵の保存先", ENTER);
        answer(&mut p, "鍵のコメント", &format!("wako{ENTER}"));
        answer(&mut p, "引数で起動するなら: meikey 12 --comment=wako", "");
        answer(&mut p, "始める？", "y");
        answer(&mut p, "パスフレーズ", "");
        assert_eq!(exit_code(p), 0);

        let public = read_to_string(dir.path().join("id_ed25519_meikey.pub")).unwrap();
        let b64 = public.split(' ').nth(1).unwrap();
        assert_eq!(
            b64["AAAAC3NzaC1lZDI1NTE5AAAAI".len()..].contains("12"),
            true,
            "{public}"
        );
        assert_eq!(public.trim_end().ends_with(" wako"), true, "{public}");
        assert_eq!(dir.path().join("id_ed25519_meikey").exists(), true);
    }

    /// 探し方で「公開鍵の最後」（下矢印で 2 つ目）を選ぶと --suffix になり、最後がその文字列の鍵を書き出す
    #[rstest]
    fn suffix_from_menu(dir: TempDir) {
        let mut p = spawn_in(dir.path());
        answer(&mut p, "何をする", ENTER);
        answer(&mut p, "探し方", &format!("\x1b[B{ENTER}"));
        answer(&mut p, "公開鍵に入れたい文字列", &format!("7{ENTER}"));
        answer(&mut p, "低セキュリティモードにする", ENTER);
        answer(&mut p, "秘密鍵の保存先", ENTER);
        answer(&mut p, "鍵のコメント", ENTER);
        answer(&mut p, "引数で起動するなら: meikey 7 --suffix", "");
        answer(&mut p, "始める？", "y");
        answer(&mut p, "パスフレーズ", "");
        assert_eq!(exit_code(p), 0);
        let public = read_to_string(dir.path().join("id_ed25519_meikey.pub")).unwrap();
        let b64 = public.split(' ').nth(1).unwrap();
        assert_eq!(b64.trim_end().ends_with('7'), true, "{public}");
    }

    /// 最初の質問で Esc を押すと、何もせずに終わる
    #[rstest]
    fn esc_cancels(dir: TempDir) {
        let mut p = spawn_in(dir.path());
        answer(&mut p, "何をする", ESC);
        answer(&mut p, "やめた", "");
        assert_eq!(exit_code(p), 0);
        assert_eq!(read_dir(dir.path()).unwrap().count(), 0);
    }

    /// 探している途中で Ctrl+C を押すと、130 で終わり、何も書き出さない
    #[rstest]
    fn ctrl_c_during_search(dir: TempDir) {
        let mut p = spawn_in(dir.path());
        answer(&mut p, "何をする", ENTER);
        answer(&mut p, "探し方", ENTER);
        // 何年もかかる文字列
        answer(
            &mut p,
            "公開鍵に入れたい文字列",
            &format!("0123456789{ENTER}"),
        );
        // 低セキュリティモードにしない（使うものは聞かれず CPU）
        answer(&mut p, "低セキュリティモードにする", ENTER);
        answer(&mut p, "秘密鍵の保存先", ENTER);
        answer(&mut p, "鍵のコメント", ENTER);
        answer(&mut p, "始める？", "y");
        answer(&mut p, "期待試行", "");
        p.send_control('c').unwrap();
        assert_eq!(exit_code(p), 130);
        assert_eq!(read_dir(dir.path()).unwrap().count(), 0);
    }

    /// 日本語でないロケールなら、質問も表示も英語
    #[rstest]
    fn english_locale(dir: TempDir) {
        let mut p = spawn_with_locale(dir.path(), "en_US.UTF-8");
        answer(&mut p, "What to do", ENTER);
        answer(&mut p, "How to match", ENTER);
        answer(
            &mut p,
            "Text to put in the public key",
            &format!("1{ENTER}"),
        );
        answer(&mut p, "Use low-security mode", ENTER);
        answer(&mut p, "Where to save the private key", ENTER);
        answer(&mut p, "Comment for the key", ENTER);
        answer(&mut p, "The same with arguments: meikey 1", "");
        answer(&mut p, "Start?", "y");
        answer(&mut p, "Passphrase", "");
        assert_eq!(exit_code(p), 0);
        assert_eq!(dir.path().join("id_ed25519_meikey.pub").exists(), true);
    }
}
