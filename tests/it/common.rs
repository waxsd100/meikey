//! IT 共通のヘルパー

use meikey::VanityKey;
use meikey::repository::key_file::private_openssh;
use rstest::fixture;
use similar_asserts::assert_eq;
use std::env::var;
use std::fs::{
    OpenOptions, create_dir_all, read_dir, read_to_string, remove_dir_all, rename, write,
};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;
use std::time::Duration;
use tempfile::{TempDir, tempdir, tempdir_in};

/// proptest を `#[rstest]` から回す補助（src/ のユニットテストと共有）
#[path = "../support/property.rs"]
pub mod property;
/// RFC 8032 Section 6 の参照実装（src/ のユニットテストと共有）
#[path = "../support/rfc8032.rs"]
pub mod rfc8032;
/// SSH 関連の形式の独立実装（src/ のユニットテストと共有）
#[path = "../support/ssh_format.rs"]
pub mod ssh_format;

use ssh_format::{parse_authorized_key, parse_private_key_file, public_blob, sha256_fingerprint};

/// cargo がビルドした meikey 本体
pub const BIN: &str = env!("CARGO_BIN_EXE_meikey");

/// IT で探す文字列（大小文字を区別しないので約800回で見つかる）
pub const PATTERN: &str = "Mei";
pub const COMMENT: &str = "meikey-it";

/// テストごとの一時ディレクトリ（drop で消える）
#[fixture]
pub fn dir() -> TempDir {
    tempdir().unwrap()
}

/// meikey を `dir` で、日本語のロケールで実行する（`-o` を付けなければ `dir` に書き出す）
pub fn meikey_in(dir: &Path, args: &[&str]) -> Output {
    meikey_with_locale(dir, args, &[("LC_ALL", "ja_JP.UTF-8")])
}

/// meikey を `dir` で、ロケールの環境変数を `locale` だけにして実行する
/// （Unix では表示の言語がこれで決まる。Windows は OS の表示言語で決まり、環境変数は見ない）
pub fn meikey_with_locale(dir: &Path, args: &[&str], locale: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(BIN);
    cmd.args(args).current_dir(dir);
    for name in LOCALE_VARS {
        cmd.env_remove(name);
    }
    cmd.envs(locale.iter().copied());
    cmd.output().expect("meikey を起動できる")
}

/// 表示の言語を決めるのに見る環境変数（この順に見る）
pub const LOCALE_VARS: [&str; 4] = ["LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"];

/// 公開鍵の計算方法（meikey の引数で切り替える）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// 既定（高速化あり）
    Fast,
    /// `--constant-time`（倍算だけ高速化前）
    ConstantTime,
    /// `--constant-time --no-batch --simd off --no-memmem`（倍算も圧縮も SIMD も検索も高速化前）
    Baseline,
}

impl Mode {
    pub fn args(self) -> &'static [&'static str] {
        match self {
            Mode::Fast => &[],
            Mode::ConstantTime => &["--constant-time"],
            Mode::Baseline => &[
                "--constant-time",
                "--no-batch",
                "--simd",
                "off",
                "--no-memmem",
            ],
        }
    }

    /// もう一方の方式
    pub fn other(self) -> Mode {
        match self {
            Mode::Fast => Mode::ConstantTime,
            Mode::ConstantTime | Mode::Baseline => Mode::Fast,
        }
    }
}

/// 鍵ファイルの中身（秘密鍵と `.pub`）
#[derive(Debug)]
pub struct KeyFiles {
    /// 秘密鍵（OpenSSH 形式の PEM）
    pub private: String,
    /// `.pub` の1行（改行なし）
    pub pub_line: String,
}

impl KeyFiles {
    /// meikey で `PATTERN` 入りの鍵を作る（呼ぶたびに新しい鍵）
    pub fn generate(mode: Mode) -> Self {
        let dir = tempdir().unwrap();
        let mut args = vec![PATTERN, "-j", "2", "-C", COMMENT, "-o", "k"];
        args.extend(mode.args());
        let out = meikey_in(dir.path(), &args);
        assert_eq!(
            out.status.success(),
            true,
            "meikey が失敗: {}",
            stderr(&out)
        );

        let read = |name| read_to_string(dir.path().join(name)).unwrap();
        let pub_file = read("k.pub");
        Self {
            private: read("k"),
            // 改行だけを取る（コメントの末尾の空白は残す）
            pub_line: pub_file
                .strip_suffix('\n')
                .expect(".pub は改行で終わる")
                .to_owned(),
        }
    }

    /// meikey でこの方式で作った鍵。テストの実行ごとに方式ごとに1回だけ作り、使い回す
    /// （どのテストも中身を読むだけで、ファイルは `write_to` でテストごとの一時ディレクトリに書く）
    pub fn shared(mode: Mode) -> &'static Self {
        static KEYS: [OnceLock<KeyFiles>; 2] = [const { OnceLock::new() }; 2];
        KEYS[mode as usize].get_or_init(|| match var("NEXTEST_RUN_ID") {
            Ok(run) => Self::shared_across_processes(&run, mode),
            Err(_) => Self::generate(mode), // cargo test: 1つのプロセスの中で使い回す
        })
    }

    /// nextest はテストを1件ずつ別のプロセスで動かすので、同じ実行（run）の中ではファイルにして使い回す。
    /// 置き場所は target/tmp/shared-keys/<実行 ID>/<方式>/
    fn shared_across_processes(run: &str, mode: Mode) -> Self {
        let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("shared-keys");
        let dir = root.join(run).join(format!("{mode:?}"));
        if let Some(keys) = Self::read_from(&dir) {
            return keys;
        }
        let keys = Self::generate(mode);
        let parent = dir.parent().unwrap();
        create_dir_all(parent).unwrap();
        remove_old_runs(&root, run);
        // 一時ディレクトリに書いてから名前を変えて置く（同時に置いたプロセスがあれば、先に置かれた方を使う）
        let tmp = tempdir_in(parent).unwrap();
        keys.write_to(tmp.path());
        if rename(tmp.path(), &dir).is_ok() {
            let _ = tmp.keep(); // 置いたので消さない
            return keys;
        }
        Self::read_from(&dir).expect("ほかのプロセスが置いた鍵を読める")
    }

    fn read_from(dir: &Path) -> Option<Self> {
        let private = read_to_string(dir.join("id_ed25519")).ok()?;
        let pub_file = read_to_string(dir.join("id_ed25519.pub")).ok()?;
        Some(Self {
            private,
            pub_line: pub_file.strip_suffix('\n')?.to_owned(),
        })
    }

    /// ライブラリで作った鍵（`ssh-keygen` に読ませる用）
    pub fn from_key(key: &VanityKey, comment: &str) -> Self {
        Self {
            private: private_openssh(key, comment, 0xc0ffee),
            pub_line: key.public_openssh(comment),
        }
    }

    /// `dir` に `id_ed25519` と `id_ed25519.pub` を書いて、そのパスを返す（秘密鍵は Unix では 0600）
    pub fn write_to(&self, dir: &Path) -> (PathBuf, PathBuf) {
        let private = dir.join("id_ed25519");
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        opts.mode(0o600);
        opts.open(&private)
            .unwrap()
            .write_all(self.private.as_bytes())
            .unwrap();
        let public = private.with_added_extension("pub");
        write(&public, format!("{}\n", self.pub_line)).unwrap();
        (private, public)
    }

    /// `ssh-ed25519 AAAA...`（コメントを除いた部分）
    pub fn key_part(&self) -> String {
        key_part(&self.pub_line)
    }

    /// `.pub` の 32 バイトの公開鍵（独立実装で読む）
    pub fn public(&self) -> [u8; 32] {
        parse_authorized_key(&self.pub_line).unwrap().public_key
    }

    /// 秘密鍵ファイルの中のシード k（独立実装で読む）
    pub fn seed(&self) -> [u8; 32] {
        parse_private_key_file(&self.private).unwrap().keys[0].k
    }
}

/// 前の実行で使い回した鍵（1時間より前のもの）を消す
fn remove_old_runs(root: &Path, current: &str) {
    for entry in read_dir(root).into_iter().flatten().flatten() {
        let old = entry
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|t| t.elapsed().is_ok_and(|age| age > Duration::from_hours(1)));
        if old && entry.file_name().to_str() != Some(current) {
            let _ = remove_dir_all(entry.path());
        }
    }
}

/// OpenSSH の `ssh-keygen -t ed25519` で `dir/name` に作った鍵（meikey を通さない、比べる相手）。
/// `passphrase` が空なら暗号化しない。`args` は `-Z`（暗号方式）・`-a`（bcrypt の回数）など
pub fn ssh_keygen_ed25519(
    dir: &Path,
    name: &str,
    comment: &str,
    passphrase: &str,
    args: &[&str],
) -> KeyFiles {
    let path = dir.join(name);
    run(Command::new("ssh-keygen")
        .args(["-q", "-t", "ed25519", "-N", passphrase, "-C", comment])
        .args(args)
        .arg("-f")
        .arg(&path));
    let pub_file = read_to_string(path.with_added_extension("pub")).unwrap();
    KeyFiles {
        private: read_to_string(&path).unwrap(),
        pub_line: pub_file.strip_suffix('\n').unwrap().to_owned(),
    }
}

/// `.pub` の1行の SHA256 フィンガープリント（独立実装で計算）
pub fn fingerprint(pub_line: &str) -> String {
    sha256_fingerprint(&public_blob(
        &parse_authorized_key(pub_line).unwrap().public_key,
    ))
}

/// パスフレーズを標準入力から読まない OpenSSH のコマンド（ssh・ssh-add）に、パスフレーズを渡すスクリプト。
/// `SSH_ASKPASS` に指定し、`SSH_ASKPASS_REQUIRE=force` で使わせる（[`with_askpass`]）。
/// 最初の1回だけ `passphrase` を答え、2回目からは空を答える（ssh-add は違うパスフレーズだと
/// 空が返るまで聞き直すので、違うパスフレーズのテストが止まらないように）
#[cfg(unix)]
pub fn write_askpass(dir: &Path, passphrase: &str) -> PathBuf {
    let secret = dir.join("passphrase");
    write(&secret, format!("{passphrase}\n")).unwrap();
    let asked = dir.join("asked");
    let script = dir.join("askpass.sh");
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true).mode(0o700);
    let mut f = opts.open(&script).unwrap();
    write!(
        f,
        "#!/bin/sh\nif [ -e '{asked}' ]; then echo; else : > '{asked}'; cat '{secret}'; fi\n",
        asked = asked.display(),
        secret = secret.display()
    )
    .unwrap();
    script
}

/// `cmd` が端末ではなく `askpass` からパスフレーズを読むようにする
#[cfg(unix)]
pub fn with_askpass<'a>(cmd: &'a mut Command, askpass: &Path) -> &'a mut Command {
    cmd.env("SSH_ASKPASS", askpass)
        .env("SSH_ASKPASS_REQUIRE", "force")
        .env_remove("DISPLAY")
        .stdin(Stdio::null())
}

/// `.pub` の行からコメントを除いた部分
pub fn key_part(line: &str) -> String {
    line.split_whitespace()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

pub fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// コマンドが存在するか（起動できるか）
pub fn has_command(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("-?")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

/// 外部コマンドがなければテストをスキップする
macro_rules! require {
    ($cmd:expr) => {
        if !$crate::common::has_command($cmd) {
            eprintln!("スキップ: {} が見つかりません", $cmd);
            return;
        }
    };
}
pub(crate) use require;

/// コマンドを実行し、失敗したら標準エラーを付けてパニックする
pub fn run(cmd: &mut Command) -> Output {
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("{cmd:?} を起動できない: {e}"));
    assert_eq!(
        out.status.success(),
        true,
        "{cmd:?} が失敗\nstdout: {}\nstderr: {}",
        stdout(&out),
        stderr(&out)
    );
    out
}

/// `allowed_signers` ファイル（`ssh-keygen -Y verify` 用）を作る
pub fn write_allowed_signers(dir: &Path, identity: &str, pub_key: &str) -> PathBuf {
    let path = dir.join("allowed_signers");
    write(&path, format!("{identity} {pub_key}\n")).unwrap();
    path
}
