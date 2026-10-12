//! CLI の振る舞い（終了コード・エラー・ファイル保護・出力）のテスト（[`tests`]）と、そのヘルパー
//!
//! どのテストも新しい一時ディレクトリで meikey を起動し、そこに何が書かれたかまで確かめる。

use crate::common::*;
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Output;

/// 既定の出力先（`-o` を付けないとき）
const KEY: &str = "id_ed25519_meikey";
const PUB: &str = "id_ed25519_meikey.pub";
/// 先に置いておくファイルの中身
const OLD: &str = "old content\n";

/// `existing` のファイルを置いた一時ディレクトリで meikey を実行する
fn run_in_tempdir(args: &[&str], existing: &[&str]) -> (tempfile::TempDir, Output) {
    let dir = tempfile::tempdir().unwrap();
    for name in existing {
        std::fs::write(dir.path().join(name), OLD).unwrap();
    }
    let out = meikey_in(dir.path(), args);
    (dir, out)
}

/// ディレクトリの中身（ファイル名 → 中身）
fn files_in(dir: &Path) -> BTreeMap<String, String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            let name = e.file_name().into_string().unwrap();
            (name, std::fs::read_to_string(e.path()).unwrap())
        })
        .collect()
}

mod tests {
    use super::*;
    use rstest::rstest;

    /// 失敗するときは探索を始める前に止まり、ディレクトリを一切変えない（既存のファイルもそのまま）
    #[rstest]
    // 探す文字列が使えない
    #[case::hyphen(&["wako-kara"], &[], "使えない文字: '-'")]
    #[case::space(&["wako kara"], &[], "使えない文字: ' '")]
    #[case::padding(&["wako="], &[], "使えない文字: '='")]
    #[case::japanese(&["わこから"], &[], "使えない文字: 'わ'")]
    #[case::empty(&[""], &[], "空")]
    #[case::too_long(&[&*"a".repeat(43)], &[], "長すぎる")]
    // 引数が不正（clap が弾く）
    #[case::no_target(&[], &[], "<TARGET>")]
    #[case::unknown_flag(&["wako", "--nope"], &[], "'--nope'")]
    #[case::zero_threads(&["wako", "-j", "0"], &[], "'--threads <THREADS>'")]
    #[case::too_many_threads(&["wako", "-j", "65536"], &[], "'--threads <THREADS>'")]
    #[case::bench_zero(&["wako", "--bench", "0"], &[], "0 より大きい秒数")]
    #[case::bench_negative(&["wako", "--bench", "-1"], &[], "0 より大きい秒数")]
    #[case::bench_infinite(&["wako", "--bench", "inf"], &[], "0 より大きい秒数")]
    #[case::bench_nan(&["wako", "--bench", "NaN"], &[], "0 より大きい秒数")]
    #[case::bench_not_a_number(&["wako", "--bench", "abc"], &[], "秒数ではない")]
    #[case::unknown_simd(&["wako", "--simd", "avx1024"], &[], "'--simd <LEVEL>'")]
    // コメントに改行（.pub が2行に割れる）
    #[case::comment_lf(&[PATTERN, "-C", "wako\nkara"], &[], "コメントに使えない文字")]
    #[case::comment_cr(&[PATTERN, "-C", "wako\rkara"], &[], "コメントに使えない文字")]
    #[case::comment_crlf(&[PATTERN, "-C", "wako\r\nkara"], &[], "コメントに使えない文字")]
    // 出力先が既にある（--force なし）
    #[case::exists_both(&[PATTERN], &[KEY, PUB], "既に存在")]
    #[case::exists_private(&[PATTERN], &[KEY], "既に存在")]
    #[case::exists_pub(&[PATTERN], &[PUB], "既に存在")]
    fn fails_without_writing(
        #[case] args: &[&str],
        #[case] existing: &[&str],
        #[case] message: &str,
    ) {
        let (dir, out) = run_in_tempdir(args, existing);
        assert!(!out.status.success());
        assert!(stderr(&out).contains(message), "{}", stderr(&out));

        let want: BTreeMap<String, String> = existing
            .iter()
            .map(|&name| (name.to_owned(), OLD.to_owned()))
            .collect();
        assert_eq!(files_in(dir.path()), want, "ディレクトリは元のまま");
    }

    /// 鍵を探さないコマンドは成功し、何も書かない
    #[rstest]
    #[case::version(&["--version"], &["meikey 0."], &[])]
    #[case::short_version(&["-V"], &["meikey 0."], &[])]
    #[case::help(&["--help"], &["Usage: meikey", "--constant-time", "--no-batch", "--simd <LEVEL>", "--no-memmem", "--bench <SECS>"], &[])]
    #[case::short_help(&["-h"], &["Usage: meikey"], &[])]
    // --bench は、どの組み合わせで測ったかを表示する
    #[case::bench_fast(
        &["wakokara", "--bench", "0.1", "-j", "1"],
        &["鍵/秒", "期待時間"],
        &[("並列", "1 スレッド"), ("倍算", "テーブル"), ("逆元", "256 鍵ごとに 1 回"), ("検索", "memmem"), ("計測", "0.1 秒")]
    )]
    #[case::bench_fast_2_threads(
        &["wakokara", "--bench", "0.1", "-j", "2"],
        &["鍵/秒", "期待時間"],
        &[("並列", "2 スレッド"), ("文字列", "wakokara（大小文字の区別なし）")]
    )]
    #[case::bench_no_batch(
        &["wakokara", "--bench", "0.1", "-j", "1", "--no-batch"],
        &["鍵/秒"],
        &[("倍算", "テーブル"), ("逆元", "1 鍵ごとに 1 回")]
    )]
    #[case::bench_constant_time(
        &["wakokara", "--bench", "0.1", "-j", "1", "--constant-time"],
        &["鍵/秒"],
        &[("倍算", "定数時間"), ("逆元", "256 鍵ごとに 1 回")]
    )]
    #[case::bench_no_simd(
        &["wakokara", "--bench", "0.1", "-j", "1", "--simd", "off"],
        &["鍵/秒"],
        &[("SIMD", "なし")]
    )]
    #[case::bench_no_memmem(
        &["wakokara", "--bench", "0.1", "-j", "1", "--no-memmem"],
        &["鍵/秒"],
        &[("検索", "総当たり")]
    )]
    // 1文字なら毎回のように当たるが、--bench は当たっても止まらずに最後まで測る
    #[case::bench_frequent_hits(
        &["a", "--bench", "0.1", "-j", "1", "-s"],
        &["鍵/秒", "期待時間"],
        &[("文字列", "a（大小文字の区別あり）"), ("計測", "0.1 秒")]
    )]
    #[case::bench_baseline(
        &["wakokara", "--bench", "0.1", "-j", "1", "--constant-time", "--no-batch", "--simd", "off", "--no-memmem"],
        &["鍵/秒"],
        &[("SIMD", "なし"), ("倍算", "定数時間"), ("逆元", "1 鍵ごとに 1 回"), ("検索", "総当たり")]
    )]
    fn succeeds_without_writing(
        #[case] args: &[&str],
        #[case] want_stdout: &[&str],
        #[case] want_stderr: &[(&str, &str)],
    ) {
        let (dir, out) = run_in_tempdir(args, &[]);
        assert!(out.status.success(), "{}", stderr(&out));
        for want in want_stdout {
            assert!(
                stdout(&out).contains(want),
                "{want:?} が無い: {}",
                stdout(&out)
            );
        }
        for (label, value) in want_stderr {
            let want = meikey::controller::format::row(label, value);
            assert!(
                stderr(&out).lines().any(|l| l == want),
                "{want:?} が無い: {}",
                stderr(&out)
            );
        }
        assert_eq!(files_in(dir.path()), BTreeMap::new(), "何も書かない");
    }

    /// `--check` は、各 SIMD に CPU が対応しているか（✓・✗）と、`--simd` の上限で実際に使うもの（CPU が
    /// 対応していなければ下がったもの）を表示する
    #[rstest]
    fn check_reports_simd_in_use(
        #[values(
            ("avx512-ifma", meikey::Simd::Avx512Ifma),
            ("avx512", meikey::Simd::Avx512),
            ("avx2", meikey::Simd::Avx2),
            ("off", meikey::Simd::Off)
        )]
        level: (&str, meikey::Simd),
    ) {
        let (arg, limit) = level;
        let (dir, out) = run_in_tempdir(&["--check", "--simd", arg], &[]);
        assert!(out.status.success(), "{}", stderr(&out));
        let err = stderr(&out);
        let used = limit.effective();
        let row = meikey::controller::format::row;
        assert!(err.lines().any(|l| l == row("SIMD", used.name())), "{err}");
        // CPU の行には、SIMD ごとに対応していれば ✓、していなければ ✗
        let cpu = err
            .lines()
            .find(|l| l.starts_with("CPU "))
            .unwrap_or_else(|| panic!("CPU の行が無い: {err}"));
        for simd in meikey::Simd::ALL
            .into_iter()
            .filter(|&s| s != meikey::Simd::Off)
        {
            let mark = if simd.supported() { "✓" } else { "✗" };
            let want = format!("{} {mark}", simd.name());
            assert!(
                cpu.split("  ").any(|item| item.trim() == want),
                "{want} が無い: {cpu}"
            );
        }
        assert_eq!(files_in(dir.path()), BTreeMap::new(), "何も書かない");
    }

    /// 鍵を探して書き出す。標準出力は `.pub` と同じ1行で、標準エラーの `^` が一致した位置を指す。
    /// `-s` なら大小文字まで一致し、`-f` なら既存のファイルを新しい鍵で置き換える
    #[rstest]
    #[case::letters("Mei", &[], &[])]
    #[case::with_digit("w4k", &[], &[])]
    #[case::symbols("+/", &[], &[])]
    #[case::upper_and_digit("Z9", &[], &[])]
    #[case::case_sensitive_lower_upper("mE", &["-s"], &[])]
    #[case::case_sensitive_upper_lower("Wa", &["-s"], &[])]
    #[case::case_sensitive_with_digit("k0", &["-s"], &[])]
    #[case::force_both(PATTERN, &["-f"], &[KEY, PUB])]
    #[case::force_private(PATTERN, &["-f"], &[KEY])]
    #[case::force_pub(PATTERN, &["-f"], &[PUB])]
    fn writes_key(
        #[case] pattern: &str,
        #[case] flags: &[&str],
        #[case] existing: &[&str],
        #[values(Mode::Fast, Mode::ConstantTime, Mode::Baseline)] mode: Mode,
    ) {
        let mut args = vec![pattern, "-j", "2", "-C", COMMENT];
        args.extend(flags);
        args.extend(mode.args());
        let (dir, out) = run_in_tempdir(&args, existing);
        assert!(out.status.success(), "{}", stderr(&out));

        // 書かれるのは秘密鍵と .pub の2つだけ（既存のファイルは置き換わる）
        let files = files_in(dir.path());
        assert_eq!(files.keys().collect::<Vec<_>>(), [KEY, PUB]);
        let (private, pub_file) = (&files[KEY], &files[PUB]);

        // 標準出力は .pub と同じ1行（パイプで authorized_keys に流せる）。.pub は「種類 base64 コメント」
        assert_eq!(&stdout(&out), pub_file);
        let pub_line = pub_file.strip_suffix('\n').expect(".pub は改行で終わる");
        let fields: Vec<&str> = pub_line.split(' ').collect();
        assert_eq!(fields.len(), 3, "{pub_line}");
        assert_eq!((fields[0], fields[2]), ("ssh-ed25519", COMMENT));

        // ^ は固定部分（ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI）より後ろの、一致した位置を指す
        let marker = stderr(&out)
            .lines()
            .find(|l| l.trim_start().starts_with('^'))
            .expect("^ の行がある")
            .to_owned();
        let start = marker.find('^').unwrap();
        assert_eq!(marker.trim(), "^".repeat(pattern.len()));
        assert!(start >= "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI".len());
        let found = &pub_line[start..start + pattern.len()];
        if flags.contains(&"-s") {
            assert_eq!(found, pattern);
        } else {
            assert!(found.eq_ignore_ascii_case(pattern), "{found} ≠ {pattern}");
        }

        // 秘密鍵は OpenSSH 形式の PEM（70 文字で折り返し）で、Unix では所有者だけが読める
        let body = private
            .strip_circumfix(ssh_format::PEM_BEGIN, ssh_format::PEM_END)
            .expect("OpenSSH 秘密鍵の PEM");
        assert!(
            body.lines().all(|l| (1..=70).contains(&l.len())),
            "{private}"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.path().join(KEY))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600, "秘密鍵は所有者だけ読める");
        }
    }
}
