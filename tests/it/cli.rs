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
    #[case::hyphen(&["wako-kara"], &[], "'-' は使えません")]
    #[case::space(&["wako kara"], &[], "' ' は使えません")]
    #[case::padding(&["wako="], &[], "'=' は使えません")]
    #[case::japanese(&["わこから"], &[], "'わ' は使えません")]
    #[case::empty(&[""], &[], "空")]
    #[case::too_long(&[&*"a".repeat(43)], &[], "長すぎます")]
    // 引数が不正（clap が弾く）
    #[case::no_target(&[], &[], "<TARGET>")]
    #[case::unknown_flag(&["wako", "--nope"], &[], "'--nope'")]
    #[case::zero_threads(&["wako", "-j", "0"], &[], "'--threads <THREADS>'")]
    #[case::too_many_threads(&["wako", "-j", "65536"], &[], "'--threads <THREADS>'")]
    #[case::bench_zero(&["wako", "--bench", "0"], &[], "0 より大きい秒数")]
    #[case::bench_negative(&["wako", "--bench", "-1"], &[], "0 より大きい秒数")]
    #[case::bench_infinite(&["wako", "--bench", "inf"], &[], "0 より大きい秒数")]
    #[case::bench_nan(&["wako", "--bench", "NaN"], &[], "0 より大きい秒数")]
    #[case::bench_not_a_number(&["wako", "--bench", "abc"], &[], "は秒数ではありません")]
    #[case::unknown_simd(&["wako", "--simd", "avx1024"], &[], "'--simd <LEVEL>'")]
    // コメントに改行（.pub が2行に割れる）
    #[case::comment_lf(&[PATTERN, "-C", "wako\nkara"], &[], "コメントに使えない文字")]
    #[case::comment_cr(&[PATTERN, "-C", "wako\rkara"], &[], "コメントに使えない文字")]
    #[case::comment_crlf(&[PATTERN, "-C", "wako\r\nkara"], &[], "コメントに使えない文字")]
    // 出力先が既にある（--force なし）
    #[case::exists_both(&[PATTERN], &[KEY, PUB], "既に存在します")]
    #[case::exists_private(&[PATTERN], &[KEY], "既に存在します")]
    #[case::exists_pub(&[PATTERN], &[PUB], "既に存在します")]
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
        &["鍵/秒（1スレッド）", "「wakokara」の期待時間"],
        &["1スレッド・倍算=テーブル・圧縮=まとめて・SIMD=", "・検索=memmem で0.1秒間"]
    )]
    #[case::bench_fast_2_threads(
        &["wakokara", "--bench", "0.1", "-j", "2"],
        &["鍵/秒（2スレッド）", "「wakokara」の期待時間"],
        &["2スレッド・倍算=テーブル・圧縮=まとめて・SIMD=", "・検索=memmem で0.1秒間"]
    )]
    #[case::bench_no_batch(
        &["wakokara", "--bench", "0.1", "-j", "1", "--no-batch"],
        &["鍵/秒（1スレッド）"],
        &["1スレッド・倍算=テーブル・圧縮=1鍵ずつ・SIMD=", "・検索=memmem で0.1秒間"]
    )]
    #[case::bench_constant_time(
        &["wakokara", "--bench", "0.1", "-j", "1", "--constant-time"],
        &["鍵/秒（1スレッド）"],
        &["1スレッド・倍算=定数時間・圧縮=まとめて・SIMD=", "・検索=memmem で0.1秒間"]
    )]
    #[case::bench_no_simd(
        &["wakokara", "--bench", "0.1", "-j", "1", "--simd", "off"],
        &["鍵/秒（1スレッド）"],
        &["1スレッド・倍算=テーブル・圧縮=まとめて・SIMD=なし・検索=memmem で0.1秒間"]
    )]
    #[case::bench_no_memmem(
        &["wakokara", "--bench", "0.1", "-j", "1", "--no-memmem"],
        &["鍵/秒（1スレッド）"],
        &["1スレッド・倍算=テーブル・圧縮=まとめて・SIMD=", "・検索=総当たり で0.1秒間"]
    )]
    // 1文字なら毎回のように当たるが、--bench は当たっても止まらずに最後まで測る
    #[case::bench_frequent_hits(
        &["a", "--bench", "0.1", "-j", "1"],
        &["鍵/秒（1スレッド）", "「a」の期待時間"],
        &["で0.1秒間"]
    )]
    #[case::bench_baseline(
        &["wakokara", "--bench", "0.1", "-j", "1", "--constant-time", "--no-batch", "--simd", "off", "--no-memmem"],
        &["鍵/秒（1スレッド）"],
        &["1スレッド・倍算=定数時間・圧縮=1鍵ずつ・SIMD=なし・検索=総当たり で0.1秒間"]
    )]
    fn succeeds_without_writing(
        #[case] args: &[&str],
        #[case] want_stdout: &[&str],
        #[case] want_stderr: &[&str],
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
        for want in want_stderr {
            assert!(
                stderr(&out).contains(want),
                "{want:?} が無い: {}",
                stderr(&out)
            );
        }
        assert_eq!(files_in(dir.path()), BTreeMap::new(), "何も書かない");
    }

    /// `--check` は、各 SIMD に CPU が対応しているかと、`--simd` の上限で実際に使うもの（CPU が
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
        assert!(err.contains(&format!("・SIMD={}", used.name())), "{err}");
        for simd in meikey::Simd::ALL
            .into_iter()
            .filter(|&s| s != meikey::Simd::Off)
        {
            let line = err
                .lines()
                .find(|l| l.starts_with(&format!("- {}（", simd.name())))
                .unwrap_or_else(|| panic!("{} の行が無い: {err}", simd.name()));
            let want = match (simd == used, simd.supported()) {
                (true, _) => "対応（使います）",
                (false, true) => ": 対応",
                (false, false) => "非対応",
            };
            assert!(line.ends_with(want), "{line}");
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
