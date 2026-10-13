//! CLI の振る舞い（終了コード・エラー・ファイル保護・出力）のテスト（[`tests`]）と、そのヘルパー
//!
//! どのテストも新しい一時ディレクトリで meikey を起動し、そこに何が書かれたかまで確かめる。

use crate::common::*;
use std::collections::BTreeMap;
use std::fs::{read_dir, read_to_string, write};
use std::path::Path;
use std::process::Output;

/// 既定の出力先（`-o` を付けないとき）
const KEY: &str = "id_ed25519_meikey";
const PUB: &str = "id_ed25519_meikey.pub";
/// 先に置いておくファイルの中身
const OLD: &str = "old content\n";

/// `dir` に `existing` のファイルを置いてから、そこで meikey を実行する
fn run_with(dir: &Path, args: &[&str], existing: &[&str]) -> Output {
    for name in existing {
        write(dir.join(name), OLD).unwrap();
    }
    meikey_in(dir, args)
}

/// ディレクトリの中身（ファイル名 → 中身）
fn files_in(dir: &Path) -> BTreeMap<String, String> {
    read_dir(dir)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            let name = e.file_name().into_string().unwrap();
            (name, read_to_string(e.path()).unwrap())
        })
        .collect()
}

mod tests {
    use super::*;
    use crate::common::ssh_format::{PEM_BEGIN, PEM_END};
    use meikey::Simd;
    use meikey::controller::format::row;
    use rstest::rstest;
    use similar_asserts::assert_eq;
    #[cfg(unix)]
    use std::fs::metadata;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use tempfile::TempDir;

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
        dir: TempDir,
    ) {
        let out = run_with(dir.path(), args, existing);
        assert_eq!(out.status.success(), false);
        assert_eq!(stderr(&out).contains(message), true, "{}", stderr(&out));

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
        dir: TempDir,
    ) {
        let out = run_with(dir.path(), args, &[]);
        assert_eq!(out.status.success(), true, "{}", stderr(&out));
        for want in want_stdout {
            assert_eq!(
                stdout(&out).contains(want),
                true,
                "{want:?} が無い: {}",
                stdout(&out)
            );
        }
        for (label, value) in want_stderr {
            let want = row(label, value);
            assert_eq!(
                stderr(&out).lines().any(|l| l == want),
                true,
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
            ("avx512-ifma", Simd::Avx512Ifma),
            ("avx512", Simd::Avx512),
            ("avx2", Simd::Avx2),
            ("off", Simd::Off)
        )]
        level: (&str, Simd),
        dir: TempDir,
    ) {
        let (arg, limit) = level;
        let out = run_with(dir.path(), &["--check", "--simd", arg], &[]);
        assert_eq!(out.status.success(), true, "{}", stderr(&out));
        let err = stderr(&out);
        let used = limit.effective();
        assert_eq!(
            err.lines().any(|l| l == row("SIMD", used.name())),
            true,
            "{err}"
        );
        // CPU の行には、SIMD ごとに対応していれば ✓、していなければ ✗
        let cpu = err
            .lines()
            .find(|l| l.starts_with("CPU "))
            .unwrap_or_else(|| panic!("CPU の行が無い: {err}"));
        for simd in Simd::ALL.into_iter().filter(|&s| s != Simd::Off) {
            let mark = if simd.supported() { "✓" } else { "✗" };
            let want = format!("{} {mark}", simd.name());
            assert_eq!(
                cpu.split("  ").any(|item| item.trim() == want),
                true,
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
        dir: TempDir,
    ) {
        let mut args = vec![pattern, "-j", "2", "-C", COMMENT];
        args.extend(flags);
        args.extend(mode.args());
        let out = run_with(dir.path(), &args, existing);
        assert_eq!(out.status.success(), true, "{}", stderr(&out));

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
        assert_eq!(
            start >= "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI".len(),
            true,
            "{marker}"
        );
        let found = &pub_line[start..start + pattern.len()];
        if flags.contains(&"-s") {
            assert_eq!(found, pattern);
        } else {
            assert_eq!(found.to_ascii_lowercase(), pattern.to_ascii_lowercase());
        }

        // 秘密鍵は OpenSSH 形式の PEM（70 文字で折り返し）で、Unix では所有者だけが読める
        let body = private
            .strip_circumfix(PEM_BEGIN, PEM_END)
            .expect("OpenSSH 秘密鍵の PEM");
        assert_eq!(
            body.lines().all(|l| (1..=70).contains(&l.len())),
            true,
            "{private}"
        );
        #[cfg(unix)]
        {
            let mode = metadata(dir.path().join(KEY)).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "秘密鍵は所有者だけ読める");
        }
    }
}
