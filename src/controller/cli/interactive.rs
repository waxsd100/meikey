//! 引数なしで端末から起動したときの対話: 質問に答えてもらい、同じ意味の引数を組み立てる
//!
//! 答えは引数に直して [`super::Cli`] で読み直すので、引数で起動したときと同じ検査・同じ処理になる。
//! 質問は [`Prompt`] 越しにするので、テストでは答えを並べた台本に差し替えられる

use super::{CliError, ValueError, execute, parse_from, parse_secs};
use crate::lang::{t, tf};
use crate::model::key::{InvalidComment, check_comment};
use crate::model::pattern::{Pattern, PatternError};
use crate::repository::key_file::{KeyFileError, KeyFiles, KeyRepository};
use dialoguer::theme::ColorfulTheme;
use dialoguer::{Confirm, Input, Select};
use std::io::{self, IsTerminal};
use std::iter::once;
use std::path::Path;

/// 秘密鍵の保存先の既定（`-o` の既定と同じ）
const DEFAULT_OUTPUT: &str = "id_ed25519_meikey";
/// 速さを測る秒数の既定
const DEFAULT_BENCH: &str = "10";

/// 質問を続けられない理由
#[derive(Debug, thiserror::Error)]
pub enum AskError {
    /// やめた（Esc・Ctrl+C か、「始める？」にいいえ）
    #[error("{}", t!("やめた", "Cancelled"))]
    Cancelled,
    /// 端末を読み書きできない
    #[error(transparent)]
    Io(#[from] io::Error),
    /// 保存先に鍵のファイルがあるか調べられない
    #[error(transparent)]
    KeyFile(#[from] KeyFileError),
}

/// 端末での質問のエラー。Ctrl+C で読むのを止められたら、やめたことにする
impl From<dialoguer::Error> for AskError {
    fn from(e: dialoguer::Error) -> Self {
        let dialoguer::Error::IO(e) = e;
        match e.kind() {
            io::ErrorKind::Interrupted => AskError::Cancelled,
            _ => AskError::Io(e),
        }
    }
}

/// 入れた文字列が使えない理由（見せて入れ直してもらう）
#[derive(Debug, thiserror::Error)]
pub enum InputError {
    #[error(transparent)]
    Pattern(#[from] PatternError),
    #[error(transparent)]
    Comment(#[from] InvalidComment),
    #[error(transparent)]
    Value(#[from] ValueError),
    /// 何も入っていない（空白だけ）
    #[error("{}", t!("空", "empty"))]
    Empty,
}

/// 引数なしで、標準入力と標準エラーが端末のときに対話にする（パイプやテストからの起動では今までどおり）
pub(super) fn wanted() -> bool {
    std::env::args_os().len() == 1 && io::stdin().is_terminal() && io::stderr().is_terminal()
}

/// 質問に答えてもらって実行する。途中でやめたら（Esc・Ctrl+C）何もせずに終わる
pub(super) fn run() -> Result<(), CliError> {
    // 選ぶ途中で Ctrl+C されると、選択肢を出す間に隠したカーソルが消えたままになるので、戻してから終わる
    // （終了コードは Ctrl+C で止められたときと同じ 130）
    let _ = ctrlc::set_handler(|| {
        let _ = dialoguer::console::Term::stderr().show_cursor();
        eprintln!();
        std::process::exit(130);
    });
    let args = match ask(&mut Terminal, &KeyFiles) {
        Ok(args) => args,
        Err(e @ AskError::Cancelled) => {
            let _ = dialoguer::console::Term::stderr().show_cursor();
            eprintln!("{e}");
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };
    let cli = parse_from(once("meikey".to_string()).chain(args))?;
    execute(&cli)
}

/// このプロセスのためだけに開いたコンソール（エクスプローラーからダブルクリックしたときなど）なら、
/// 終わってすぐ閉じないように Enter を待つ
pub(super) fn finish() {
    if launched_alone() {
        eprint!("{}", t!("Enter で閉じる", "Press Enter to close"));
        let _ = io::stdin().read_line(&mut String::new());
    }
}

/// コンソールにつながっているのがこのプロセスだけか（Windows のみ。ほかでは端末を閉じないので false）
#[cfg(windows)]
fn launched_alone() -> bool {
    use windows_sys::Win32::System::Console::GetConsoleProcessList;
    let mut ids = [0u32; 2];
    // SAFETY: ids は2要素あり、その数を渡している
    unsafe { GetConsoleProcessList(ids.as_mut_ptr(), ids.len() as u32) == 1 }
}

#[cfg(not(windows))]
fn launched_alone() -> bool {
    false
}

/// 質問のしかた。やめたら（Esc・Ctrl+C）[`AskError::Cancelled`]
pub(super) trait Prompt {
    /// `items` から1つ選んでもらう（何番目か）
    fn select(&mut self, question: &str, items: &[&str], default: usize)
    -> Result<usize, AskError>;

    /// 文字列を入れてもらう（何も入れなければ `default`）。`check` が通らなければ、理由を見せて入れ直してもらう
    fn input(
        &mut self,
        question: &str,
        default: &str,
        check: &dyn Fn(&str) -> Result<(), InputError>,
    ) -> Result<String, AskError>;

    /// はい・いいえで答えてもらう
    fn confirm(&mut self, question: &str, default: bool) -> Result<bool, AskError>;

    /// 知らせるだけの行
    fn note(&mut self, message: &str);
}

/// 端末で質問する [`Prompt`]（本番で使うもの）。質問も答えも標準エラーに出す
struct Terminal;

impl Prompt for Terminal {
    fn select(
        &mut self,
        question: &str,
        items: &[&str],
        default: usize,
    ) -> Result<usize, AskError> {
        Select::with_theme(&ColorfulTheme::default())
            .with_prompt(question)
            .items(items)
            .default(default)
            .interact_opt()?
            .ok_or(AskError::Cancelled)
    }

    fn input(
        &mut self,
        question: &str,
        default: &str,
        check: &dyn Fn(&str) -> Result<(), InputError>,
    ) -> Result<String, AskError> {
        Ok(Input::<String>::with_theme(&ColorfulTheme::default())
            .with_prompt(question)
            .default(default.to_string())
            .show_default(!default.is_empty())
            .allow_empty(true)
            .validate_with(|s: &String| check(s))
            .interact_text()?)
    }

    fn confirm(&mut self, question: &str, default: bool) -> Result<bool, AskError> {
        Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt(question)
            .default(default)
            .interact_opt()?
            .ok_or(AskError::Cancelled)
    }

    fn note(&mut self, message: &str) {
        eprintln!("{message}");
    }
}

/// 質問して、答えと同じ意味の引数（`meikey` の後ろ）を返す。既定と同じものは付けない
pub(super) fn ask(p: &mut impl Prompt, repo: &impl KeyRepository) -> Result<Vec<String>, AskError> {
    let mut args = Vec::new();
    let action = p.select(
        t!("何をする", "What to do"),
        &[
            t!("鍵を作る", "Make a key"),
            t!(
                "速さを測る（鍵は作らない）",
                "Measure the speed (no key is made)"
            ),
            t!(
                "使う CPU・GPU と計算方法を見る",
                "Show the CPU, GPU and methods to use"
            ),
        ],
        0,
    )?;

    if action == 2 {
        args.push("--check".to_string());
        ask_device(p, &mut args)?;
        return start(p, args);
    }

    let target = p.input(
        t!(
            "公開鍵に入れたい文字列（A-Z a-z 0-9 + /、最大42文字）",
            "Text to put in the public key (A-Z a-z 0-9 + /, up to 42 characters)"
        ),
        "",
        &|s| Ok(Pattern::new(s, false).map(drop)?),
    )?;
    let letters = target.bytes().any(|b| b.is_ascii_alphabetic());
    args.push(target);
    if letters
        && p.confirm(
            t!(
                "大文字小文字を区別する？（区別すると英字1文字ごとに約2倍かかる）",
                "Match upper and lower case exactly? (each letter makes it about 2 times slower)"
            ),
            false,
        )?
    {
        args.push("-s".to_string());
    }
    ask_device(p, &mut args)?;

    if action == 1 {
        let secs = p.input(
            t!("測る秒数", "Seconds to measure"),
            DEFAULT_BENCH,
            &|s| Ok(parse_secs(s).map(drop)?),
        )?;
        args.push(format!("--bench={secs}"));
        return start(p, args);
    }

    let (output, force) = ask_output(p, repo)?;
    if output != DEFAULT_OUTPUT {
        args.push(format!("--output={output}"));
    }
    if force {
        args.push("--force".to_string());
    }
    let comment = p.input(
        t!(
            "鍵のコメント（無くてよい）",
            "Comment for the key (optional)"
        ),
        "",
        &|s| Ok(check_comment(s)?),
    )?;
    if !comment.is_empty() {
        args.push(format!("--comment={comment}"));
    }
    start(p, args)
}

/// 探すのに使うもの（GPU を使える版だけ聞く。CPU なら引数は付けない）
fn ask_device(p: &mut impl Prompt, args: &mut Vec<String>) -> Result<(), AskError> {
    #[cfg(feature = "cuda")]
    match p.select(
        t!("使うもの", "Use"),
        &[
            "CPU",
            "GPU",
            t!("GPU と CPU の両方", "GPU and CPU together"),
        ],
        0,
    )? {
        1 => args.push("--gpu".to_string()),
        2 => args.push("--both".to_string()),
        _ => {}
    }
    let _ = (p, args);
    Ok(())
}

/// 秘密鍵の保存先と、上書きするか。既にあれば上書きするか聞き（しなければ別の保存先を聞き直す）、
/// 書けない場所なら理由を見せて聞き直す
fn ask_output(p: &mut impl Prompt, repo: &impl KeyRepository) -> Result<(String, bool), AskError> {
    loop {
        let output = p.input(
            t!(
                "秘密鍵の保存先（公開鍵は <保存先>.pub）",
                "Where to save the private key (the public key goes to <path>.pub)"
            ),
            DEFAULT_OUTPUT,
            &|s| match s.trim().is_empty() {
                true => Err(InputError::Empty),
                false => Ok(()),
            },
        )?;
        let path = Path::new(&output);
        let force = match repo.existing(path)? {
            None => false,
            Some(existing) => {
                let question = tf!(
                    "{} は既にある。上書きする？",
                    "{} already exists. Overwrite it?",
                    existing.display()
                );
                if !p.confirm(&question, false)? {
                    continue;
                }
                true
            }
        };
        match repo.check_writable(path) {
            Ok(()) => return Ok((output, force)),
            // 引数で起動したときと同じ説明（「保存できない: …」）
            Err(e) => p.note(&CliError::NotWritable(e).to_string()),
        }
    }
}

/// 同じことをする引数での起動のしかたを見せて、始めるか聞く
fn start(p: &mut impl Prompt, args: Vec<String>) -> Result<Vec<String>, AskError> {
    let line: Vec<String> = once("meikey")
        .chain(args.iter().map(String::as_str))
        .map(quote)
        .collect();
    let line = line.join(" ");
    p.note(&tf!(
        "引数で起動するなら: {line}",
        "The same with arguments: {line}"
    ));
    match p.confirm(t!("始める？", "Start?"), true)? {
        true => Ok(args),
        false => Err(AskError::Cancelled),
    }
}

/// シェルにそのまま貼れるように、必要なときだけ引用符で囲む
/// （Unix は '…'、Windows は "…"。中の引用符は、それぞれ '\'' と "" にする）
fn quote(s: &str) -> String {
    let plain = !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "+/-_.=:@,".contains(c));
    if plain {
        return s.to_string();
    }
    if cfg!(windows) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

#[cfg(test)]
mod tests {
    //! 答えを並べた台本（[`Script`]）で質問に答えて、組み立てる引数と、それを [`super::Cli`] で読めることを確かめる
    use super::*;
    use crate::repository::key_file::MockKeyRepository;
    use mockall::predicate::eq;
    use rstest::{fixture, rstest};
    use similar_asserts::assert_eq;
    use std::collections::VecDeque;
    use std::path::PathBuf;

    /// 1つの質問への答え
    #[derive(Debug, Clone, Copy)]
    enum Answer {
        Select(usize),
        /// 入れる文字列（"" なら既定のまま）
        Input(&'static str),
        Confirm(bool),
        /// Esc・Ctrl+C でやめる
        Cancel,
    }
    use Answer::*;

    /// 「使うもの」への答え（GPU を使えない版では聞かれないので、台本から外れる）
    fn device(i: usize) -> Option<Answer> {
        cfg!(feature = "cuda").then_some(Select(i))
    }

    /// 答えを順に返す [`Prompt`]。検査に通らなかった入力と、知らせた行も覚えておく
    struct Script {
        answers: VecDeque<Answer>,
        rejected: Vec<String>,
        notes: Vec<String>,
    }

    impl Script {
        fn new(answers: impl IntoIterator<Item = Option<Answer>>) -> Self {
            Script {
                answers: answers.into_iter().flatten().collect(),
                rejected: Vec::new(),
                notes: Vec::new(),
            }
        }

        fn next(&mut self, question: &str) -> Result<Answer, AskError> {
            match self.answers.pop_front() {
                Some(Cancel) => Err(AskError::Cancelled),
                Some(a) => Ok(a),
                None => panic!("台本に無い質問: {question}"),
            }
        }
    }

    impl Prompt for Script {
        fn select(
            &mut self,
            question: &str,
            items: &[&str],
            default: usize,
        ) -> Result<usize, AskError> {
            assert_eq!(default < items.len(), true);
            match self.next(question)? {
                Select(i) if i < items.len() => Ok(i),
                a => panic!("{question} に {a:?}"),
            }
        }

        fn input(
            &mut self,
            question: &str,
            default: &str,
            check: &dyn Fn(&str) -> Result<(), InputError>,
        ) -> Result<String, AskError> {
            loop {
                let Input(s) = self.next(question)? else {
                    panic!("{question} に文字列以外");
                };
                let s = if s.is_empty() { default } else { s };
                match check(s) {
                    Ok(()) => return Ok(s.to_string()),
                    Err(e) => self.rejected.push(format!("{s}: {e}")),
                }
            }
        }

        fn confirm(&mut self, question: &str, _: bool) -> Result<bool, AskError> {
            match self.next(question)? {
                Confirm(b) => Ok(b),
                a => panic!("{question} に {a:?}"),
            }
        }

        fn note(&mut self, message: &str) {
            self.notes.push(message.to_string());
        }
    }

    /// 保存先には何も無く、書ける
    #[fixture]
    fn repo() -> MockKeyRepository {
        let mut repo = MockKeyRepository::new();
        repo.expect_existing().returning(|_| Ok(None));
        repo.expect_check_writable().returning(|_| Ok(()));
        repo
    }

    /// 台本どおりに答えて引数を組み立て、台本を使い切ったことと、[`super::Cli`] で読めることを確かめる
    fn answer(script: &mut Script, repo: &MockKeyRepository) -> Vec<String> {
        let args = ask(script, repo).unwrap();
        assert_eq!(
            script.answers.len(),
            0,
            "使われなかった答え: {:?}",
            script.answers
        );
        let parsed = parse_from(once("meikey".to_string()).chain(args.clone()));
        assert_eq!(parsed.is_ok(), true, "{:?}", parsed.err());
        args
    }

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    /// 鍵を作る: 既定のまま答えれば文字列だけ、変えたものだけ引数になる。数字や記号だけなら大文字小文字は聞かない
    #[rstest]
    #[case::defaults(
        vec![Some(Select(0)), Some(Input("abc")), Some(Confirm(false)), device(0), Some(Input("")), Some(Input("")), Some(Confirm(true))],
        &["abc"]
    )]
    #[case::everything(
        vec![Some(Select(0)), Some(Input("Wako")), Some(Confirm(true)), device(0), Some(Input("keys/k")), Some(Input("me@host")), Some(Confirm(true))],
        &["Wako", "-s", "--output=keys/k", "--comment=me@host"]
    )]
    #[case::no_letters(
        vec![Some(Select(0)), Some(Input("12+/")), device(0), Some(Input("")), Some(Input("")), Some(Confirm(true))],
        &["12+/"]
    )]
    #[case::comment_with_space(
        vec![Some(Select(0)), Some(Input("abc")), Some(Confirm(false)), device(0), Some(Input("-k")), Some(Input("wako kara")), Some(Confirm(true))],
        &["abc", "--output=-k", "--comment=wako kara"]
    )]
    fn generate(
        #[case] answers: Vec<Option<Answer>>,
        #[case] want: &[&str],
        repo: MockKeyRepository,
    ) {
        let mut script = Script::new(answers);
        assert_eq!(answer(&mut script, &repo), strings(want));
        assert_eq!(script.rejected, Vec::<String>::new());
    }

    /// 使うもの: CPU なら何も付けず、GPU は --gpu、両方は --both（鍵を作る・測る・見る のどれでも聞く）
    #[cfg(feature = "cuda")]
    #[rstest]
    #[case::generate_gpu(vec![Select(0), Input("123"), Select(1), Input(""), Input(""), Confirm(true)], &["123", "--gpu"])]
    #[case::bench_both(vec![Select(1), Input("123"), Select(2), Input(""), Confirm(true)], &["123", "--both", "--bench=10"])]
    #[case::check_gpu(vec![Select(2), Select(1), Confirm(true)], &["--check", "--gpu"])]
    fn device_choice(#[case] answers: Vec<Answer>, #[case] want: &[&str], repo: MockKeyRepository) {
        let mut script = Script::new(answers.into_iter().map(Some));
        assert_eq!(answer(&mut script, &repo), strings(want));
    }

    /// 速さを測る: 秒数を聞き（既定 10 秒）、保存先やコメントは聞かない
    #[rstest]
    #[case::default_secs("", "--bench=10")]
    #[case::fraction("0.5", "--bench=0.5")]
    fn bench(#[case] secs: &'static str, #[case] want: &str) {
        // 保存先を確かめれば、モックに期待が無いのでパニックする
        let repo = MockKeyRepository::new();
        let mut script = Script::new([
            Some(Select(1)),
            Some(Input("abc")),
            Some(Confirm(false)),
            device(0),
            Some(Input(secs)),
            Some(Confirm(true)),
        ]);
        assert_eq!(answer(&mut script, &repo), strings(&["abc", want]));
    }

    /// 見る: 文字列も聞かずに --check
    #[rstest]
    fn check() {
        let mut script = Script::new([Some(Select(2)), device(0), Some(Confirm(true))]);
        assert_eq!(
            answer(&mut script, &MockKeyRepository::new()),
            strings(&["--check"])
        );
    }

    /// 検査に通らない答えは理由を見せて入れ直してもらう（文字列・秒数・保存先）
    #[rstest]
    #[case::invalid_char(
        vec![Some(Select(0)), Some(Input("ab-c")), Some(Input("abc")), Some(Confirm(false)), device(0), Some(Input("")), Some(Input("")), Some(Confirm(true))],
        &["abc"], &["ab-c: 使えない文字: '-'（使える文字: A-Z a-z 0-9 + /）"]
    )]
    #[case::empty_target(
        vec![Some(Select(0)), Some(Input("")), Some(Input("abc")), Some(Confirm(false)), device(0), Some(Input("")), Some(Input("")), Some(Confirm(true))],
        &["abc"], &[": 文字列が空"]
    )]
    #[case::too_long(
        vec![Some(Select(1)), Some(Input("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")), Some(Input("123")), device(0), Some(Input("")), Some(Confirm(true))],
        &["123", "--bench=10"], &["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: 文字列が長すぎる: 43 文字（最大 42 文字）"]
    )]
    #[case::bad_secs(
        vec![Some(Select(1)), Some(Input("123")), device(0), Some(Input("0")), Some(Input("x")), Some(Input("2")), Some(Confirm(true))],
        &["123", "--bench=2"], &["0: 0 より大きい秒数が必要", "x: 秒数ではない"]
    )]
    #[case::blank_output(
        vec![Some(Select(0)), Some(Input("123")), device(0), Some(Input("  ")), Some(Input("")), Some(Input("")), Some(Confirm(true))],
        &["123"], &["  : 空"]
    )]
    fn rejected_answers_are_asked_again(
        #[case] answers: Vec<Option<Answer>>,
        #[case] want: &[&str],
        #[case] rejected: &[&str],
        repo: MockKeyRepository,
    ) {
        let mut script = Script::new(answers);
        assert_eq!(answer(&mut script, &repo), strings(want));
        assert_eq!(script.rejected, strings(rejected));
    }

    /// 保存先が既にあれば上書きするか聞く。するなら --force、しないなら保存先を聞き直す
    #[rstest]
    #[case::overwrite(&[Confirm(true)], &["123", "--force"])]
    #[case::other_name(&[Confirm(false), Input("other")], &["123", "--output=other"])]
    fn existing_output(#[case] middle: &[Answer], #[case] want: &[&str]) {
        let mut repo = MockKeyRepository::new();
        repo.expect_existing()
            .with(eq(Path::new(DEFAULT_OUTPUT)))
            .returning(|_| Ok(Some(PathBuf::from("id_ed25519_meikey.pub"))));
        repo.expect_existing()
            .with(eq(Path::new("other")))
            .returning(|_| Ok(None));
        repo.expect_check_writable().times(1).returning(|_| Ok(()));
        let mut answers = vec![
            Some(Select(0)),
            Some(Input("123")),
            device(0),
            Some(Input("")),
        ];
        answers.extend(middle.iter().copied().map(Some));
        answers.extend([Some(Input("")), Some(Confirm(true))]);

        let mut script = Script::new(answers);
        assert_eq!(answer(&mut script, &repo), strings(want));
    }

    /// 書けない保存先（ディレクトリが無いなど）なら、理由を見せて聞き直す
    #[rstest]
    fn unwritable_output_is_asked_again() {
        let mut repo = MockKeyRepository::new();
        repo.expect_existing().returning(|_| Ok(None));
        repo.expect_check_writable()
            .with(eq(Path::new("nodir/k")))
            .returning(|_| {
                Err(KeyFileError::Io {
                    path: "nodir".into(),
                    source: io::Error::new(io::ErrorKind::NotFound, "無い"),
                })
            });
        repo.expect_check_writable()
            .with(eq(Path::new("k")))
            .returning(|_| Ok(()));
        let mut script = Script::new([
            Some(Select(0)),
            Some(Input("123")),
            device(0),
            Some(Input("nodir/k")),
            Some(Input("k")),
            Some(Input("")),
            Some(Confirm(true)),
        ]);
        assert_eq!(answer(&mut script, &repo), strings(&["123", "--output=k"]));
        assert_eq!(script.notes[0], "保存できない: nodir: 無い");
    }

    /// 始める前に、同じことをする引数での起動のしかたを見せる
    #[rstest]
    fn shows_equivalent_command(repo: MockKeyRepository) {
        let mut script = Script::new([
            Some(Select(0)),
            Some(Input("Wako")),
            Some(Confirm(true)),
            device(0),
            Some(Input("")),
            Some(Input("wako kara")),
            Some(Confirm(true)),
        ]);
        answer(&mut script, &repo);
        let want = if cfg!(windows) {
            "引数で起動するなら: meikey Wako -s \"--comment=wako kara\""
        } else {
            "引数で起動するなら: meikey Wako -s '--comment=wako kara'"
        };
        assert_eq!(script.notes, vec![want.to_string()]);
    }

    /// やめたら（Esc・Ctrl+C・「始める？」にいいえ）、AskError::Cancelled で、何も始めない
    #[rstest]
    #[case::first_question(vec![Some(Cancel)])]
    #[case::target(vec![Some(Select(0)), Some(Cancel)])]
    #[case::comment(vec![Some(Select(0)), Some(Input("123")), device(0), Some(Input("")), Some(Cancel)])]
    #[case::declined(vec![Some(Select(2)), device(0), Some(Confirm(false))])]
    fn cancel(#[case] answers: Vec<Option<Answer>>, repo: MockKeyRepository) {
        let mut script = Script::new(answers);
        let e = ask(&mut script, &repo).unwrap_err();
        assert_eq!(matches!(e, AskError::Cancelled), true, "{e:?}");
    }

    /// 引用符は要るときだけ付け、中の引用符はシェルに合わせて逃がす
    #[rstest]
    #[case::plain("--output=keys/k", "--output=keys/k", "--output=keys/k")]
    #[case::symbols("me@host,a+b:c", "me@host,a+b:c", "me@host,a+b:c")]
    #[case::space("a b", "'a b'", "\"a b\"")]
    #[case::empty("", "''", "\"\"")]
    #[case::single_quote("it's", "'it'\\''s'", "\"it's\"")]
    #[case::double_quote("say \"hi\"", "'say \"hi\"'", "\"say \"\"hi\"\"\"")]
    #[case::dollar("$HOME", "'$HOME'", "\"$HOME\"")]
    #[case::japanese("わこ", "'わこ'", "\"わこ\"")]
    fn quote_for_shell(#[case] s: &str, #[case] unix: &str, #[case] windows: &str) {
        assert_eq!(quote(s), if cfg!(windows) { windows } else { unix });
    }

    /// 英語では、質問・検査の理由・知らせる行も英語
    #[rstest]
    fn english(repo: MockKeyRepository) {
        crate::lang::set(crate::lang::Lang::En);
        let mut script = Script::new([
            Some(Select(0)),
            Some(Input("ab-c")),
            Some(Input("abc")),
            Some(Confirm(false)),
            device(0),
            Some(Input("  ")),
            Some(Input("")),
            Some(Input("")),
            Some(Confirm(true)),
        ]);
        assert_eq!(answer(&mut script, &repo), strings(&["abc"]));
        assert_eq!(
            script.rejected,
            strings(&[
                "ab-c: invalid character: '-' (allowed: A-Z a-z 0-9 + /)",
                "  : empty"
            ])
        );
        assert_eq!(
            script.notes,
            strings(&["The same with arguments: meikey abc"])
        );
        assert_eq!(AskError::Cancelled.to_string(), "Cancelled");
    }
}
