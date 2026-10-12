//! meikey の実行ファイル（中身は [`meikey::controller::cli`]）

fn main() -> std::process::ExitCode {
    meikey::controller::cli::main()
}
