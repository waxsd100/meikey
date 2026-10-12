# meikey

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waxsd100/meikey/blob/master/colab/meikey.ipynb)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#ライセンス)
[![Rust 1.99+](https://img.shields.io/badge/rust-1.99%2B-orange.svg?logo=rust)](https://www.rust-lang.org)
[![CUDA sm_75+](https://img.shields.io/badge/CUDA-sm__75%2B-76B900.svg?logo=nvidia)](#gpucuda)
[![Last commit](https://img.shields.io/github/last-commit/waxsd100/meikey.svg)](https://github.com/waxsd100/meikey/commits/master)
[![技術者倫理|遵守済み](https://gijutsusharin.li/badge.svg)](https://gijutsusharin.li)

公開鍵に好きな文字列が入った ed25519 の SSH 鍵（vanity key）を探す CLI。名前は 銘（めい）＋ key。

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAiytFbrDh3MEGkK6xS5oe22CBkwAkOiHqOUBG+RECKz wakokara
                                                               ^^^^
```

- 鍵はメモリ上で作り、全コアで並列に探す。`ssh-keygen` を繰り返し呼ぶより桁違いに速い
- 公開鍵の計算は探索用に書き直してあり、ed25519-dalek をそのまま使うより約 7 倍速い（[速度](#速度)）
- NVIDIA の GPU（CUDA）でも探せる。Google Colab 用のノートブックもある
- 出力は普通の OpenSSH 形式の ed25519 鍵。OpenSSH・[ssh-key](https://crates.io/crates/ssh-key)・RFC 8032 の参照実装で読み書き・署名・ログインできることをテストしている

## 必要なもの

- Rust 1.99 以上（`rust-toolchain.toml` で指定してあるので `rustup toolchain install` で入る）
- [cargo-make](https://github.com/sagiegurari/cargo-make)（`cargo install cargo-make`）
- GPU で探すなら NVIDIA のドライバ R525 以降。CUDA Toolkit は要らない

## 使い方

```sh
cargo make run wakokara --bench 10                                  # 速度と期待時間を測る
cargo make run wakokara -C wakokara -o ~/.ssh/id_ed25519_wakokara   # 鍵を探して書き出す
```

一致した場所に `^` が付く。秘密鍵にはパスフレーズが付いていないので、使う前に付ける。

```sh
ssh-keygen -p -f ~/.ssh/id_ed25519_wakokara
```

`cargo make build` でできる `target/release/meikey` を直接使ってもよい。

```
$ meikey --check
CPU           AVX-512 IFMA ✗  AVX-512 ✗  AVX2 ✓
SIMD          AVX2
倍算          テーブル
逆元          256 鍵ごとに 1 回
検索          memmem
並列          20 スレッド
```

### オプション

| オプション | 意味 |
|---|---|
| `-s`, `--case-sensitive` | 大文字小文字を区別する（英字 8 文字なら 256 倍遅い） |
| `-j N` | スレッド数（既定: 全コア） |
| `-o PATH` | 秘密鍵の出力先。公開鍵は `PATH.pub`（Unix では秘密鍵を 0600 で作る） |
| `-C TEXT` | 鍵のコメント |
| `-f`, `--force` | 既存のファイルを置き換える |
| `--bench SECS` | 探さずに速度だけ測る |
| `--check` | 使う CPU・GPU と計算方法を表示して終わる |
| `--gpu` | CPU ではなく GPU で探す（[GPU](#gpucuda)） |
| `--both` | GPU と CPU の両方で探す（CPU のスレッド数の既定は 全コア − 1） |
| `--constant-time` | 倍算に定数時間の実装を使う（約 5 倍遅い。[安全性](#安全性)） |
| `--no-batch` | 逆元を 1 鍵ごとに計算する（比較用） |
| `--simd LEVEL` | SIMD の上限。`avx512-ifma`（既定）・`avx512`・`avx2`・`off`。CPU が対応していなければ下がる。`amx` は feature `amx` でビルドしたときだけ（[AMX](#amx実験的)） |
| `--no-memmem` | memmem を使わず 1 文字ずつずらして比べる（比較用） |

### 入れられる文字列

- 使える文字は `A-Z a-z 0-9 + /`、最大 42 文字
- 入るのは `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI` の後ろ
- 大文字小文字を区別しないと、`WaKoKaRa` のように混ざった形で出る
- 残り時間は期待値なので、半分で出ることも倍かかることもある

## GPU（CUDA）

Turing（RTX 20 シリーズ・T4）以降の NVIDIA の GPU で動く。

```sh
cargo make run wakokara --gpu --bench 10
cargo make run wakokara --gpu -C wakokara -o ~/.ssh/id_ed25519_wakokara
cargo make run wakokara --both --bench 10   # GPU と CPU の両方
```

Windows でも同じ（PowerShell なら `-o $HOME\.ssh\id_ed25519_wakokara`）。

- カーネル（`src/service/gpu/kernel.cu`）は Turing 向けの PTX にして埋め込んである。起動時にドライバが GPU に合わせて変換する（初回だけ数秒かかり、結果はドライバがキャッシュする）
- 1 スレッドで鍵 8 本ずつ、シードの生成（ChaCha20）・SHA-512・倍算・逆元・base64 の判定まで GPU で行う。体演算は AVX2 版と同じ 25.5 ビット × 10 リム
- 当たりは通し番号だけを受け取り、CPU 側で同じシードから ed25519-dalek で作り直して確かめてから書き出す
- 画面をつないだ GPU では 1 回の起動を 0.05 秒ほどに抑える。Windows の TDR（約 2 秒でドライバをリセット）に引っかからず、画面も固まらない
- `--both` は GPU と CPU で同時に探し、どちらかが見つけたら両方止める。CPU の 1 本は GPU の操作に空ける。RTX 3080 Ti（1.1億 鍵/秒）と i7-12700K（約 600 万 鍵/秒）なら約 5% 速くなる
- ドライバが古くて PTX を読めないときは、NVRTC があればその GPU 向けにコンパイルし直す
- `kernel.cu` を変えたら `cargo make ptx` で PTX を作り直す
- GPU 対応（feature `cuda`）は既定で入っている。外すなら `--no-default-features` でビルドする（cudarc と約 1MB の PTX が入らない）

## Google Colab

上の **Open in Colab** から開き、セルを上から順に実行する（GPU・CPU・メモリなどの確認 → Rust の導入 → ビルド → 計測 → 探索 → ダウンロード → 後片付け）。

- ランタイムのタイプで GPU を選ぶ。整数演算の多い GPU ほど速いはずで、目安は RTX PRO 6000（G4）> H100 > A100 > L4 > T4。どれが使えるかはプランと空き状況しだい
- 計測と探索のフォームで、GPU・CPU・GPU + CPU のどれを使うか選べる（既定は計測が全部、探索が GPU + CPU）。CPU は AVX-512 IFMA があれば自動で使う
- Google AI Ultra ならバックグラウンド実行が使えるので、ブラウザを閉じても探索は続く
- 秘密鍵は Colab の VM 上で作られる。ダウンロードしたら VM から消し、手元でパスフレーズを付ける

## 速度

i7-12700K（20 スレッド）で約 600 万 鍵/秒、1 スレッドで約 70 万 鍵/秒。GPU は RTX 3080 Ti で約 1.1億 鍵/秒（[GPU](#gpucuda)）。1 鍵あたりの内訳（1 スレッド）:

| 工程 | ed25519-dalek | meikey |
|---|---|---|
| SHA-512 | 約 0.16µs | 約 0.08µs（4 シード同時） |
| 基点倍算 | 約 9µs（定数時間、4 ビット窓） | 約 1.1µs（12 ビット窓のテーブル、4 鍵同時） |
| 逆元 | 約 2.7µs（1 鍵ごと） | 約 0.05µs（256 鍵で 1 回） |
| 合計 | 約 12µs | 約 1.6µs（表の読み込みなどを含む） |

- 12 ビット窓の櫛形テーブル: `k × 2^(12i) × B` を起動時に作っておき、倍加なし 22 回の加算で倍算する
- 表は 1 行ずつ: 256 鍵に表の 1 行目を足し、次に 2 行目…と進めて、同時に触る範囲を L2 キャッシュに収める
- AVX2: 体演算（25.5 ビット × 10 リム）と SHA-512 を 4 鍵ずつ同時に計算する。AVX2 の無い CPU では 1 鍵ずつ
- AVX-512: 8 鍵ずつ（[下記](#avx-512)）。i7-12700K は非対応なので AVX2 を使う
- 一括逆元: 圧縮に要る 1/Z を 256 鍵まとめて逆元 1 回で求める
- memmem: base64 の中を [memchr](https://crates.io/crates/memchr) の SIMD 検索で探す

組み合わせごとの 1 鍵あたりの時間は `cargo make breakdown` で測れる。どの経路でも、当たりは書き出す前に ed25519-dalek で計算し直して確かめる。

### AVX-512

乗算命令に合わせて体演算を 2 通り用意している。表引き（`vpgatherqq`）、負の桁の符号反転（マスクレジスタ）、SHA-512（回転は `vprorq`、Σ・Ch・Maj は `vpternlogq`）は共通。

| | 対象の CPU | 体演算 | 1 回の乗算 |
|---|---|---|---|
| AVX-512 IFMA | Intel Ice Lake・Sapphire Rapids 以降、AMD Zen 4 以降 | 51 ビット × 5 リム、`vpmadd52luq`/`vpmadd52huq`（52×52→104 ビットの積和） | 積和 50 命令 |
| AVX-512F | Skylake-SP・Cascade Lake | 25.5 ビット × 10 リム、`vpmuludq`（AVX2 と同じ式を 512 ビットで） | 乗算 109 命令 |

Xeon（Cascade Lake 2.8GHz、IFMA なし）の 1 スレッドで測った 1 鍵あたりの時間:

| | AVX2 | AVX-512F |
|---|---|---|
| 基点倍算 | 約 2.8µs | 約 2.0µs |
| SHA-512 | 約 0.13µs | 約 0.05µs |
| 合計 | 約 3.0µs | 約 2.2µs |

`-C target-cpu=native`（`cargo make run`）でビルドすると AVX2 版も AVX-512VL のレジスタと回転命令を使うので、差は縮む（約 2.5µs → 約 2.25µs）。

Xeon（Emerald Rapids 2.1GHz、IFMA・AMX あり、2 コアの VM）の 1 スレッド、SHA-512 と圧縮を含む 1 鍵あたりの時間:

| SIMD なし | AVX2 | AVX-512F | AVX-512 IFMA | AMX（実験的） |
|---|---|---|---|---|
| 約 3.2µs | 約 2.9µs | 約 1.8µs | 約 0.77µs | 約 5.8µs |

### AMX（実験的）

AMX（Sapphire Rapids 以降の Xeon）の `tdpbuud`（u8 × u8 の行列積を i32 に積和）で体の乗算をする経路。遅いので既定のビルドには入れていない。試すなら `cargo build --release --features amx` でビルドして `--simd amx` を付ける。

- 体の元を 256 進 32 桁として、2 つの元の畳み込み（63 桁）を `tdpbuud` 1 回で計算し、繰り上げて 2^256 ≡ 38 で畳む。タイルの行列は元のバイト列をずらして読むだけで作る（`src/service/derive/amx.rs`）
- 鍵ごとに掛ける表の値が違うので行列の片方を使い回せず、タイル 16 行のうち 4 行しか使えない。タイルとレジスタの受け渡しもメモリ経由で遅く、乗算 1 回に約 35ns かかる
- 結果は IFMA の約 8 倍遅く、SIMD なしより遅い。AMX のある CPU はどれも IFMA を持っているので、使う理由はない
- Rust 1.99 では AMX の intrinsics と `target_feature` が nightly 限定なので `asm!` で書いている。Linux のみ（起動時に arch_prctl でタイルの使用許可を取る）

## 安全性

- シードは OS の乱数で初期化した ChaCha20 で作る（GPU も同じ）
- ファイルに書く秘密鍵は見つかった 1 つだけ
- 既定の倍算は**定数時間ではない**。表の参照位置が秘密スカラーで変わるので、同じマシンの別のプロセスがキャッシュのタイミングを観測できれば、理論上は秘密鍵の手がかりになる。共有サーバーなど信用できない環境では `--constant-time` を付ける（curve25519-dalek の定数時間実装を使う）
- できた鍵そのものは普通の ed25519 鍵で、どの方法で作っても安全性は変わらない

## 開発

| コマンド | 内容 |
|---|---|
| `cargo make` | CI と同じ確認（fmt・clippy・全テスト・doctest） |
| `cargo make test` | 全テストを [cargo-nextest](https://nexte.st/) で実行（引数は `cargo nextest run` に渡る） |
| `cargo make proptest-deep` | proptest の試行回数を 100 倍にして全テスト |
| `cargo make mutants` | [cargo-mutants](https://mutants.rs/) でミューテーションテスト |
| `cargo make sde` | [Intel SDE](https://www.intel.com/content/www/us/en/download/684897/) で Sapphire Rapids・Ice Lake サーバー・Skylake サーバー・Alder Lake を真似て SIMD の経路をテスト（環境変数 `SDE` に sde64 / sde.exe のパス） |
| `cargo make run …` | 最適化ビルドで実行（引数はそのまま meikey に渡る） |
| `cargo make build` | 最適化ビルド（`target/release/meikey`） |
| `cargo make breakdown` | 1 鍵あたりの計算時間・検索時間を高速化の組み合わせごとに比べる |
| `cargo make ptx` | `kernel.cu` から埋め込む PTX を作る（NVRTC が要る。`Makefile.toml` を参照） |

`run`・`build`・`breakdown` は `-C target-cpu=native` でビルドする。配布用のバイナリは `cargo build --release` で作る。
RustRover では Rustfmt の **Use rustfmt instead of built-in formatter** をオンにする（`rustfmt.toml` の書式に揃えるため）。

### 構成

Model・Controller・Service・Repository の 4 つに分けている。

- model: データの型（探す文字列・鍵・計算方法・探索の設定と結果）と、その型だけで完結する検証・変換。CPU や I/O に依存しない
- controller: CLI。引数を読んで service と repository を呼び、結果を表示する。探索（`Searcher`）と保存（`KeyRepository`）は trait 越しに受け取るので、テストでは [mockall](https://crates.io/crates/mockall) のモックに差し替えられる
- service: 並列探索・文字列の判定・シードから公開鍵の計算・GPU。体演算や SIMD の部品は `service/derive/` の中
- repository: 鍵を OpenSSH 形式のファイルに保存する

```
src/
├── main.rs            起動するだけ（controller::cli を呼ぶ）
├── lib.rs             4つの区画の宣言
├── model.rs           model のモジュール宣言（以下、各ディレクトリと同名の .rs が親モジュール）
├── model/
│   ├── pattern.rs     探す文字列（検証・一致した位置・期待試行回数）
│   ├── key.rs         シード → ed25519 鍵、公開鍵の base64・authorized_keys の1行
│   ├── backend.rs     公開鍵の計算方法（倍算・逆元・SIMD）
│   └── search.rs      探索の設定と結果
├── controller.rs
├── controller/
│   ├── cli.rs         引数を読んで探索・保存を呼び、進捗と結果を表示する
│   └── format.rs      件数・時間の表示
├── service.rs
├── service/
│   ├── search.rs      全コアで鍵を作って当たりを探す（Searcher trait と本番の Parallel）
│   ├── matcher.rs     探索ループでの文字列の判定（memmem・総当たり）
│   ├── derive.rs      シード → 公開鍵を256個ずつ計算（計算方法の切り替え・CPU の検出・表の準備）
│   ├── derive/
│   │   ├── field.rs     GF(2^255-19) の体演算
│   │   ├── comb.rs      櫛形テーブルによる固定基点倍算
│   │   ├── amx.rs       AMX のタイル演算で体の乗算をする倍算（実験的）
│   │   ├── avx2.rs      AVX2 で4鍵同時の倍算
│   │   ├── avx512.rs    AVX-512 で8鍵同時の倍算（表・表引き・点の演算）
│   │   ├── avx512/
│   │   │   ├── ifma.rs    IFMA の体演算（51ビット×5リム）
│   │   │   └── f.rs       IFMA の無い CPU の体演算（25.5ビット×10リム）
│   │   ├── sha512x4.rs  AVX2 で4シード同時の SHA-512
│   │   └── sha512x8.rs  AVX-512 で8シード同時の SHA-512
│   ├── gpu.rs         GPU（CUDA）で探す（feature cuda。カーネルのコンパイル・起動・当たりの確認）
│   └── gpu/
│       ├── kernel.cu    GPU のカーネル（シードの生成から文字列の判定まで）
│       ├── kernel.ptx   kernel.cu を Turing 向けにした PTX（埋め込む。cargo make ptx で作る）
│       └── host.cpp     テスト用に kernel.cu を CPU 向けにコンパイルする（build.rs）
├── repository.rs
└── repository/
    └── key_file.rs    秘密鍵（openssh-key-v1、0600）と .pub の保存（KeyRepository trait と本番の KeyFiles）
tests/
├── it/          IT（CLI の振る舞い、作った鍵が OpenSSH で使えるか）
├── data/        RFC 8032 のテストベクタ
└── support/     テスト用の独立実装（RFC 8032 の参照実装、SSH 形式）と補助
colab/
└── meikey.ipynb Google Colab の GPU で動かすノートブック
```

### テスト

- ユニットテスト: 体演算・倍算・SHA-512・鍵の書き出しを、多倍長整数・RFC 8032 の参照実装・curve25519-dalek・ed25519-dalek と proptest で突き合わせる。RFC 8032 Section 7.1 のテストベクタも全部通す
- モックテスト: controller を、探索と保存を mockall のモックに差し替えて確かめる。呼ぶ順番（既存ファイルの確認 → 探索 → 保存）、オプションが探索の設定に届くか、エラーのときに探索や保存をしないか
- IT: meikey を実際に起動して鍵を作り、ssh-key・`ssh-keygen`・一時的に立てた sshd で読み込み・署名・検証・ログインができるか確かめる。ツールが無い環境ではその部分だけ飛ばす
- GPU: `kernel.cu` を C++ として CPU 向けにもコンパイルし、ChaCha20・SHA-512・体演算・倍算・base64 の判定・探索全体を rand・sha2・curve25519-dalek・ed25519-dalek・`Pattern` と突き合わせる（GPU が無くても回る）。埋め込んだ PTX が今の `kernel.cu` から作ったものかも確かめる。GPU があれば、実機で計算した約 8,000 鍵が ed25519-dalek と一致し、探索で鍵が見つかるかも見る（`cargo test`）
- SIMD の経路: CPU が対応していない命令セットのテストは飛ばす。環境変数 `MEIKEY_REQUIRE_SIMD`（例: `avx2,avx512,avx512-ifma`）に書いたものは飛ばさずに失敗させる。手元に無い命令セットは `cargo make sde` で確かめる
- 正解には、本体とは別に仕様書（RFC 8032 / 8709 / 4251 / 4648、PROTOCOL.key / sshsig）から書いた独立実装（`tests/support/`）を使う

## ライセンス

[MIT](LICENSE-MIT) または [Apache-2.0](LICENSE-APACHE) のどちらかを選んで使える。
