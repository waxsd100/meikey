# meikey

> 銘（めい）＋ key — SSH 鍵に自分の銘を刻むツール

公開鍵に好きな文字列を含む ed25519 SSH 鍵（vanity key）を探す CLI です。

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAiytFbrDh3MEGkK6xS5oe22CBkwAkOiHqOUBG+RECKz wakokara
                                                               ^^^^
```

## 特長

- 鍵はメモリ上で生成し、全コアで並列に探索します。`ssh-keygen` を繰り返し呼ぶ方式より桁違いに速く回ります
- 公開鍵の計算を探索専用に実装していて、ed25519-dalek をそのまま使うより約7倍速く回ります（[速度](#速度)）
- NVIDIA の GPU（CUDA）でも探せます。Google Colab 用のノートブックもあります（[Google Colab の GPU で動かす](#google-colab-の-gpu-で動かす)）
- 出力は通常の OpenSSH 形式の ed25519 鍵です。OpenSSH・[ssh-key](https://crates.io/crates/ssh-key)・RFC 8032 の参照実装で読み書き・署名・ログインできることをテストしています

## 必要なもの

- Rust 1.99 以上（`rust-toolchain.toml` で指定しているので `rustup toolchain install` で入ります）
- [cargo-make](https://github.com/sagiegurari/cargo-make)：`cargo install cargo-make`

## 使い方

```sh
# 速度と期待時間を確認（10秒測る）
cargo make run wakokara --bench 10

# 鍵を探して書き出す
cargo make run wakokara -C wakokara -o ~/.ssh/id_ed25519_wakokara
```

見つかると、一致した場所に `^` を付けて表示します。
出力される秘密鍵はパスフレーズなしなので、使う前に設定してください。

```sh
ssh-keygen -p -f ~/.ssh/id_ed25519_wakokara
```

`cargo make build` でビルドした `target/release/meikey` を直接使うこともできます。

### オプション

| オプション | 意味 |
|---|---|
| `-s`, `--case-sensitive` | 大文字小文字を区別する（英字8文字なら256倍遅くなる） |
| `-j N` | スレッド数（既定: 全コア） |
| `-o PATH` | 秘密鍵の出力先（公開鍵は `PATH.pub`。Unix では秘密鍵を 0600 で作る） |
| `-C TEXT` | 鍵のコメント |
| `-f`, `--force` | 既存のファイルを置き換える |
| `--bench SECS` | 探索せず速度だけ測る |
| `--constant-time` | 基点倍算に定数時間の実装を使う（約5倍遅い。[安全性](#安全性)を参照） |
| `--no-batch` | 圧縮の逆元を1鍵ずつ計算する（比較用） |
| `--simd LEVEL` | 使う SIMD の上限（`amx`・`avx512-ifma`・`avx512`・`avx2`・`off`。比較用）。既定は `avx512-ifma`（CPU が対応していなければ下がる）。`amx` は実験的な経路で、feature `amx` を有効にしてビルドしたときだけ指定できます（[下記](#amx実験的)） |
| `--check` | 使える SIMD と、実際に使うものを表示して終わる |
| `--no-memmem` | 文字列を memmem で探さず総当たりで比べる（比較用） |
| `--gpu` | GPU（CUDA）で探す。feature `cuda` を有効にしてビルドしたときだけ指定できる（[下記](#gpucuda)） |

### GPU（CUDA）

NVIDIA の GPU で探すときは、feature `cuda` を有効にしてビルドします。

```sh
cargo build --release --features cuda
target/release/meikey --gpu --bench 10 wakokara
target/release/meikey --gpu wakokara -C wakokara -o ~/.ssh/id_ed25519_wakokara
```

- CUDA のライブラリ（ドライバの libcuda と NVRTC の libnvrtc）は実行時に読み込むので、ビルドに CUDA Toolkit は要りません
  （C++ コンパイラは要ります）。libnvrtc が見つからないときは、`LD_LIBRARY_PATH` に libnvrtc.so のあるディレクトリを足してください
- GPU 向けのプログラム（`src/service/gpu/kernel.cu`）は、起動時にその GPU の世代向けにコンパイルします（10 秒ほど）。
  Turing（T4）から Blackwell（RTX PRO 6000）まで動きます
- 1スレッドが鍵 8 本ずつ、シードの生成（ChaCha20）・SHA-512・櫛形テーブルの倍算・逆元をまとめた圧縮・base64 の判定まで
  GPU で行います。体演算は AVX2 版と同じ 25.5 ビット × 10 リムです
- 当たりは通し番号だけを受け取り、CPU で同じシードを作り直して ed25519-dalek で計算し、文字列が含まれることを確かめてから書き出します

### Google Colab の GPU で動かす

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waxsd100/meikey/blob/master/colab/meikey.ipynb)

[`colab/meikey.ipynb`](colab/meikey.ipynb) を Colab で開き、上のセルから順に実行します（GPU の確認 → Rust を入れる →
ビルド → 速さを測る → 鍵を探す → ダウンロード）。

- ランタイムのタイプで GPU を選んでください。この計算は 32 ビット整数の掛け算が中心なので、整数演算の多い GPU ほど速い見込みです
  （目安: RTX PRO 6000（G4） > H100 > A100 > L4 > T4。どれが使えるかはプランと空き状況によります）
- Google AI Ultra ではプレミアム GPU とバックグラウンド実行が使えるので、長い探索もブラウザを閉じたまま続けられます
- 秘密鍵は Colab の VM の上で作られます。ダウンロードしたら VM から消し、手元でパスフレーズを付けてください

### 入れられる文字列

- 使える文字は `A-Z a-z 0-9 + /`、最大42文字
- 入るのは `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI` の直後から
- 大文字小文字を区別しない場合、`WaKoKaRa` のように混ざった形で出ます
- 表示される残り時間は確率の期待値です。半分で出ることも、倍かかることもあります

## 速度

i7-12700K（20スレッド）で約600万 鍵/秒、1スレッドで約70万 鍵/秒です。1鍵あたりの内訳（1スレッド）:

| 工程 | ed25519-dalek | meikey |
|---|---|---|
| SHA-512 | 約0.16µs | 約0.08µs（4シード同時） |
| 基点倍算 | 約9µs（定数時間、4ビット窓） | 約1.1µs（12ビット窓のテーブル、4鍵同時） |
| 圧縮（逆元） | 約2.7µs（1鍵ごとに逆元） | 約0.05µs（256鍵で逆元1回） |
| 合計 | 約12µs | 約1.6µs（表の読み込みなどを含む） |

- **12ビット窓の櫛形テーブル**：`k × 2^(12i) × B` を起動時に作っておき、倍加なしの22回の加算で倍算します
- **表の1行ずつ処理**：256鍵に表の1行目を足し、次に2行目を…と進め、同時に使う範囲を L2 キャッシュに収めます
- **AVX2**：倍算（25.5ビット×10リムの体演算）と SHA-512 を4鍵ずつ同時に計算します。AVX2 のない CPU では自動で1鍵ずつになります
- **AVX-512**：対応した CPU では8鍵ずつ同時に計算します（[下記](#avx-512)）。i7-12700K は AVX-512 に対応していないので AVX2 を使います
- **AMX**（実験的、既定では無効）：feature `amx` を有効にしてビルドし、`--simd amx` を付けたときだけ、体の乗算を AMX のタイル演算でします（[下記](#amx実験的)）。IFMA より遅いので既定のビルドには入れていません
- **一括逆元**：圧縮に必要な 1/Z を256鍵ぶんまとめて1回の逆元で求めます
- **memmem**：base64 の中の文字列を [memchr](https://crates.io/crates/memchr) の SIMD 検索で探します

### AVX-512

CPU が持っている乗算命令に合わせて、体演算を2通り用意しています。どちらも表引き（`vpgatherqq`）・
負の桁の符号反転（マスクレジスタ）・SHA-512（回転は `vprorq`、Σ・Ch・Maj は `vpternlogq` の1命令）は共通です。

| | 対象の CPU | 体演算 | 1回の乗算 |
|---|---|---|---|
| AVX-512 IFMA | Intel Ice Lake・Sapphire Rapids 以降、AMD Zen 4 以降 | 51ビット×5リム、`vpmadd52luq`/`vpmadd52huq`（52×52→104ビットの積和） | 積和 50 命令 |
| AVX-512F | Skylake-SP・Cascade Lake | 25.5ビット×10リム、`vpmuludq`（AVX2 と同じ式を 512 ビットで） | 乗算 109 命令 |

Xeon（Cascade Lake 2.8GHz、IFMA なし）の1スレッドで測った1鍵あたりの時間:

| | AVX2 | AVX-512F |
|---|---|---|
| 基点倍算 | 約2.8µs | 約2.0µs |
| SHA-512 | 約0.13µs | 約0.05µs |
| 合計 | 約3.0µs | 約2.2µs |

`-C target-cpu=native`（`cargo make run`）でビルドすると、AVX2 の経路も AVX-512VL の 32 本のレジスタや
回転命令を使えるようになるので差は縮み、合計は約2.5µs → 約2.25µs（約1.1倍）です。

Xeon（Emerald Rapids 2.1GHz、IFMA・AMX あり、2コアの仮想マシン）の1スレッドで測った1鍵あたりの時間
（`cargo make breakdown` の「倍算=テーブル・圧縮=まとめて」、SHA-512 と圧縮を含む）:

| SIMD なし | AVX2 | AVX-512F | AVX-512 IFMA | AMX（実験的） |
|---|---|---|---|---|
| 約3.2µs | 約2.9µs | 約1.8µs | 約0.77µs | 約5.8µs |

### AMX（実験的）

AMX（Advanced Matrix Extensions、Sapphire Rapids 以降の Xeon）の `tdpbuud`（u8 × u8 の行列積を i32 に積和）で
体の乗算をする経路です。体の元を 256 進 32 桁のバイト列として、2つの元の畳み込み（63 桁）を `tdpbuud` 1回で
計算し、普通のレジスタで繰り上げて 2^256 ≡ 38 で畳みます。タイルの行列は、元のバイト列を少しずつずらして
読むだけで作れます（詳しくは `src/service/derive/amx.rs`）。

ただし、この計算には AMX が向いていません。

- 行列積は片方の行列を何行ぶんにも使い回すと速いのに、倍算では鍵ごとに掛ける表のエントリが違うので使い回せず、
  タイルの 16 行のうち 4 行しか使えません
- 普通のレジスタとタイルの間の受け渡しがメモリ経由で遅く、乗算1回あたり約 35ns かかります
  （IFMA は 8 鍵ぶんの乗算を 50 命令ほどで終えます）

その結果、上の表のとおり IFMA の約 8 倍遅く、SIMD なしよりも遅いです。AMX のある CPU はすべて IFMA にも対応しているので、
既定のビルドでは無効にしています。試すときは feature `amx` を有効にしてビルドし（`cargo build --release --features amx`）、
`--simd amx` を付けて実行します。Rust 1.99 では AMX の intrinsics と `target_feature` が
nightly 限定なので、命令は `asm!` で書いています。Linux では起動時に arch_prctl でタイルを使う許可を取ります。
Linux 以外では試していないので使いません。

当たりの鍵は書き出す前に ed25519-dalek で計算し直して確かめるので、高速化にバグがあっても誤った鍵は書きません。
組み合わせごとの1鍵あたりの時間は `cargo make breakdown` で測れます。

## 安全性

- シードは OS の乱数で初期化した ChaCha20 で生成します
- 秘密鍵は見つかった1つだけをファイルに書きます
- 既定の倍算は**定数時間ではありません**。テーブルの参照位置が秘密スカラーに依存するので、同じマシン上の悪意あるプロセスがキャッシュのタイミングを観測すれば、理論上は秘密鍵の手がかりを得られます。共有サーバーなど信用できない環境では `--constant-time` を付けてください（curve25519-dalek の定数時間実装を使います）
- 出てきた鍵そのものは通常の ed25519 鍵で、どちらの方式で作っても安全性は同じです

## 開発

| コマンド | 内容 |
|---|---|
| `cargo make` | CI と同じ確認（fmt・clippy・全テスト・doctest） |
| `cargo make test` | 全テストを [cargo-nextest](https://nexte.st/) で実行（引数は `cargo nextest run` に渡る） |
| `cargo make proptest-deep` | proptest の試行回数を100倍にして全テスト |
| `cargo make mutants` | [cargo-mutants](https://mutants.rs/) でミューテーションテスト |
| `cargo make sde` | [Intel SDE](https://www.intel.com/content/www/us/en/download/684897/) で Sapphire Rapids・Ice Lake サーバー・Skylake サーバー・Alder Lake を真似て、SIMD の経路をテスト（環境変数 `SDE` に sde64 / sde.exe のパス） |
| `cargo make run …` | 最適化ビルドで実行 |
| `cargo make build` | 最適化ビルド（`target/release/meikey`） |
| `cargo make breakdown` | 1鍵あたりの計算時間・検索時間を高速化の組み合わせごとに比較 |

`run`・`build`・`breakdown` は `-C target-cpu=native` でビルドします。配布用のバイナリは `cargo build --release` で作ってください。
RustRover を使う場合は、Rustfmt の **Use rustfmt instead of built-in formatter** をオンにしてください（`rustfmt.toml` の書式に揃えるため）。

### 構成

Model・Controller・Service・Repository の4つに分けています。

- **model**：データの型（探す文字列・鍵・計算方法・探索の設定と結果）と、その型だけで完結する処理（検証・変換）。CPU や I/O には依存しません
- **controller**：CLI。引数を読んで service と repository を呼び、結果を表示します。探索（`Searcher`）と保存（`KeyRepository`）は trait 越しに受け取るので、テストでは [mockall](https://crates.io/crates/mockall) のモックに差し替えられます
- **service**：探索の処理（並列探索・文字列の判定・シードから公開鍵の計算）。体演算や SIMD の型は、公開鍵の計算の内部の部品として `service/derive/` に置いています
- **repository**：鍵を OpenSSH 形式のファイルに保存します

```
src/
├── main.rs            起動するだけ（controller::cli を呼ぶ）
├── lib.rs             4つの区画の宣言
├── model.rs           model のモジュール宣言（以下、各ディレクトリと同名の .rs が親モジュール）
├── model/
│   ├── pattern.rs     探す文字列（検証・一致した位置・期待試行回数）
│   ├── key.rs         シード → ed25519 鍵、公開鍵の base64・authorized_keys の1行
│   ├── backend.rs     公開鍵の計算方法（倍算・圧縮・SIMD）
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

- **ユニットテスト**：体演算・倍算・SHA-512・鍵の書き出しを、多倍長整数・RFC 8032 の参照実装・curve25519-dalek・ed25519-dalek と proptest で突き合わせます。RFC 8032 Section 7.1 のテストベクタもすべて通します
- **モックテスト**：controller の配線を、探索と保存を mockall のモック（`MockSearcher`・`MockKeyRepository`）に差し替えて確かめます。鍵を計算せず・ファイルに触れずに、呼ぶ順番（既存ファイルの確認 → 探索 → 保存）、オプションが探索の設定に届くか、エラーのときに探索や保存をしないかを見ます
- **IT**：meikey を実際に起動して鍵を作り、ssh-key・`ssh-keygen`・一時的に立てた sshd で、読み込み・署名・検証・ログインができることを確かめます。必要なツールがない環境ではその部分だけスキップします
- **GPU**：カーネル（`kernel.cu`）を C++ として CPU 向けにもコンパイルし、ChaCha20・SHA-512・体演算・倍算・base64 の判定・探索全体を rand・sha2・curve25519-dalek・ed25519-dalek・`Pattern` と突き合わせます（GPU が無くても回ります）。NVRTC があれば Turing〜Blackwell の各世代向けにコンパイルできることを、GPU があれば実機で計算した 8,000 鍵ほどが ed25519-dalek と一致し、探索で鍵が見つかることを確かめます（`cargo test --features cuda`）
- **SIMD の経路**：CPU が対応していない命令セットのテストは飛ばします。環境変数 `MEIKEY_REQUIRE_SIMD`（例: `avx2,avx512,avx512-ifma,amx`）に書いた命令セットは、飛ばさずに失敗させます。手元の CPU に無い命令セットは `cargo make sde` でエミュレートして確かめます
- 正解には、本体とは別に仕様書（RFC 8032 / 8709 / 4251 / 4648、PROTOCOL.key / sshsig）から書いた独立実装（`tests/support/`）を使います

## ライセンス

[MIT](LICENSE-MIT) または [Apache-2.0](LICENSE-APACHE) のどちらかを選んで使えます。
