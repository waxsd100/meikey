# meikey

> 銘（めい）＋ key — SSH 鍵に自分の銘を刻むツール

公開鍵に好きな文字列を含む ed25519 SSH 鍵（vanity key）を探す CLI です。

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAiytFbrDh3MEGkK6xS5oe22CBkwAkOiHqOUBG+RECKz wakokara
                                                               ^^^^
```

## 特長

- 鍵はメモリ上で生成し、全コアで並列に探索します。`ssh-keygen` を繰り返し呼ぶ方式より桁違いに速く回ります
- 公開鍵の計算を探索専用に実装していて、ed25519-dalek をそのまま使うより約3.5倍速く回ります（[速度](#速度)）
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
| `--constant-time` | 定数時間の実装を使う（約3倍遅い。[安全性](#安全性)を参照） |

### 入れられる文字列

- 使える文字は `A-Z a-z 0-9 + /`、最大42文字
- 入るのは `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI` の直後から
- 大文字小文字を区別しない場合、`WaKoKaRa` のように混ざった形で出ます
- 表示される残り時間は確率の期待値です。半分で出ることも、倍かかることもあります

## 速度

1鍵あたりの内訳（2コアのクラウド環境、1スレッド）:

| 工程 | ed25519-dalek | meikey |
|---|---|---|
| SHA-512 | 約0.3µs | 約0.3µs |
| 基点倍算 | 約13µs（定数時間、4ビット窓） | 約5µs |
| 圧縮（逆元） | 約4µs（1鍵ごと） | 約0.2µs |

- **8ビット窓の櫛形テーブル**：`k × 256^i × B` を起動時に作っておき、倍加なしの32回の加算で倍算します。テーブルは約500KBで L2 キャッシュに収まります
- **一括逆元**：圧縮に必要な 1/Z を256鍵ぶんまとめて1回の逆元で求めます

当たりの鍵は書き出す前に ed25519-dalek で計算し直して確かめるので、高速化にバグがあっても誤った鍵は書きません。
実装ごとの1鍵あたりの時間は `cargo make breakdown` で測れます。

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
| `cargo make run …` | 最適化ビルドで実行 |
| `cargo make build` | 最適化ビルド（`target/release/meikey`） |
| `cargo make breakdown` | 1鍵あたりの計算時間を実装ごとに比較 |

`run`・`build`・`breakdown` は `-C target-cpu=native` でビルドします。配布用のバイナリは `cargo build --release` で作ってください。
RustRover を使う場合は、Rustfmt の **Use rustfmt instead of built-in formatter** をオンにしてください（`rustfmt.toml` の書式に揃えるため）。

### 構成

```
src/
├── main.rs      CLI
├── key.rs       シード → ed25519 鍵、OpenSSH 形式への書き出し
├── pattern.rs   文字列の検証・判定・期待試行回数
├── search.rs    並列探索
├── derive.rs    シード → 公開鍵を256個ずつ計算
├── comb.rs      櫛形テーブルによる固定基点倍算
├── field.rs     GF(2^255-19) の体演算
└── format.rs    件数・時間の表示
tests/
├── it/          IT（CLI の振る舞い、作った鍵が OpenSSH で使えるか）
├── data/        RFC 8032 のテストベクタ
└── support/     テスト用の独立実装（RFC 8032 の参照実装、SSH 形式）
```

### テスト

- **ユニットテスト**：体演算・倍算・SHA-512・鍵の書き出しを、多倍長整数・RFC 8032 の参照実装・curve25519-dalek・ed25519-dalek と proptest で突き合わせます。RFC 8032 Section 7.1 のテストベクタもすべて通します
- **IT**：meikey を実際に起動して鍵を作り、ssh-key・`ssh-keygen`・一時的に立てた sshd で、読み込み・署名・検証・ログインができることを確かめます。必要なツールがない環境ではその部分だけスキップします
- 正解には、本体とは別に仕様書（RFC 8032 / 8709 / 4251 / 4648、PROTOCOL.key / sshsig）から書いた独立実装（`tests/support/`）を使います
