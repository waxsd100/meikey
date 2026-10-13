# meikey

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waxsd100/meikey/blob/master/colab/meikey.ipynb)
[![Benchmark: 7.0億 鍵/秒](https://img.shields.io/badge/benchmark-7.0%E5%84%84%20%E9%8D%B5%2F%E7%A7%92-brightgreen.svg)](docs/benchmark.md)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#ライセンス)
[![Rust 1.99+](https://img.shields.io/badge/rust-1.99%2B-orange.svg?logo=rust)](https://www.rust-lang.org)
[![CUDA sm_75+](https://img.shields.io/badge/CUDA-sm__75%2B-76B900.svg?logo=nvidia)](#gpu)
[![Last commit](https://img.shields.io/github/last-commit/waxsd100/meikey.svg)](https://github.com/waxsd100/meikey/commits/master)
[![技術者倫理|遵守済み](https://gijutsusharin.li/badge.svg)](https://gijutsusharin.li)

公開鍵に好きな文字列が入った ed25519 の SSH 鍵（vanity key）を探す CLI。名前は 銘（めい）＋ key。

GPU にも対応していて、Google Colab の RTX PRO 6000 の環境なら GPU だけで **毎秒 7.0億 鍵** を探索できる（[ベンチマーク](docs/benchmark.md)）。

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAiytFbrDh3MEGkK6xS5oe22CBkwAkOiHqOUBG+RECKz wakokara
                                                               ^^^^
```

- 鍵はメモリ上で作り、全コアで並列に探す
- AVX2・AVX-512・AVX-512 IFMA を使い分け、NVIDIA の GPU（CUDA）でも探せる
- 出力は普通の OpenSSH 形式の ed25519 鍵

## 必要なもの

- Rust 1.99 以上（`rustup toolchain install` で入る）
- [cargo-make](https://github.com/sagiegurari/cargo-make)（`cargo install cargo-make`）
- GPU で探すなら NVIDIA のドライバ R525 以降（CUDA Toolkit は要らない）

## 使い方

```sh
cargo make run wakokara --bench 10                                  # 速度と期待時間を測る
cargo make run wakokara -C wakokara -o ~/.ssh/id_ed25519_wakokara   # 鍵を探して書き出す
cargo make run wakokara --low-security --bench 10                   # 低セキュリティモード（速さを優先）
```

引数を付けずに起動すると、することと文字列・保存先・コメントなどを順に聞かれ、選んだり入力したりして始められる。始める前に、同じことをする引数を表示する。

表示は、OS のロケールが日本語なら日本語、それ以外は英語になる。

秘密鍵にはパスフレーズが付いていないので、使う前に付ける。

```sh
ssh-keygen -p -f ~/.ssh/id_ed25519_wakokara
```

### GPU

Turing（RTX 20 シリーズ・T4）以降の NVIDIA の GPU で動く。`--gpu` で GPU だけ、`--both` で GPU と CPU の両方で探す。GPU の計算は定数時間ではないので、`--low-security` と一緒に付ける。

```sh
cargo make run wakokara --low-security --gpu --bench 10
cargo make run wakokara --low-security --both -C wakokara -o ~/.ssh/id_ed25519_wakokara
```

### オプション

| オプション | 意味 |
|---|---|
| `-s`, `--case-sensitive` | 大文字小文字を区別する |
| `-j N` | スレッド数（既定: 全コア） |
| `-o PATH` | 秘密鍵の出力先。公開鍵は `PATH.pub` |
| `-C TEXT` | 鍵のコメント |
| `-f`, `--force` | 既存のファイルを置き換える |
| `--bench SECS` | 探さずに速度だけ測る |
| `--check` | 使う CPU・GPU と計算方法を表示して終わる |
| `--low-security` | 低セキュリティモード（速さを優先する。下の「安全性」を参照） |
| `--gpu` | CPU ではなく GPU で探す（`--low-security` のとき） |
| `--both` | GPU と CPU の両方で探す（`--low-security` のとき） |
| `--gpu-width W` | GPU の表の幅（2〜20。`--gpu`・`--both` のとき）。付けなければ、探す前に GPU で測って速い設定を選ぶ |
| `--gpu-batch N` | GPU で逆元をまとめる鍵の数（1〜64。`--gpu`・`--both` のとき）。付けなければ、探す前に GPU で測って速い設定を選ぶ |
| `--gpu-limbs L` | GPU の体演算のリムの数（10 か 8。`--gpu`・`--both` のとき）。付けなければ、探す前に GPU で測って速い設定を選ぶ |
| `--simd LEVEL` | 使う SIMD の上限（`avx512-ifma`・`avx512`・`avx2`・`off`） |

### 入れられる文字列

- 使える文字は `A-Z a-z 0-9 + /`、最大 42 文字
- 入るのは `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI` の後ろ
- 大文字小文字を区別しないと、`WaKoKaRa` のように混ざった形で出る

## Google Colab

上の **Open in Colab** から開き、セルを上から順に実行する。ランタイムのタイプで GPU を選び、計測と探索のフォームで GPU・CPU・GPU + CPU のどれを使うか選べる。

秘密鍵は Colab の VM 上で作られる。ダウンロードしたら VM から消し、手元でパスフレーズを付ける。

ほかのツール（[ツールの比較](docs/comparison.md)）と同じマシンで速さを比べるノートブックは [colab/compare.ipynb](https://colab.research.google.com/github/waxsd100/meikey/blob/master/colab/compare.ipynb)。

## 安全性

既定では、倍算に定数時間の実装を使い、シードは鍵ごとに ChaCha20 で作る。

`--low-security` を付けると、速さを優先する。

- 倍算に定数時間ではない表の倍算を使う。同じマシン上のほかのプロセスから、キャッシュのタイミングで秘密鍵を推測される余地がある
- シードは、OS の乱数（スレッドごとに 1 回）に通し番号を足したものにする
- `--gpu`・`--both` は、このモードでだけ使える

## ライセンス

[MIT](LICENSE-MIT) または [Apache-2.0](LICENSE-APACHE) のどちらかを選んで使える。

## 免責事項

meikey は ed25519 の鍵生成をどこまで速くできるかを試すための研究用プロジェクト。本番環境で使う鍵の生成には使わないこと。
