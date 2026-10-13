# ほかのツールとの比較

ed25519 の SSH 公開鍵（か、そのフィンガープリント）に好きな文字列を入れるツールを集めて、meikey と並べた。内容は 2026 年 10 月に各リポジトリの README とソースを見て書いた。

[colab/compare.ipynb](../colab/compare.ipynb) を使うと、ここに挙げたツールを Google Colab でまとめてビルドし、同じマシンで速さを測れる。

## ツール

| ツール | 言語 | 使うもの | 文字列を探す場所 | 文字列の指定 | 大小文字 | ライセンス | 最終コミット |
|---|---|---|---|---|---|---|---|
| [meikey](https://github.com/waxsd100/meikey) | Rust（GPU は CUDA） | CPU・GPU | 公開鍵 | 部分一致 | 区別しない（`-s` で区別） | MIT OR Apache-2.0 | — |
| [vanityssh-go](https://github.com/danielewood/vanityssh-go) | Go | CPU | 公開鍵・フィンガープリント（`-f`） | 正規表現 | 区別する（`(?i)` で区別しない） | MIT | 2026-03-12 |
| [vanity-ssh-keygen](https://github.com/DumAdudus/vanity-ssh-keygen) | Go | CPU | 公開鍵 | 部分一致 | 区別しない | 記載なし | 2026-10-12 |
| [vanity-ssh-rs](https://github.com/mogottsch/vanity-ssh-rs) | Rust | CPU | 公開鍵 | 末尾。`/…/` で正規表現 | 区別する（正規表現の `(?i)` で区別しない） | MIT | 2025-11-02 |
| [vanissh](https://crates.io/crates/vanissh)（[ソース](https://zff.dev/ava/vanissh)） | Rust | CPU | 公開鍵 | 部分一致 | 区別しない（`-c` で区別） | MPL-2.0 | 2025-10-26 |
| [ssh-keygen-ed25519-vanity](https://github.com/ionathanch/ssh-keygen-ed25519-vanity) | C・Rust・シェル | CPU | 公開鍵 | C 版は部分一致、Rust 版は正規表現 | Rust 版は `-i` で区別しない | MIT-0 | 2025-08-12 |
| [ed25519brute](https://github.com/akiym/ed25519brute) | Go | CPU | 公開鍵の末尾・フィンガープリントの先頭と末尾 | 文字列 | 区別する | 記載なし | 2024-03-26 |
| [SSH-Ed25519-VanityGen](https://github.com/Aminuxer/SSH-Ed25519-VanityGen) | Python（GPU は OpenCL） | CPU・GPU | 公開鍵 | 部分一致。ファイルで複数 | `-i` で区別しない | GPL-3.0 | 2026-09-17 |
| [VaniSSH](https://github.com/scottvr/vanissh) | Python | CPU | 公開鍵 | 正規表現（どこでも・先頭・末尾）。回文も探せる | 区別しない（`-ca` などで区別） | MIT | 2025-07-28 |
| [vanity-key-ssh](https://github.com/irdbl/vanity-key-ssh) | C++・CUDA | CPU・GPU | 公開鍵の末尾 | 文字列（複数可） | `--ci` で区別しない | 記載なし | 2026-09-12 |
| [vanity-keygen](https://github.com/pcarrier/vanity-keygen) | Rust・OpenCL | GPU（OpenCL。pocl を使えば CPU でも動く） | 公開鍵 | 正規表現の一部（文字・文字クラス・`^`・`$`） | 文字クラスで指定する | ISC | 2026-07-18 |
| [ed25519brute-cuda](https://github.com/4equest/ed25519brute-cuda) | CUDA | GPU | フィンガープリントの先頭・末尾 | 文字列 | 区別する | 記載なし | 2026-01-03 |

## 鍵の出し方

| ツール | 出力 |
|---|---|
| meikey | OpenSSH 形式の秘密鍵と `.pub` のファイル（`-o` で場所を指定） |
| vanityssh-go | 今のディレクトリに `id_ed25519` と `id_ed25519.pub`。パイプのときは秘密鍵を標準出力に |
| vanity-ssh-keygen | PEM のファイルか JSON のファイル（`-o`）。ED25519 のほかに RSA-2048・RSA-4096 も作れる |
| vanity-ssh-rs | `out/` に保存する |
| vanissh | 実行したディレクトリの `found-keys/` に保存する |
| ssh-keygen-ed25519-vanity | 標準出力。`-o` でファイル |
| ed25519brute | 今のディレクトリに `out` と `out.pub` |
| SSH-Ed25519-VanityGen | 標準出力。`-o` でファイル |
| VaniSSH | 今のディレクトリにファイル。`-l` で結果を JSON に |
| vanity-key-ssh | シードを表示する。`tools/keytool.py privkey` で OpenSSH 形式の秘密鍵にする |
| vanity-keygen | 公開鍵と秘密鍵を標準出力に |
| ed25519brute-cuda | `found_key.txt`・`id_ed25519`・`id_ed25519.pub` |

ed25519brute と ed25519brute-cuda の README には、作った鍵を実際の認証に使わないよう書かれている。

## README などに書かれている速さ

各ツールの作者が自分のマシンで測った値。マシンが違うので、この表どうしでは比べられない。

| ツール | 速さ | マシン | 出典 |
|---|---|---|---|
| vanity-ssh-keygen | 約 33万 鍵/秒 | Apple M2（8 コア） | README |
| vanissh | 1 コアあたり 6万〜17万 鍵/秒 | — | README |
| SSH-Ed25519-VanityGen（CPU） | 約 64万 鍵/秒 | AMD EPYC 9334 × 2 | README |
| SSH-Ed25519-VanityGen（GPU） | 約 2780万 鍵/秒 | H100 NVL | README |
| vanity-key-ssh（CPU） | 約 90万 鍵/秒 | Apple M シリーズ（14 スレッド） | README |
| vanity-key-ssh（GPU） | 約 3.3億 鍵/秒 | RTX 4090 | README |
| vanity-keygen | 約 3億 鍵/秒 | RTX 4090 | [作者の記事](https://nothing.pcarrier.com/posts/vanity-keygen-opencl) |
| ed25519brute-cuda | 44.34M 鍵/秒（実行例） | — | README |

meikey の速さは [ベンチマーク](benchmark.md) にある。

## 同じマシンで測る

[colab/compare.ipynb](../colab/compare.ipynb) は、上のツールをビルドして、1つずつ同じ秒数だけ動かして速さを測る。測る処理は [colab/compare.py](../colab/compare.py) にある。

- ビルド
  - Rust のツールは `-C target-cpu=native` を付けてビルドする。Go のツールは最新の Go でビルドする
  - vanity-ssh-rs と vanissh は crates.io から、ほかは GitHub の最新のコミットから入れる
  - vanity-ssh-keygen は Go の内部の関数を `go:linkname` で呼んでいるので、`-ldflags=-checklinkname=0` を付けてビルドする
  - ed25519brute-cuda は README では Windows（MSVC と CMake）でビルドするので、Colab では `nvcc` で直接ビルドする
  - vanity-key-ssh の GPU 版は `-arch=native` でビルドする
- オプションは、スレッド数などを含めて各ツールの既定のまま。スレッド数の既定が 1 の ssh-keygen-ed25519-vanity の Rust 版だけ、`-j` に CPU のスレッド数を渡す。1 スレッドで動く C 版は、CPU のスレッド数だけ同時に動かす
- 速さの読み方
  - 速さを表示するツールは、見つからない 10 文字の文字列で決めた秒数だけ動かし、最後に表示した速さを読む
  - VaniSSH は見つけたときだけ速さを表示するので、4 文字の文字列を見つけさせて、そのとき表示した速さを読む
  - 速さを表示しないツール（vanissh・ssh-keygen-ed25519-vanity・ed25519brute）は、短い文字列で決めた秒数のあいだ何度も探させ、「見つかった回数 ÷（その文字列が当たる確率 × かかった時間）」で速さを出す。起動にかかる時間は、すぐ見つかる 1 文字で測って差し引く。何文字で何回見つかったかは、結果の表の「測り方」に出る
