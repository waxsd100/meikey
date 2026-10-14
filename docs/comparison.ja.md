# ほかのツールとの比較

[English](comparison.md) | 日本語

ed25519 の SSH 鍵の公開鍵に好きな文字列を入れるツールを、同じマシンで 30 秒ずつ動かして測った（[colab/compare.ipynb](../colab/compare.ipynb)、2026 年 10 月）。

## 結果

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition、CPU: AMD EPYC 9B45（48 スレッド）

| ツール | 使うもの | 探す場所 | 鍵/秒 | メモ |
|---|---|---|---|---|
| **meikey --low-security --both** | **GPU + CPU** | **公開鍵** | **9.6億** | **低セキュリティモード** |
| vanity-key-ssh（GPU） | GPU | 公開鍵の末尾 | 4.0億 | `--benchmark` |
| vanity-keygen | GPU | 公開鍵 | 2.7億 | |
| ed25519brute-cuda | GPU | フィンガープリント | 2.6億 | |
| **meikey** | **CPU** | **公開鍵** | **5130.0万** | **既定（定数時間の倍算）** |
| SSH-Ed25519-VanityGen（GPU） | GPU | 公開鍵 | 4770.0万 | |
| vanity-key-ssh（CPU） | CPU | 公開鍵の末尾 | 688.7万 | `--benchmark` |
| ssh-keygen-ed25519-vanity（Rust） | CPU | 公開鍵 | 278.3万 | `-j 48` |
| vanity-ssh-rs | CPU | 公開鍵の末尾 | 243.9万 | |
| ssh-keygen-ed25519-vanity（C） | CPU | 公開鍵 | 236.5万 | 48 個同時に動かす |
| ed25519brute | CPU | 公開鍵の末尾 | 236.4万 | |
| vanissh | CPU | 公開鍵 | 216.4万 | |
| vanityssh-go | CPU | 公開鍵 | 182.3万 | |
| SSH-Ed25519-VanityGen | CPU | 公開鍵 | 108.1万 | |
| vanity-ssh-keygen --voi | CPU | 公開鍵 | 78.2万 | |
| vanity-ssh-keygen | CPU | 公開鍵 | 58.2万 | |
| VaniSSH | CPU | 公開鍵 | 7.6万 | 4 文字を見つけるまでの速さ |

## ツール

| ツール | 言語 | 使うもの | 文字列の指定 | ライセンス |
|---|---|---|---|---|
| **[meikey](https://github.com/waxsd100/meikey)** | **Rust・CUDA** | **CPU・GPU** | **部分一致・末尾・正規表現** | **MIT OR Apache-2.0** |
| [vanityssh-go](https://github.com/danielewood/vanityssh-go) | Go | CPU | 正規表現（フィンガープリントも） | MIT |
| [vanity-ssh-keygen](https://github.com/DumAdudus/vanity-ssh-keygen) | Go | CPU | 部分一致 | 記載なし |
| [vanity-ssh-rs](https://github.com/mogottsch/vanity-ssh-rs) | Rust | CPU | 末尾・正規表現 | MIT |
| [vanissh](https://crates.io/crates/vanissh) | Rust | CPU | 部分一致 | MPL-2.0 |
| [ssh-keygen-ed25519-vanity](https://github.com/ionathanch/ssh-keygen-ed25519-vanity) | C・Rust | CPU | 部分一致（C）・正規表現（Rust） | MIT-0 |
| [ed25519brute](https://github.com/akiym/ed25519brute) | Go | CPU | 公開鍵の末尾・フィンガープリント | 記載なし |
| [SSH-Ed25519-VanityGen](https://github.com/Aminuxer/SSH-Ed25519-VanityGen) | Python・OpenCL | CPU・GPU | 部分一致 | GPL-3.0 |
| [VaniSSH](https://github.com/scottvr/vanissh) | Python | CPU | 正規表現 | MIT |
| [vanity-key-ssh](https://github.com/irdbl/vanity-key-ssh) | C++・CUDA | CPU・GPU | 末尾 | 記載なし |
| [vanity-keygen](https://github.com/pcarrier/vanity-keygen) | Rust・OpenCL | GPU | 正規表現の一部 | ISC |
| [ed25519brute-cuda](https://github.com/4equest/ed25519brute-cuda) | CUDA | GPU | フィンガープリント | 記載なし |

## 作者が書いている速さ

各ツールの README などにある値（マシンが違うので、互いには比べられない）。

| ツール | 鍵/秒 | マシン |
|---|---|---|
| vanity-key-ssh（GPU） | 約 3.3億 | RTX 4090 |
| vanity-keygen | 約 3億 | RTX 4090 |
| SSH-Ed25519-VanityGen（GPU） | 約 2780万 | H100 NVL |
| vanity-key-ssh（CPU） | 約 90万 | Apple M シリーズ（14 スレッド） |
| SSH-Ed25519-VanityGen（CPU） | 約 64万 | AMD EPYC 9334 × 2 |
| vanity-ssh-keygen | 約 33万 | Apple M2（8 コア） |
