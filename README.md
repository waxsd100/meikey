# meikey — SSH vanity key generator

English | [日本語](README.ja.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waxsd100/meikey/blob/master/colab/meikey.ipynb)
[![Benchmark: 955M keys/s](https://img.shields.io/badge/benchmark-955M%20keys%2Fs-brightgreen.svg)](docs/benchmark.md)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust 1.99+](https://img.shields.io/badge/rust-1.99%2B-orange.svg?logo=rust)](https://www.rust-lang.org)
[![CUDA sm_75+](https://img.shields.io/badge/CUDA-sm__75%2B-76B900.svg?logo=nvidia)](#gpu)
[![Last commit](https://img.shields.io/github/last-commit/waxsd100/meikey.svg)](https://github.com/waxsd100/meikey/commits/master)
[![技術者倫理|遵守済み](https://gijutsusharin.li/badge.svg)](https://gijutsusharin.li)

**Makes SSH keys (ed25519) whose public key contains the text you choose.** It searches by brute force on the CPU and NVIDIA GPUs, and writes keys in the same OpenSSH format as `ssh-keygen -t ed25519`. The name is 銘 (mei, "inscription") + key.

**About 1 billion keys per second** (955.4M keys/s with the GPU and CPU of an RTX PRO 6000 and an AMD EPYC 9B45; see the [comparison](docs/comparison.md)).

Output when searching for `WAKOKARA` (case-sensitive) on Google Colab (RTX PRO 6000, GPU + CPU):

```
$ meikey WAKOKARA -s --low-security --both -o /content/keys/id_ed25519_WAKOKARA
GPU           NVIDIA RTX PRO 6000 Blackwell Server Edition
Compute       sm_120
SM            188
Memory        102 GB
Grid          752 blocks × 128 threads × 8 keys
CPU           AVX-512 IFMA ✓  AVX-512 ✓  AVX2 ✓
SIMD          AVX-512 IFMA
Multiply      table
Inverse       once per 256 keys
Search        memmem
Threads       47
Text          WAKOKARA (case-sensitive)
Est. tries    about 8042.1B
Elapsed 5m 40s  Tries 183.1B  538.5M keys/s  About 4h 3m left
Tries         185.6B keys
Elapsed       5m 44s
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHzN3Wjtv6JdM+HoExVacyBsO4Bm2OM+mX4ZWAKOKARA
                                                                        ^^^^^^^^
Private key   /content/keys/id_ed25519_WAKOKARA
Public key    /content/keys/id_ed25519_WAKOKARA.pub
Passphrase    none (to set one: ssh-keygen -p -f /content/keys/id_ed25519_WAKOKARA)
```

![Output on Google Colab when the WAKOKARA key was found](docs/images/colab-wakokara.png)

- The keys it writes (`id_ed25519_…` and `.pub`) can be used as they are in `~/.ssh/` and `authorized_keys`
- The text can be anywhere, at the end (`--suffix`), or a regular expression (`--regex`)
- Uses AVX2, AVX-512 and AVX-512 IFMA on the CPU, and CUDA (Turing or later) on the GPU

## Install

- Rust 1.99 or later (`rustup toolchain install`)
- [cargo-make](https://github.com/sagiegurari/cargo-make) (`cargo install cargo-make`)
- For the GPU, an NVIDIA driver R525 or later (the CUDA Toolkit is not needed)

```sh
git clone https://github.com/waxsd100/meikey && cd meikey
cargo make build        # builds target/release/meikey
```

`meikey` in the examples below means `target/release/meikey`. `cargo make run …` builds and runs it with the same arguments.

## Usage

```sh
meikey wakokara -o ~/.ssh/id_ed25519_wakokara      # find a key whose public key contains wakokara and write it
meikey wakokara --bench 10                         # measure the speed and the expected time without searching
meikey wakokara --suffix                           # a key whose public key ends with wakokara
meikey 'wako(kara)?$' --regex                      # search with a regular expression
meikey wakokara --low-security --both              # search on the GPU and the CPU together (see Security)
```

Run it with no arguments to be asked for each setting before it starts.

The private key has no passphrase. Set one before using it:

```sh
ssh-keygen -p -f ~/.ssh/id_ed25519_wakokara
```

### Text

- The characters are `A-Z a-z 0-9 + /` (the base64 characters), up to 42 of them
- The text goes after the fixed part `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI`
- Case-insensitive by default (`-s` makes it case-sensitive), so the result can be mixed case like `WaKoKaRa`

With `--regex`, the regular expression is matched against the 43 characters after the fixed part.

- `^` is right after the fixed part, and `$` is the end
- The first character after the fixed part is always one of `A`-`P`, so a pattern no key can match (such as `^Q`) is an error
- A pattern that matches the empty string (such as `a*`) matches every key, so it cannot be used
- Works on the GPU too

### Options

| Option | Meaning |
|---|---|
| `-s` | Match upper and lower case exactly |
| `--suffix` | Only keys whose public key ends with the text |
| `--regex` | Treat the text as a regular expression |
| `-o PATH` | Where to write the private key (default: `id_ed25519_meikey`). The public key goes to `PATH.pub` |
| `-C TEXT` | Comment for the key |
| `-f` | Overwrite existing files |
| `-j N` | Number of threads (default: all cores) |
| `--bench SECS` | Only measure the speed |
| `--check` | Show the CPU, GPU and methods to use, then exit |
| `--low-security` | Low-security mode (prioritizes speed) |
| `--gpu` / `--both` | Search on the GPU only / on the GPU and the CPU (with `--low-security`). All GPUs are used if there are several |
| `--gpu-width`, `--gpu-batch`, `--gpu-limbs` | GPU settings (without them, the fastest is measured on the GPU before searching) |
| `--simd LEVEL` | Highest SIMD level to use (`avx512-ifma`, `avx512`, `avx2`, `off`) |

`meikey --help` lists every option. Messages are in Japanese if the locale (the LANG, LC_ALL and similar environment variables, or else the OS setting) is Japanese, and in English otherwise.

### GPU

Works on NVIDIA GPUs from Turing (RTX 20 series, T4) on. The GPU computation is not constant-time, so `--gpu` and `--both` need `--low-security`.

## How long it takes

Number of keys to try before an `N`-letter text is found with probability `P`:

$$ C = \frac{b^{N} \ln \bigl(\frac{1}{1-P}\bigr)}{43 - N} $$

`b` is 32 when case-insensitive and 64 when case-sensitive (`-s`). `43 − N` is the number of positions the text can be in (1 with `--suffix`). `--bench` shows the expected time for your text.

Case-insensitive:

| N \ P | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|
| 2 | 17 | 58 | 115 | 173 |
| 3 | 568 | 1.9K | 3.8K | 5.7K |
| 4 | 18.6K | 61.9K | 123.8K | 185.7K |
| 5 | 612.1K | 2.0M | 4.1M | 6.1M |
| 6 | 20.1M | 66.8M | 133.6M | 200.5M |
| 7 | 661.6M | 2.2B | 4.4B | 6.6B |
| 8 | 21.8B | 72.3B | 144.7B | 217.0B |
| 9 | 717.3B | 2.4T | 4.8T | 7.1T |
| 10 | 23.6T | 78.6T | 157.1T | 235.7T |
| 11 | 780.4T | 2.6e15 | 5.2e15 | 7.8e15 |
| 12 | 2.6e16 | 8.6e16 | 1.7e17 | 2.6e17 |

Case-sensitive (`-s`):

| N \ P | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|
| 2 | 69 | 230 | 460 | 690 |
| 3 | 4.5K | 15.1K | 30.2K | 45.3K |
| 4 | 298.2K | 990.5K | 2.0M | 3.0M |
| 5 | 19.6M | 65.1M | 130.1M | 195.2M |
| 6 | 1.3B | 4.3B | 8.6B | 12.8B |
| 7 | 84.7B | 281.3B | 562.6B | 843.9B |
| 8 | 5.6T | 18.5T | 37.0T | 55.6T |
| 9 | 367.3T | 1.2e15 | 2.4e15 | 3.7e15 |
| 10 | 2.4e16 | 8.0e16 | 1.6e17 | 2.4e17 |
| 11 | 1.6e18 | 5.3e18 | 1.1e19 | 1.6e19 |
| 12 | 1.1e20 | 3.5e20 | 7.0e20 | 1.1e21 |

Average time ([speed measured](docs/comparison.md#results) on a machine with an RTX PRO 6000 and an AMD EPYC 9B45; CPU in the default mode, GPU + CPU in low-security mode):

| Length | Case-insensitive: est. tries | CPU (51.3M keys/s) | GPU + CPU (955.4M keys/s) | Case-sensitive: est. tries | CPU | GPU + CPU |
|---|---|---|---|---|---|---|
| 4 | 26.9K | < 1 s | < 1 s | 430.2K | < 1 s | < 1 s |
| 5 | 883.0K | < 1 s | < 1 s | 28.3M | < 1 s | < 1 s |
| 6 | 29.0M | < 1 s | < 1 s | 1.9B | 36 s | 2 s |
| 7 | 954.4M | 19 s | < 1 s | 122.2B | 40 min | 2 min |
| 8 | 31.4B | 10 min | 33 s | 8.0T | 2 days | 2.3 h |
| 9 | 1.0T | 5.6 h | 18 min | 529.8T | 120 days | 6 days |
| 10 | 34.1T | 8 days | 9.9 h | 3.5e16 | 22 years | 1 year |
| 11 | 1.1e15 | 254 days | 14 days | 2.3e18 | 1,425 years | 77 years |
| 12 | 3.7e16 | 23 years | 1 year | 1.5e20 | 94,162 years | 5,056 years |

Speeds on other machines are in the [benchmark](docs/benchmark.md), and other tools in the [comparison](docs/comparison.md).

## Google Colab

Open it with **Open in Colab** above and run the cells from the top. Choose a GPU in the runtime type, and choose GPU, CPU or GPU + CPU in the form.

The private key is made on the Colab VM. After downloading it, delete it from the VM and set a passphrase on your own machine.

[colab/compare.ipynb](https://colab.research.google.com/github/waxsd100/meikey/blob/master/colab/compare.ipynb) compares the speed with other tools on the same machine ([comparison](docs/comparison.md)).

## Security

| | Default mode | Low-security mode (`--low-security`) |
|---|---|---|
| Multiplication | Constant-time (the time and the memory read do not depend on the private key) | Table-based, not constant-time. Other processes or VMs on the same CPU may infer the private key from cache timing |
| Seeds | ChaCha20 for each key | OS randomness (once per thread on the CPU, once per search on the GPU) plus a counter. If one seed made from the same randomness leaks, the other seeds can be found too |
| GPU | Not available | Available (the GPU computation is not constant-time either) |

In either mode, the administrator of the machine can still read its memory and disk. Keys made on Colab or other clouds are made where the operator can see them.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

## Disclaimer

meikey is a research project to see how fast ed25519 key generation can go. Do not use it to make keys for production.
