# Comparison with other tools

English | [日本語](comparison.ja.md)

Tools that put the text you choose into the public key of an ed25519 SSH key, each run for 30 seconds on the same machine ([colab/compare.ipynb](../colab/compare.ipynb), October 2026).

## Results

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, CPU: AMD EPYC 9B45 (48 threads)

| Tool | Uses | Where | Keys/s | Notes |
|---|---|---|---|---|
| **meikey --low-security --both** | **GPU + CPU** | **Public key** | **955.4M** | **Low-security mode** |
| vanity-key-ssh (GPU) | GPU | End of the public key | 400.0M | `--benchmark` |
| vanity-keygen | GPU | Public key | 270.0M | |
| ed25519brute-cuda | GPU | Fingerprint | 263.7M | |
| **meikey** | **CPU** | **Public key** | **51.3M** | **Default (constant-time multiplication)** |
| SSH-Ed25519-VanityGen (GPU) | GPU | Public key | 47.7M | |
| vanity-key-ssh (CPU) | CPU | End of the public key | 6.9M | `--benchmark` |
| ssh-keygen-ed25519-vanity (Rust) | CPU | Public key | 2.8M | `-j 48` |
| vanity-ssh-rs | CPU | End of the public key | 2.4M | |
| ssh-keygen-ed25519-vanity (C) | CPU | Public key | 2.4M | 48 copies at once |
| ed25519brute | CPU | End of the public key | 2.4M | |
| vanissh | CPU | Public key | 2.2M | |
| vanityssh-go | CPU | Public key | 1.8M | |
| SSH-Ed25519-VanityGen | CPU | Public key | 1.1M | |
| vanity-ssh-keygen --voi | CPU | Public key | 781.7K | |
| vanity-ssh-keygen | CPU | Public key | 581.9K | |
| VaniSSH | CPU | Public key | 75.8K | Speed until it found 4 characters |

## Tools

| Tool | Language | Uses | Text | License |
|---|---|---|---|---|
| **[meikey](https://github.com/waxsd100/meikey)** | **Rust, CUDA** | **CPU, GPU** | **Anywhere, at the end, regular expression** | **MIT OR Apache-2.0** |
| [vanityssh-go](https://github.com/danielewood/vanityssh-go) | Go | CPU | Regular expression (also the fingerprint) | MIT |
| [vanity-ssh-keygen](https://github.com/DumAdudus/vanity-ssh-keygen) | Go | CPU | Anywhere | Not stated |
| [vanity-ssh-rs](https://github.com/mogottsch/vanity-ssh-rs) | Rust | CPU | At the end, regular expression | MIT |
| [vanissh](https://crates.io/crates/vanissh) | Rust | CPU | Anywhere | MPL-2.0 |
| [ssh-keygen-ed25519-vanity](https://github.com/ionathanch/ssh-keygen-ed25519-vanity) | C, Rust | CPU | Anywhere (C), regular expression (Rust) | MIT-0 |
| [ed25519brute](https://github.com/akiym/ed25519brute) | Go | CPU | End of the public key, fingerprint | Not stated |
| [SSH-Ed25519-VanityGen](https://github.com/Aminuxer/SSH-Ed25519-VanityGen) | Python, OpenCL | CPU, GPU | Anywhere | GPL-3.0 |
| [VaniSSH](https://github.com/scottvr/vanissh) | Python | CPU | Regular expression | MIT |
| [vanity-key-ssh](https://github.com/irdbl/vanity-key-ssh) | C++, CUDA | CPU, GPU | At the end | Not stated |
| [vanity-keygen](https://github.com/pcarrier/vanity-keygen) | Rust, OpenCL | GPU | Part of regular expressions | ISC |
| [ed25519brute-cuda](https://github.com/4equest/ed25519brute-cuda) | CUDA | GPU | Fingerprint | Not stated |

## Speeds stated by the authors

Values from each tool's README and the like (the machines differ, so they cannot be compared with each other).

| Tool | Keys/s | Machine |
|---|---|---|
| vanity-key-ssh (GPU) | about 330M | RTX 4090 |
| vanity-keygen | about 300M | RTX 4090 |
| SSH-Ed25519-VanityGen (GPU) | about 27.8M | H100 NVL |
| vanity-key-ssh (CPU) | about 900K | Apple M series (14 threads) |
| SSH-Ed25519-VanityGen (CPU) | about 640K | AMD EPYC 9334 × 2 |
| vanity-ssh-keygen | about 330K | Apple M2 (8 cores) |
