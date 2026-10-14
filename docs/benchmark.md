# Benchmark (Google Colab)

English | [日本語](benchmark.ja.md)

The speed of meikey on each Google Colab runtime.

- Text: `TARGET` (case-insensitive, about 29.0M estimated tries)
- Time: 10 seconds each (`--bench 10`)
- Multiplication: all values were measured with the table-based multiplication, which is not constant-time (in the current version, the multiplication of `--low-security`; the default is constant-time)

## Speed

In keys/s.

| Runtime | GPU | CPU | GPU + CPU |
|---|---|---|---|
| RTX PRO 6000 (G4) | 580M | 62.5M | **955.4M** |
| A100 | 110M | 2.93M | 120M |
| L4 | 81.9M | 2.98M | 84.2M |
| T4 | 21.9M | 1.78M | 23.2M |
| CPU only | — | 1.07M | — |

| Runtime | Grid | CPU SIMD | CPU threads (CPU / GPU + CPU) |
|---|---|---|---|
| RTX PRO 6000 (G4) | 752 blocks × 128 threads × 16 keys | AVX-512 IFMA | 48 / 47 |
| A100 | 324 blocks × 128 threads × 8 keys | AVX-512 | 12 / 11 |
| L4 | 174 blocks × 128 threads × 8 keys | AVX-512 | 12 / 11 |
| T4 | 120 blocks × 128 threads × 8 keys | AVX-512 | 8 / 7 |
| CPU only | — | AVX2 | 8 / — |

The RTX PRO 6000 GPU + CPU value was measured with the current version (during the [comparison](comparison.md), `--low-security --both` for 30 seconds).
The RTX PRO 6000 GPU and CPU values were measured with the version using a GPU table width W = 12 and 16 keys per batch.
The A100, L4, T4 and CPU-only rows were measured with the version before it (8 keys per batch).

### Speed by GPU setting (RTX PRO 6000)

GPU speed with different `--gpu-width` (table width W) and `--gpu-batch` (keys per batch).
This table was measured with the version that measured these 6 combinations. The current version measures 20 combinations on the GPU before searching (widths 12, 14, 16, 18, 20 × 32 or 64 keys per batch × 10 or 8 limbs in the field arithmetic) and uses the fastest.

| W | Keys per batch | GPU |
|---|---|---|
| 12 | 16 | 570M |
| 12 | 32 | 570M |
| 14 | 16 | 610M |
| 14 | 32 | 630M |
| 16 | 16 | 680M |
| 16 | 32 | 700M |

## Environment

### GPU

| Runtime | GPU | Compute | SM | Memory | Power limit | Max clock (SM, memory) | PCIe |
|---|---|---|---|---|---|---|---|
| RTX PRO 6000 (G4) | NVIDIA RTX PRO 6000 Blackwell Server Edition | sm_120 | 188 | 95.6 GiB | 600 W | 2430 MHz, 12481 MHz | Gen5 x16 |
| A100 | NVIDIA A100-SXM4-80GB | sm_80 | 108 | 80.0 GiB | 400 W | 1410 MHz, 1593 MHz | Gen4 x16 |
| L4 | NVIDIA L4 | sm_89 | 58 | 22.5 GiB | 72 W | 2040 MHz, 6251 MHz | Gen3 x16 |
| T4 | Tesla T4 | sm_75 | 40 | 15.0 GiB | 70 W | 1590 MHz, 5001 MHz | Gen3 x16 |

The driver is 580.82.07 (CUDA 13.0) on all of them.

### CPU

| Runtime | CPU | Cores, threads | Clock | Cache (L2, L3) | AVX2 | AVX-512 | AVX-512 IFMA |
|---|---|---|---|---|---|---|---|
| RTX PRO 6000 (G4) | AMD EPYC 9B45 | 24, 48 | 2700 MHz | 24 MiB, 96 MiB | ✓ | ✓ | ✓ |
| A100 | Intel Xeon CPU @ 2.20GHz | 6, 12 | 2200 MHz | 6 MiB, 38.5 MiB | ✓ | ✓ | ✗ |
| L4 | Intel Xeon CPU @ 2.20GHz | 6, 12 | 2200 MHz | 6 MiB, 38.5 MiB | ✓ | ✓ | ✗ |
| T4 | Intel Xeon CPU @ 2.00GHz | 4, 8 | 2000 MHz | 4 MiB, 38.5 MiB | ✓ | ✓ | ✗ |
| CPU only | Intel Xeon CPU @ 2.20GHz | 4, 8 | 2200 MHz | 1 MiB, 55 MiB | ✓ | ✗ | ✗ |

### Other

| Runtime | Memory | Disk | OS |
|---|---|---|---|
| RTX PRO 6000 (G4) | 176.9 GiB | 236 GiB | Ubuntu 24.04.4 LTS |
| A100 | 167.1 GiB | 236 GiB | Ubuntu 24.04.4 LTS |
| L4 | 53.0 GiB | 236 GiB | Ubuntu 24.04.4 LTS |
| T4 | 51.0 GiB | 236 GiB | Ubuntu 24.04.4 LTS |
| CPU only | 51.0 GiB | 226 GiB | Ubuntu 24.04.5 LTS |

The kernel is 6.6.122+ on all of them.
