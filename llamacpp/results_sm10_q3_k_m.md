# llama.cpp Q3_K_M on BPI-SM10 / K3 (A100 IME2)

One model, 2026-10-29. **Qwen2.5-0.5B-Instruct Q3_K_M**, 406.35 MiB,
630.17 M parameters (`llama-bench` labels it `qwen2 1B Q3_K - Medium`).
Same bench shape as the [Q4_0 K3 note](results_sm10_q4_0.md).

The GGUF is named Q3_K_M. This file has **no Q3_K tensors** and no Q6.
The large matmuls are **IQ4_NL**: FFN gate and up, attention Q and K, and the
token embedding (97 tensors). FFN down is Q4_K except block 0, which is Q5_K
(23 + 1). Attention output and 22 of the attention V projections are Q5_0;
the other two V projections are Q5_1 (46 + 2). The `output` weight is one
Q8_0 tensor.

| | |
| --- | --- |
| Board | Banana Pi BPI-SM10, SpacemiT K3, Bianbu 4.0.6 |
| Fork | `opensolvers/llama.cpp` branch `x60-ime-rvv` @ `0e90371` |
| Toolchain | Bianbu GCC 15.2.0, binutils 2.46 |
| IME build | `-DGGML_CPU_RISCV64_SPACEMIT=ON`. Startup: `use_ime2: 1`, `use_ime1: 0`, `cpu_mask: ff00` |
| Bench | `llama-bench -p 128 -n 32 -r 3` |
| Clocks | A100 userspace 1.800 GHz (cores 8–15) |
| Launch | `echo $$ > /proc/set_ai_thread` before `exec`, then `taskset -c 8-15` |
| TCM | `SPACEMIT_DISABLE_TCM=1`. `/dev/tcm_sync_mem` is absent; the init barrier falls back to the heap |

Tokens per second, IME2, A100. Repetitions: 3. Q5_0, Q5_1, and Q5_K go through
`gemm_kernel_i8i5` (`m4` when the row count is at least 4, `m1` otherwise).

| Threads | pp128 | tg32 |
| --- | ---: | ---: |
| 1 | 12.50 ± 0.01 | 8.66 ± 0.00 |
| 4 | 101.41 ± 0.02 | 37.30 ± 0.01 |
| 8 | 187.64 ± 0.08 | 48.96 ± 0.01 |

The same model’s Q4_0 file on this board, IME2, 8 threads, is **525.96 / 38.78**.
This file’s prefill is **2.8×** below that. Decode at 8 threads is **48.96**,
above the Q4_0 8-thread decode and below the Q4_0 peak at 4 threads (**58.67**).

Q4_K and the Q8_0 head use IME2 (`gemm_kernel_i8i4`, `gemm_kernel_i8i8`).
Q5 uses the IME2 vmadot kernel too. The high-bit plane is one bit per element
in column-major order, and that is the `v0.t` mask applied after the nibble
unpack. IQ4_NL does not use vmadot. At VLEN=1024 the repack selects a
64-column RVV kernel when the row count is divisible by 64, which every
IQ4_NL tensor here is.

Leaving Q5_0 out of `compute_forward` after repack makes this model repeat
one token. With Q5_0 on the vmadot path, a greedy `llama-cli` completion
(`-n 32 -t 1 --temp 0`) began: “RISC-V vector extensions are a set of
instructions that allow a CPU to perform operations on multiple data items
simultaneously, rather than processing them sequentially.”

Before `0e90371`, Q5_0 used a scalar reference on the same IME2 build
(IQ4_NL already on the 64-column kernel). Same protocol:

| Threads | pp128 | tg32 |
| --- | ---: | ---: |
| 1 | 4.30 ± 0.00 | 3.76 ± 0.00 |
| 4 | 20.93 ± 0.00 | 15.37 ± 0.00 |
| 8 | 41.16 ± 0.00 | 22.29 ± 0.00 |

The vmadot dispatch is **4.6×** on 8-thread prefill and **2.2×** on decode.

The 16-column IQ4 kernel, still with scalar Q5_0, was **37.63 / 20.16** at
8 threads. Widening IQ4_NL from 16 to 64 lanes was about **10%** while Q5_0
stayed scalar. The nibble table has 16 entries; it is copied into every
16-lane slice so a 64-lane gather reads it. VLEN=256 still uses the
16-column kernel.

An RVV build (`GGML_CPU_RISCV64_SPACEMIT=OFF`) on the A100 at 8 threads, with
the 16-lane VLEN=1024 IQ4 kernel, was **18.25 ± 0.00 / 7.67 ± 0.00**. Q4_K,
Q5, and Q8_0 are not on IME2 in that build. Threads 1 and 4 were not measured.
The BXM-4-64 GPU is still unmeasured.
