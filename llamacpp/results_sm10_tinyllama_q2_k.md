# llama.cpp TinyLlama Q2_K on BPI-SM10 / K3 (A100 IME2)

One model, 2026-10-31. **TinyLlama 1.1B Chat v1.0 Q2_K**
([TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)),
483116416 bytes. `llama-bench` labels it `llama 1B Q2_K - Medium`
(459.11 MiB, 1.10 B parameters).

The file is tagged Q2_K. Most of the matmul work is **Q3_K**: FFN down, gate,
and up, attention output, and attention V (110 tensors). Q2_K is attention Q
and K, plus the token embedding (45 tensors). The token embedding is a gather,
not a matmul. The `output` weight is one Q6_K tensor. There is no IQ4_NL.

| | |
| --- | --- |
| Board | Banana Pi BPI-SM10, SpacemiT K3, Bianbu 4.0.6 |
| Fork | `opensolvers/llama.cpp` branch `x60-ime-rvv` @ `107ead2f2` |
| Toolchain | Bianbu GCC 15.2.0, binutils 2.46 |
| IME build | `-DGGML_CPU_RISCV64_SPACEMIT=ON`. Startup: `use_ime2: 1`, `use_ime1: 0`, `cpu_mask: ff00` |
| Bench | `llama-bench -p 128 -n 32 -r 3` |
| Clocks | A100 userspace 1.800 GHz (cores 8–15) |
| Launch | `echo $$ > /proc/set_ai_thread` before `exec`, then `taskset -c 8-15` |
| TCM | `SPACEMIT_DISABLE_TCM=1`. `/dev/tcm_sync_mem` is absent; the init barrier falls back to the heap |

Tokens per second, IME2, A100. Repetitions: 3. Q3_K prefill uses
`gemm_kernel_i8i3k_m4`: the 3-bit weights are unpacked once for four rows,
swizzled the same way as the Q2_K 4-row kernel, and dotted with `vmadot.hp`.
The scale inside the dot is `int8_scale * scales16 / 16`. The superblock
epilogue multiplies by `a_scale * 16`. A single generated token still uses
the existing 1-row `vmadot.hp` kernel. Q2_K uses `gemm_kernel_i8i2k`. The
Q6_K output weight is expanded into the Q8_0 tile.

| Threads | pp128 | tg32 |
| --- | ---: | ---: |
| 1 | 35.73 ± 0.01 | 10.93 ± 0.00 |
| 4 | 138.25 ± 0.04 | 35.13 ± 0.02 |
| 8 | 254.37 ± 0.17 | 43.39 ± 0.03 |

The same binary with the previous integer `vmadot` Q3_K prefill kernel, same
protocol, before this swap:

| Threads | pp128 | tg32 |
| --- | ---: | ---: |
| 1 | 27.43 ± 0.01 | 10.82 ± 0.00 |
| 4 | 107.06 ± 0.01 | 35.07 ± 0.07 |
| 8 | 205.17 ± 0.07 | 43.48 ± 0.06 |

Eight-thread prefill is **1.24×** (205.17 → 254.37). One thread is **1.30×**
and four threads are **1.29×**. Decode stays on the 1-row kernel, so tg32
is unchanged. A tile of the new kernel against the integer kernel differed
by at most `7.6e-5`. A greedy `llama-cli` completion
(`-n 40 -t 8 --temp 0`) began: “The RISC-V instruction set architecture (ISA)
is a set of instructions that are optimized for the RISC architecture. The
RISC-V ISA is designed to provide”.

The BXM-4-64 GPU is still unmeasured.
