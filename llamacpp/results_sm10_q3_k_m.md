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
| Fork | `opensolvers/llama.cpp` branch `x60-ime-rvv` @ `4239429` |
| Toolchain | Bianbu GCC 15.2.0, binutils 2.46 |
| IME build | `-DGGML_CPU_RISCV64_SPACEMIT=ON`. Startup: `use_ime2: 1`, `use_ime1: 0`, `cpu_mask: ff00` |
| Bench | `llama-bench -p 128 -n 32 -r 3` |
| Clocks | A100 userspace 1.800 GHz (cores 8–15) |
| Launch | `echo $$ > /proc/set_ai_thread` before `exec`, then `taskset -c 8-15` |
| TCM | `SPACEMIT_DISABLE_TCM=1`. `/dev/tcm_sync_mem` is absent; the init barrier falls back to the heap |

Tokens per second, IME2, A100. Repetitions: 3.

| Threads | pp128 | tg32 |
| --- | ---: | ---: |
| 1 | 4.30 ± 0.00 | 3.76 ± 0.00 |
| 4 | 20.93 ± 0.00 | 15.37 ± 0.00 |
| 8 | 41.16 ± 0.00 | 22.29 ± 0.00 |

The same model’s Q4_0 file on this board, IME2, 8 threads, is **525.96 / 38.78**.
This file is about **13×** slower on prefill. Q4_K and the Q8_0 head use IME2
(`gemm_kernel_i8i4`, `gemm_kernel_i8i8`). IQ4_NL does not. At VLEN=1024 the
repack selects a 64-column RVV kernel when the row count is divisible by 64,
which every IQ4_NL tensor here is. Q5_0 uses a scalar reference: the asm Q5
kernel consumes the high-bit plane as a predicate over an LMUL=8 group, and
that does not match the per-column repack. Q5_1 and Q5_K stay on the asm path.

If Q5_0 is repacked and then left out of `compute_forward`, this model repeats
one token. Dispatching it through the scalar reference produces coherent
English. A greedy `llama-cli` completion (`-n 32 -t 1 --temp 0`) began:
“RISC-V vector extensions are a set of instructions that allow a CPU to
perform operations on multiple data items simultaneously, rather than
processing them sequentially.”

The 16-column IQ4 kernel on the same IME2 build, before `4239429` selected
the 64-column kernel, same protocol (means of 3 repetitions):

| Threads | pp128 | tg32 |
| --- | ---: | ---: |
| 1 | 4.71 | 3.43 |
| 4 | 19.05 | 13.37 |
| 8 | 37.63 | 20.16 |

Widening IQ4_NL from 16 to 64 lanes is about **10%** at 8 threads. One-thread
prefill is slightly slower. The nibble table has 16 entries. Loading only
those into a 64-lane register made columns 16–63 gather empty lanes, and the
model repeated one token. The table is copied into every 16-lane slice.
VLEN=256 still uses the 16-column kernel.

An RVV build (`GGML_CPU_RISCV64_SPACEMIT=OFF`) on the A100 at 8 threads, with
the 16-lane VLEN=1024 IQ4 kernel, was **18.25 ± 0.00 / 7.67 ± 0.00**. Q4_K
and Q8_0 are not on IME2 in that build. Threads 1 and 4 were not measured.
The BXM-4-64 GPU is still unmeasured.
