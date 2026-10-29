# llama.cpp Q4_0 on BPI-SM10 / K3 (A100 IME2 vs X100 RVV)

One model, 2026-10-29. Same GGUF and bench shape as the first row of the
[RV2 Q4_0 table](models.md): **Qwen2.5-0.5B-Instruct Q4_0**, 403.20 MiB,
630.17 M parameters (`llama-bench` labels it `qwen2 1B`).

| | |
| --- | --- |
| Board | Banana Pi BPI-SM10, SpacemiT K3, Bianbu 4.0.6 |
| Fork | `opensolvers/llama.cpp` branch `x60-ime-rvv` @ `72d1824` |
| Toolchain | Bianbu GCC 15.2.0, binutils 2.46 (`-march=…_xsmtvdotii`) |
| IME build | `-DGGML_CPU_RISCV64_SPACEMIT=ON` plus the RVV/Zfh/Zba/Zicbop flags. CMake: `RISCV64_SPACEMIT_IME1;RISCV64_SPACEMIT_IME2` |
| RVV build | same flags, `GGML_CPU_RISCV64_SPACEMIT=OFF` |
| Bench | `llama-bench -p 128 -n 32` |
| Clocks | A100 userspace 1.800 GHz (cores 8–15), X100 2.000 GHz (cores 0–7) |
| Launch | `echo $$ > /proc/set_ai_thread` before `exec`, then `taskset -c 8-15` for A100. X100 is `taskset -c 0-7` with no AI-thread write |
| Runtime log | `use_ime2: 1`, `use_ime1: 0`, `perfer_core_arch_id: a064`, `cpu_mask: ff00`, `mem_backend: HPAGE` |
| TCM | `SPACEMIT_DISABLE_TCM=1`. `/dev/tcm_sync_mem` is absent; the init barrier falls back to the heap |

Tokens per second. IME2 repetitions: 3 at 1 thread, 5 otherwise. X100 RVV: 5.

| Build | Cores | Threads | pp128 | tg32 |
| --- | --- | ---: | ---: | ---: |
| IME2 | A100 | 1 | 76.43 ± 0.02 | 21.90 ± 0.04 |
| IME2 | A100 | 2 | 149.65 ± 0.11 | 40.24 ± 0.05 |
| IME2 | A100 | 4 | 291.12 ± 0.26 | 58.67 ± 0.49 |
| IME2 | A100 | 8 | 525.96 ± 1.37 | 38.78 ± 0.47 |
| RVV (VLEN=256 kernel) | X100 | 1 | 27.69 ± 0.05 | 11.72 ± 0.01 |
| RVV (VLEN=256 kernel) | X100 | 8 | 193.68 ± 2.75 | 41.95 ± 3.55 |
| RVV (no VLEN=1024 kernel) | A100 | 1 | 0.87 ± 0.00 | 0.59 ± 0.00 |
| RVV (no VLEN=1024 kernel) | A100 | 8 | 6.68 ± 0.00 | 4.27 ± 0.00 |

Prefill at 8 A100 threads is **2.7×** the X100 RVV prefill. Decode peaks at 4 threads.
Eight A100 threads are slower on tg32 than four. The same model on the RV2 at
8 threads is IME 69.03 / 1.47 and RVV 100.60 / 12.18.

The A100 RVV rows are the stock build. `ggml/src/ggml-cpu/repack.cpp` switches
on `__riscv_vlenb() * 8` and the Q4_0 case for 1024 is empty, so a VLEN=1024
process never selects `q4_0_16x1_q8_0`. The X100 column is the RVV comparison.

A short greedy `llama-cli` completion on the IME2 build (`-t 8`, then again
with `-t 4 -tb 8`) produced coherent English. `llama-bench` itself has no
`-tb` flag; the split is a `llama-cli` setting.
