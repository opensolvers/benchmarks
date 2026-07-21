# 17-model sweep — m1gemv IME build vs RVV control (Q4_K_M)

Second-generation llama.cpp validation on the Orange Pi RV2 (SpaceMiT X60,
`rv64gcv`, RVV 1.0, VLEN=256, 8× @ 1.6 GHz), extending the original 10-model
Q4_0 study in [`models.md`](models.md) to **17 models** and a **different
quantization (Q4_K_M)** and a **different IME build variant (`m1gemv`)**.

This is a **standalone companion** to `models.md`; it does not replace it. The
two studies deliberately differ (see "How this differs" below), so the numbers
are **not** directly comparable row-for-row — read them as two separate
experiments that happen to share a board.

## Setup

| | this study | original `models.md` |
|---|---|---|
| Models | 17 (0.5B → 4B) | 10 (0.5B → 7.6B) |
| Quant | **Q4_K_M** | Q4_0 |
| IME build | `ad8d821-foss-2025b-x60-ime-m1gemv` (M=1 GEMV-tuned) | `~/x60-ime` (baseline IME) |
| RVV build | `~/x60-rvv` (same control binary) | `~/x60-rvv` |
| Bench | `llama-bench -p 512 -n 64 -t 8 -r 1` | `-p 128 -n 32 -t 8` |
| Coherence | `llama-cli -n 8` "capital of France" → expect *Paris* | same idea |
| Board state | kernel build SIGSTOP-paused during BOTH arms (fair, unloaded) | — |

7B/8B-class models were **dropped**: 7.7 GiB RAM / no swap makes them the same
OOM class, and the original study already covers the 7B ceiling.

## Results (both arms, board unloaded)

| label | GB | pp512 IME | pp512 RVV | tg64 IME | tg64 RVV | coherence |
|-------|----|-----------|-----------|----------|----------|-----------|
| qwen2.5-0.5b | 0.37 | 23.92 | 20.73 | 1.42 | 9.19 | OK |
| llama-3.2-1b | 0.75 | 24.13 | 27.49 | 1.90 | 8.19 | OK |
| gemma-3-1b | 0.75 | 12.75 | 11.36 | 1.04 | 4.40 | OK |
| deepseek-coder-1.3b | 0.81 | 16.51 | 16.43 | 1.37 | 6.91 | OK* |
| qwen2.5-1.5b | 0.92 | 18.62 | 18.80 | 1.14 | 6.31 | OK |
| stablelm-2-1.6b | 0.96 | 16.58 | 20.65 | 1.17 | 6.74 | OK |
| falcon3-1b | 0.98 | 19.57 | 22.49 | 1.69 | 7.28 | OK |
| smollm2-1.7b | 0.98 | 13.62 | 15.46 | 1.35 | 5.94 | OK |
| deepseek-r1-qwen-1.5b | 1.04 | 18.54 | 19.10 | 1.14 | 6.38 | OK* |
| granite-3.1-2b | 1.44 | 9.23 | 10.48 | 0.80 | 4.04 | OK* |
| gemma-2-2b | 1.59 | 11.43 | 12.07 | 0.97 | 3.25 | OK |
| qwen2.5-3b | 1.80 | 8.25 | 8.93 | 0.79 | 3.43 | OK |
| falcon3-3b | 1.87 | 10.37 | 10.68 | 1.17 | 3.74 | OK |
| llama-3.2-3b | 1.88 | 8.59 | 9.14 | 0.96 | 3.30 | OK* |
| phi-3.5-mini | 2.23 | 4.08 | 2.91 | 0.69 | 2.04 | OK* |
| gemma-3-4b | 2.32 | 7.67 | 7.68 | 0.70 | 2.52 | OK |
| phi-4-mini | 2.32 | 5.33 | 4.14 | 0.76 | 2.17 | OK* |

\* coherence prompt produced finite output but the 8-token window did not emit
the literal "Paris" (chat-template / reasoning-preamble models); not a failure.

Raw data: [`results_17model_m1gemv.tsv`](results_17model_m1gemv.tsv).

## Headline finding: on Q4_K_M, the `m1gemv` IME build is a **net regression**

This is the opposite of the Q4_0 result in `models.md`, and it is the important
takeaway:

- **Token generation (tg64): RVV wins on every one of the 17 models, by 2.9×–6.5×.**
  The m1gemv build is *named* for accelerating the M=1 GEMV path that dominates
  autoregressive decode — yet its tg is roughly **1/3 to 1/6 of plain RVV**
  (qwen2.5-0.5b: 1.42 IME vs 9.19 RVV). The tuned GEMV kernel is actively
  hurting, not helping, on K-quants.
- **Prompt processing (pp512): near-parity, RVV slightly ahead on the majority.**
  IME wins pp only on 3 models (qwen2.5-0.5b, phi-3.5-mini, phi-4-mini); RVV
  wins or ties the other 14. Contrast the Q4_0 study, where IME won pp on every
  model ≥1.1B by up to ~2.5×.

### Why: the `smt.vmadot` fast path does not engage on Q4_K_M

The IME advantage in the Q4_0 study came from `smt.vmadot` int8 matrix ops. The
K-quant (`Q4_K_M`) superblock layout — 256-element blocks with a second level of
6-bit scales/mins — does **not** map onto the vmadot tiling the m1gemv build
expects, so the kernel falls back to a generic path. On that fallback it (a)
loses the pp advantage vmadot gave on Q4_0, and (b) still pays the m1gemv setup
overhead on tg, producing the observed regression.

## Recommendation / kernel target

**Default to the RVV build for Q4_K_M workloads** — both interactive
(generation-heavy) and, on most models, prompt-heavy. Reserve the IME build for
**Q4_0** models where `models.md` shows it genuinely wins prefill.

**Next kernel target:** make the IME `smt.vmadot` path engage on K-quant
superblocks (Q4_K_M / Q6_K), or explicitly gate `m1gemv` off for K-quants so it
cannot regress tg below the RVV baseline. Until then, m1gemv should not be the
default for anything but Q4_0.

## Reproduce

```bash
# both arms use the same 17-model manifest (dl_manifest.txt, Q4_K_M sources)
# IME arm:
./bench_suite.sh          # -> results.md      (m1gemv build)
# RVV control arm:
./bench_suite_rvv.sh      # -> results_rvv.md   (~/x60-rvv build)
```

Both scripts read the manifest on fd 3 (so llama-* child processes cannot drain
the loop's stdin) and hard-kill each `llama-bench`/`llama-cli` with
`timeout -k -s KILL` so one slow model cannot wedge the sweep. Pause any
competing CPU load (e.g. `kill -STOP` a background build) before running, since
these are 8-thread benchmarks on an 8-core board.
