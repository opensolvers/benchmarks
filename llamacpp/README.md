# llama.cpp on RV2 — Q4_0 model validation (X60 IME vs RVV)

End-to-end validation of the X60 **IME `smt.vmadot` Q4_0** path in llama.cpp on
the Orange Pi RV2 (SpaceMiT X60, `rv64gcv`, RVV 1.0, VLEN=256, 8× @ 1.6 GHz),
against the plain **RVV** build as the control arm. Ten Q4_0 GGUF models from
0.5B → 7.6B were each loaded, coherence-checked (`llama-cli`), and benched
(`llama-bench -p 128 -n 32 -t 8`) on **both** builds.

This is the model-coverage companion to [`../ime`](../ime) (the IME kernel
A/B): where `ime/` isolates the `smt.vmadot` kernel, this directory answers
"which real models fit and run, and where does IME actually help?"

> **See also — second study (Q4_K_M, 17 models):**
> [`results_17model_m1gemv_vs_rvv.md`](results_17model_m1gemv_vs_rvv.md) extends
> this to 17 models on **Q4_K_M** with the M=1-GEMV-tuned `m1gemv` IME build.
> Key finding: on K-quants the IME `smt.vmadot` fast path does **not** engage,
> so `m1gemv` is a **net regression** vs plain RVV (RVV wins tg by 2.9–6.5× on
> every model). The IME-wins-prefill result below is **Q4_0-specific**.

> **Q8_0 + i8i8 kernel (2026-08-22):** the `-x60-ime-q8_0` build adds
> `gemm_kernel_i8i8` for Q8_0/Q6_K activations. On `qwen2.5-0.5b-q8_0.gguf`,
> **pp64 @ t4 = 62.7 t/s** (i8i8 build) vs **25.5** (stock `-x60-ime` RVV fallback)
> and **56.8 pp64 @ t8** (plain RVV build). tg32 stays RVV-favoured (0.83 vs 4.9 vs 8.9).
>
> **Port of ime-bench pipelining (2026-08-23):** load/`smt.vmadot` interleave in
> the i8i8 M4 INNER loop ([`../ime/run-llama-pipe-build.sh`](../ime/run-llama-pipe-build.sh)).
> Kernel: **~4–5 %** (`bench_i8i8` 512³ 24.8→25.8, 768²×512 28.6→29.9 GOP/s median,
> bit-identical). End-to-end `pp512 @ t4` interleaved A/B: stock **77.3** vs pipe
> **78.4** t/s mean — **within noise** (±4–5). Full dual-bank pipelining does not
> fit (FP acc holds `v24..v31`); llama INNER is only `BlkLen/16 = 2` steps, so the
> synthetic 1.45× kloop win does not transfer. Amortized packing N/A (B pre-repacked).
>
> **Hybrid prefill/decode split (2026-08-23):** one binary, two paths inside
> `libggml-cpu` ([`../ime/apply-hybrid.py`](../ime/apply-hybrid.py),
> [`../ime/run-llama-hybrid.sh`](../ime/run-llama-hybrid.sh)). Weights are stored
> **twice** in the spacemit buffer (native GGUF layout + IME tiles). `M >= 4` → IME
> M4; `M < 4` (token-gen) → stock `ggml_compute_forward_mul_mat` on native weights.
> Env: `SPACEMIT_HYBRID=1`, optional `SPACEMIT_IME_MIN_M=4`. Use **`LD_PRELOAD`**
> on the rebuilt `libggml-cpu.so` (q8_0 binary RPATH ignores `LD_LIBRARY_PATH`).
>
> | build | Q8_0 pp512 @ t4 | Q8_0 tg32 @ t4 |
> |---|--:|--:|
> | IME-only (q8_0) | 83.7 | **0.83** |
> | **Hybrid** | **90.1** | **6.68** |
> | RVV-only | 27.0 | 5.11 |
>
> Q4_0 on stock `~/x60-ime` already has decent tg (~7.3); hybrid mainly lifts Q8_0
> decode (~8×) while slightly improving pp. Cost: **~2× weight RAM** in spacemit buffers.

## Contents

| File | Purpose |
|---|---|
| [`models.md`](models.md) | full write-up: board constraints, recommended set, measured IME-vs-RVV table, per-model procedure |
| `model_validation.tsv` | raw results (10 rows: pp128/tg32 for IME and RVV) |
| `validate_model.sh` | per-model harness: download → coherence (IME) → bench IME + RVV → append TSV row → reclaim RAM |
| `run_all.sh` | driver: runs all 10 models (verified HF Q4_0 sources baked in) |
| [`../ime/bench_i8i8.cpp`](../ime/bench_i8i8.cpp) | links `gemm_kernel_i8i8` from `-x60-ime-q8_0` for layout selftest + GOP/s |
| [`../ime/run-i8i8-selftest-ab.sh`](../ime/run-i8i8-selftest-ab.sh) | RV2 runner: chk gate + kernel GOP/s vs ime-bench RVV |
| [`results_17model_m1gemv_vs_rvv.md`](results_17model_m1gemv_vs_rvv.md) | **second study**: 17 models, Q4_K_M, `m1gemv` IME build vs RVV control (full write-up + finding) |
| `results_17model_m1gemv.tsv` | raw results for the second study (17 rows: pp512/tg64 for IME and RVV) |
| `bench_suite.sh` | second-study IME arm: fd-3 manifest read, force-killed timeouts, writes `results.md` |
| `bench_suite_rvv.sh` | second-study RVV control arm: same 17 models on `~/x60-rvv`, writes `results_rvv.md` |

## Headline results (all 10 validated, no OOM)

Every model **loaded, reported `use_ime1: 1`, and produced coherent finite
output.** The two builds split the win by phase:

- **IME wins prefill (pp) on every model ≥1.1B** — up to ~2.5× RVV
  (Mistral-7B: 5.84 vs 2.35 pp t/s). Only the 0.5B favours RVV on pp (per-call
  IME setup overhead dominates at that size).
- **RVV wins token-gen (tg) universally** — ~5–8× at small sizes narrowing to
  ~1.9× at 7B; IME's tg path is roughly half RVV's throughout.
- **tg scales down with model size** as the ~3.85 GB/s memory-bandwidth wall
  predicts (RVV: 12.18 → 1.38 tg t/s from 0.5B → 7B).

**Net: prompt-heavy / batch → IME build; generation-heavy / interactive → RVV
build.** See [`models.md`](models.md) for the full table and takeaways.

## Board ceiling

7.7 GiB RAM, **no swap** → ~5.5 GiB safe model+KV budget. Largest safe fit is
7–7.6B at Q4_0 (Qwen2.5-7B held ~2.5–2.8 GiB free during run). ≥13B Q4_0 and
7–8B at Q8_0/fp16 will OOM — do not attempt.

## Reproduce

Requires the two builds on the board: `~/x60-ime` (IME) and `~/x60-rvv` (RVV
control), plus `~/llama-x60/models/`.

```bash
# single model: URL FILENAME LABEL PARAMS
./validate_model.sh \
  "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf" \
  qwen2.5-1.5b.q4_0.gguf qwen2.5-1.5b 1.5B

# full sweep (all 10, ~appends to ~/model_validation.tsv)
./run_all.sh
```

Each run downloads the Q4_0 GGUF, runs a coherence prompt on the IME build,
benches pp128/tg32 on IME then RVV, appends one TSV row, and deletes the GGUF to
respect the no-swap RAM ceiling.
