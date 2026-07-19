# Models that fit the RV2 and run on llama.cpp — end-to-end validation set

A list of GGUF models that fit and run on the Orange Pi RV2 (SpaceMiT X60,
`rv64gcv`, RVV 1.0, VLEN=256, 8× @ 1.6 GHz) via the X60 IME-accelerated
llama.cpp build, for end-to-end (load → prefill → generate → coherent output)
validation of the IME `smt.vmadot` Q4_0 path.

## Board constraints (measured)

- **RAM: 7.7 GiB total, ~6.3 GiB available, NO swap** — hard ceiling. Model
  weights + KV cache + context buffers + runtime must fit. Safe working budget:
  **~5.5 GiB** of model+KV before the OOM killer risks the run.
- **Disk: 408 GB free** — model storage is not a constraint; RAM is.
- **IME accelerates `Q4_0`** (`use_ime1: 1`). Prefer **Q4_0** so validation
  exercises the accelerated kernel. Q4_K_M / other quants run but on the RVV/scalar
  path, not the IME `smt.vmadot` core.
- **Token-gen is memory-bandwidth-bound** (~3.85 GB/s single-stream): tg t/s ≈
  bandwidth / model-bytes-per-token. Prefill (pp) is compute/IME-bound and fast.
  So larger models stay *loadable* but tg drops ~linearly with size.
- **IME and RVV split the win by phase** (measured, all 10 models below): the IME
  `smt.vmadot` build wins **prefill** (pp) on every model ≥1.1B — up to ~2.5× the
  RVV build (Mistral-7B: 5.84 vs 2.35 pp t/s). The plain **RVV** build wins
  **token-gen** (tg) everywhere by ~5–8× (0.5B: 12.18 vs 1.47 tg t/s), because the
  IME path's tg is roughly half RVV's. Net: **prompt-heavy / batch → IME;
  generation-heavy / interactive → RVV.** The doc's earlier "~1.5 t/s tg wall" was
  an IME-build artifact, not a board limit.

### Baseline (measured, this build)

| Model | Quant | Size | pp128 (t/s) | tg32 (t/s) |
|---|---|---|---:|---:|
| Qwen2.5-0.5B-Instruct | Q4_0 | 403 MiB | 69.2 | 1.48 |

(This is the **IME** build; its tg is ~1.5 t/s even at 0.5B. The **RVV** build
hits 12.18 tg t/s on the same model — see the measured table below. So on the IME
build treat tg as a *correctness/coherence* check; for interactive speed use RVV.)

## Recommended validation set (Q4_0, IME path)

Ordered smallest→largest. "RAM est." = weights + a modest 2–4k-ctx KV; all fit
the ~5.5 GiB budget with headroom.

| # | Model | Params | Q4_0 size | RAM est. | Notes |
|---|---|---:|---:|---:|---|
| 1 | **Qwen2.5-0.5B-Instruct** | 0.5B | ~0.4 GiB | <1 GiB | already on board; smoke-test baseline |
| 2 | **TinyLlama-1.1B-Chat** | 1.1B | ~0.6 GiB | ~1 GiB | classic tiny-chat sanity model |
| 3 | **Qwen2.5-1.5B-Instruct** | 1.5B | ~0.9 GiB | ~1.5 GiB | strong small instruct |
| 4 | **Llama-3.2-1B-Instruct** | 1.2B | ~0.7 GiB | ~1.2 GiB | Meta baseline, well-known outputs |
| 5 | **Gemma-2-2B-it** | 2.6B | ~1.5 GiB | ~2.2 GiB | different arch family (coverage) |
| 6 | **Qwen2.5-3B-Instruct** | 3.1B | ~1.8 GiB | ~2.6 GiB | mid-size instruct |
| 7 | **Llama-3.2-3B-Instruct** | 3.2B | ~1.9 GiB | ~2.7 GiB | Meta mid-size |
| 8 | **Phi-3.5-mini-instruct** | 3.8B | ~2.2 GiB | ~3.2 GiB | long-ctx capable arch |
| 9 | **Mistral-7B-Instruct-v0.3** | 7.2B | ~3.9 GiB | ~4.8 GiB | upper practical bound at 7B Q4_0 |
| 10 | **Qwen2.5-7B-Instruct** | 7.6B | ~4.2 GiB | ~5.2 GiB | largest safe fit; keep ctx ≤2k |

## Measured results — all 10 validated (IME vs RVV, this build)

Every model below **loaded, reported `use_ime1: 1`, and produced coherent, finite
output** (no repetition-collapse). No OOM at any size — the largest (Qwen2.5-7B,
4.43 GiB of weights across 2 shards) held at ~2.5–2.8 GiB available during run.
Bench = `llama-bench -p 128 -n 32 -t 8`; t/s reported as the mean (± omitted).

| Model | Params | Q4_0 (MiB) | Coherent | pp128 IME | tg32 IME | pp128 RVV | tg32 RVV | pp win | tg win |
|---|---:|---:|:--:|---:|---:|---:|---:|:--:|:--:|
| Qwen2.5-0.5B-Instruct | 0.5B | 408 | yes | 69.03 | 1.47 | 100.60 | 12.18 | RVV | RVV |
| TinyLlama-1.1B-Chat | 1.1B | 608 | yes | 37.72 | 1.72 | 34.95 | 9.47 | **IME** | RVV |
| Qwen2.5-1.5B-Instruct | 1.5B | 1016 | yes | 27.14 | 1.19 | 22.77 | 6.10 | **IME** | RVV |
| Llama-3.2-1B-Instruct | 1.2B | 737 | yes | 38.87 | 2.00 | 25.27 | 7.74 | **IME** | RVV |
| Gemma-2-2B-it | 2.6B | 1554 | yes | 17.56 | 1.01 | 10.04 | 3.15 | **IME** | RVV |
| Qwen2.5-3B-Instruct | 3.1B | 1905 | yes | 14.13 | 0.85 | 7.85 | 3.12 | **IME** | RVV |
| Llama-3.2-3B-Instruct | 3.2B | 1832 | yes | 13.99 | 1.05 | 6.95 | 3.09 | **IME** | RVV |
| Phi-3.5-mini-instruct | 3.8B | 2081 | yes | 11.37 | 1.11 | 4.59 | 2.02 | **IME** | RVV |
| Mistral-7B-Instruct-v0.3 | 7.2B | 3922 | yes | 5.84 | 0.73 | 2.35 | 1.38 | **IME** | RVV |
| Qwen2.5-7B-Instruct | 7.6B | 3798 | yes | 6.06 | 0.73 | 2.52 | 1.38 | **IME** | RVV |

**Takeaways:**
- **IME wins prefill on every model ≥1.1B** (1.2×–2.6×); the only exception is the
  0.5B, where per-call IME setup overhead dominates and RVV's raw pp is higher.
- **RVV wins token-gen universally**, by ~5–8× at small sizes narrowing to ~1.9× at
  7B. IME's tg path is ~half RVV's throughout.
- **tg scales down with size as predicted** by the bandwidth wall (RVV: 12.18 →
  1.38 t/s from 0.5B → 7B); pp stays IME-favoured across the whole range.

### Verified Q4_0 sources (HF repos used)

- Qwen2.5-{0.5B,1.5B,3B}: `Qwen/Qwen2.5-*-Instruct-GGUF`; **7B is sharded**
  (`*-00001-of-00002` + `*-00002-of-00002`).
- TinyLlama: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`.
- Llama-3.2-{1B,3B}: `unsloth/Llama-3.2-*-Instruct-GGUF` (the `hugging-quants`
  repos are gated → 401).
- Gemma-2-2B-it & Mistral-7B-Instruct-v0.3: `second-state/*-GGUF` (bartowski has
  **no Q4_0** for these two).
- Phi-3.5-mini-instruct: `bartowski/Phi-3.5-mini-instruct-GGUF`.

### Fits but marginal (validate load only, short ctx)

- **Llama-3.1-8B-Instruct** (8B, Q4_0 ~4.5 GiB, ~5.6 GiB w/ small KV) — right at
  the budget edge; run with `-c 1024`, no other RAM pressure, expect ~0.7 t/s tg.

### Do NOT attempt (exceeds RAM)

- Anything ≥13B at Q4_0 (~7+ GiB weights alone) — no swap, will OOM.
- 7–8B at Q8_0 / fp16 — weights alone blow the 7.7 GiB ceiling.

## Validation procedure (per model)

1. `wget` the Q4_0 GGUF into `~/llama-x60/models/`.
2. **Load + coherence** (`llama-cli`), IME build:
   ```bash
   cd ~/x60-ime
   LD_LIBRARY_PATH=~/x60-ime taskset -c 0-7 ./llama-cli \
     -m ~/llama-x60/models/<model>.q4_0.gguf -t 8 -c 2048 -n 64 \
     -p "Explain what RISC-V vector extensions are, in two sentences."
   ```
   Pass = loads, `use_ime1: 1`, emits coherent, finite (no repetition-collapse) text.
3. **Throughput** (`llama-bench`): `-p 128 -n 32 -t 8`, record pp128 / tg32.
4. **IME-path confirmation**: startup log shows `use_ime1: 1`; pp128 markedly
   above the RVV-only build (compare against `~/x60-rvv/llama-bench`).

## Notes

- Threads: `-t 8` (all cores) is fastest for pp on this board; tg is
  bandwidth-bound so extra threads help little past ~4.
- Keep context modest (≤2–4k) on ≥3B models — KV cache competes for the same
  scarce RAM.
- The `~/x60-rvv/` build is the non-IME control arm for any A/B.
