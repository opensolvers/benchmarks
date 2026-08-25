# MLAS IME CompInt8 — M=1 improvement log

Target: raise ONNX/MLAS int4 decode (`1×4096×11008`) from **~0.47 GOP/s** toward
llama.cpp’s M1 `i8i4` bar (**~12 GOP/s**) / STREAM ceiling (**~15 GOP/s**).

Board: Orange Pi RV2, ORT 1.29.0 `foss-2025b-xsmtvdot`, `taskset -c 0`.

## Root cause (kernel read)

`sqnbitgemm_kernel_ime.cpp` (IME `SQ4BitGemmKernel_CompInt8`):

1. **A tiles rebuilt inside the N loop** — fixed (hoist).
2. **`std::vector` tile buffers** — fixed (stack).
3. **B stays packed 4-bit** → on-the-fly nibble unpack (rvvgather) or pack-time Q4_0×16 (m1pack).
4. **Always 4×4 IME tiles** (zero-pads M=1) — fixed for M=1 via llama GEMV path.

## A/B (2026-08-23 / 08-24)

| Build | M×K×N | med GOP/s | packedB | notes |
| ----- | ----- | --------: | ------: | ----- |
| stock | 1×4096×11008 | **0.48** | 22.5 MB | |
| hoist | 1×4096×11008 | **0.57** | 22.5 MB | |
| rvvgather | 1×4096×11008 | **0.60** | 22.5 MB | |
| m1 (runtime repack) | 1×4096×11008 | **0.54** | 22.5 MB | correct; repack tax |
| unpacked int8 | 1×4096×11008 | **0.30** | 45.1 MB | bandwidth regression |
| **m1pack** | 1×4096×11008 | **10.26–10.42** | 25.4 MB | pack-time Q4_0×16; vec A quantize |
| llama i8i4 m1 | 1×4096×11008 | **12.1** | — | **kernel-only** (A pre-quantized) |
| m1pack | 1×4096×4096 | **10.33** | 9.4 MB | |
| m1pack | 4×4096×11008 | **2.09** | 25.4 MB | gather from x16 |
| rvvgather | 4×4096×11008 | **2.02** | 22.5 MB | |

Artifacts: `~/onnx-mlas-bench/qnbit-mlas-bench-{stock,hoist,rvvgather,m1,m1pack,unpacked}`.

## Pack-time Q4_0×16 (2026-08-24) — **ship**

Panel = 288 B/kb (16×fp16 scales + 256 B weights); `ldb = BlockCountK * 18`;
pack size = `ceil(N/16) * BlockCountK * 288`.

- `SQ4BitGemmPackQuantBDataAndBlkSum` — weights and/or scales into panels (no BlkSum remap of `QuantBScale`)
- M=1 → `SQ4BitGemmM1Kernel_CompInt8` asm on packed panels (no per-call repack)
- M≥4 → RVV gather from the same x16 layout
- Bench: two-phase `MlasQNBitGemmPackQuantBData` (weights, then scales)

**Result:** M=1 **~17×** vs rvvgather. Full ORT path **~10.4 GOP/s** (quantize+kernel);
llama **~12.1 GOP/s** on the same shape with A pre-quantized → **~15%** gap is mostly
per-token `QuantizeARow`, not the M1 asm. RVV quantize (llama `quantize_a_row_i8` BlkLen=32)
adds **~1–2%** vs scalar (~10.26→10.42).

### Further headroom

1. Kernel-only MLAS ≈ llama (panel C++ loop / B traffic — needs split bench hook).
2. Faster or amortized A quantize if activations are reused across layers.
3. Optional: ggml-identical Q4_0 pack (constant-scale microbench shows llama reads same bytes).

Until then **ship m1pack** with vec quantize for M=1 decode.

## End-to-end ONNX (2026-08-24)

Full graph `int4_ffn_acc4.onnx` (16× `MatMulNBits`, `accuracy_level=4`) via
`onnxruntime_perf_test` after `apply-ime-m1pack.sh` + ORT relink.

**Runtime:** `LD_LIBRARY_PATH` must include EESSI **cvmfs** `libiconv` + GCC 14
(`~/eessi-x60/...` on the board is a partial tree — use `/cvmfs/dev.eessi.io/...`).

| Path | x1 P50 | x8 P50 | notes |
| ---- | -----: | -----: | ----- |
| stock CompInt8 (README) | **3522 ms** | **590–960 ms** | pre-m1pack |
| **m1pack e2e** | **139 ms** | **1027 ms** | microbench 8.8 GOP/s same build |

M=1 decode (**x1**) is **~25×** faster end-to-end. x8 is flat/slightly slower —
M≥4 gather path does not win on this graph yet; decode workloads should use x1.

Script: `run-e2e-m1pack.sh` (cvmfs libiconv fallback + relink + microbench + perf_test).

## Real LLM (2026-08-24)

Downloaded **AMD Qwen2.5-0.5B-Instruct int4** ONNX GenAI package
(`~/onnx-models/qwen25-0.5b-int4`, ~780 MB weights, 24 layers, `MatMulNBits` +
`accuracy_level` present). Ran greedy decode with ORT C++ (no genai), linking
the m1pack `libonnxruntime.so`.

| Step | Result |
| ---- | ------ |
| Prefill 15 tok | **19.4 s**, finite logits |
| Decode (greedy) | **~16 s/tok** (1 thread), 12 new tokens |
| Output text | `The answer to 2+2 is 4.` |

### fp16-native MatMulNBits (2026-08-24 evening)

Added `QuantizeARow_CompInt8_FromFp16` + `MlasQNBitGemmBatch_CompInt8_FromFp16A`
so non-ARM fp16 activations skip bulk A-matrix fp16→fp32 before CompInt8 IME.
Also replaced per-call `std::vector<float>` C buffer with arena alloc in
`matmul_nbits.cc`. Runner: arena + pre-sized KV (IoBinding removed — breaks
dynamic present shapes).

| Step | Before | After opt |
| ---- | -----: | --------: |
| Prefill 15 tok | ~19.4 s | **18.6 s** |
| Decode | ~16.0 s/tok | **~15.9 s/tok** |

Symbols verified: `MlasQNBitGemmBatch_CompInt8_FromFp16A`, `FromFp16` quantize.
No meaningful speedup — fp16↔fp32 convert was never the main cost.

### ORT profiling (2026-08-24 evening)

Chrome-trace: prefill + 1 decode token (`run-profile-qwen.sh`).

**Decode (~16 s wall) — kernel time**

| Op | ms | % |
|----|---:|--:|
| **MatMulNBits** | **15,938** | **99.7%** |
| GroupQueryAttention | 9.5 | 0.06% |
| LayerNorm / GELU / Mul | ~13 | 0.08% |

**MatMulNBits decode by projection**

| Projection | ms |
|------------|---:|
| **lm_head** (1×896×151936) | **5,006** |
| gate / down / up_proj (×24) | ~3,200 each |
| v_proj + o_proj | ~1,340 |

**Why microbench wins but Qwen doesn't**

All 121 `MatMulNBits` nodes in AMD `model.onnx` have:

- `accuracy_level=0` → ORT uses **SQNBIT_CompFp32** (full dequant), not CompInt8/m1pack
- `block_size=128` → m1pack IME requires **BlkLen=32**

Synthetic FFN used acc4 + block_size=32 → 139 ms. Qwen hits ~135 ms **per**
FFN matmul on the slow unpacked path.

`patch_accuracy_level.py` updated to force 0→4. Still need BlkLen=128 CompInt8
kernel or weight repack for full m1pack speed. Scripts: `run-profile-qwen.sh`,
`parse_ort_profile.py`, `patch_accuracy_level.py`.

### BlkLen=128 IME CompInt8 (2026-08-25) — **ship for Qwen**

Extended m1pack dual-path:

- `BlkLen=32` → Q4_0×16 pack + llama M1 (unchanged)
- `BlkLen=128` → column-major pack + RVV nibble gather + IME `smt.vmadot`

Fixed RVV unpack loop (was one `vsetvl` only — truncated BlkLen=128 to 64
nibbles → garbage logits). With `model_acc4.onnx` (accuracy_level=4):

| Step | CompFp32 (acc0) | **IME CompInt8 BlkLen=128** |
| ---- | --------------: | --------------------------: |
| Prefill 15 tok | ~18.4 s | **3.9 s** (~4.7×) |
| Decode | ~16.0 s/tok | **~0.93 s/tok** (~17×) |
| Output | `…2+2 is 4.` | **same tokens** |

Log: `~/logs/qwen-blk128-ime2.out`. Remaining headroom vs BlkLen=32 m1pack
(~10 GOP/s) is mostly BlkLen=128 gather tax + no dedicated M1 for 128.

Prompt tokens were approximate (`system` vs `user` role) but generation is
coherent — real MatMulNBits path through m1pack ORT. Per-token cost is much
higher than the synthetic FFN microbench (graph has attention + KV + vocab
proj; this AMD export is int4/**fp16**). Scripts: `run_real_llm_ort.cpp`,
`run-real-llm-ort.sh`.

## Files

| Path | Role |
| ---- | ---- |
| `vendor/sqnbitgemm_kernel_ime.m1pack.cpp` | **ship** — pack-time Q4_0×16 + M1 + M≥4 gather |
| `vendor/sqnbitgemm_kernel_ime.rvvgather.cpp` | hoist + RVV B gather (pre-m1pack) |
| `vendor/sqnbitgemm_kernel_ime.m1.cpp` | runtime-repack M1 (superseded by m1pack) |
| `vendor/sqnbitgemm_ime_quantize.inc` | llama RVV A-row quantize (BlkLen=32) |
| `vendor/sqnbitgemm_ime_m1_panel.inc` | llama M1 asm macros |
| `vendor/qnbitgemm.h`, `qnbitgemm.cpp` | `ldb` hook + M1 CompInt8 dispatch |
| `vendor/llama_ime1_kernels.cpp` | upstream reference |
| `bench_qnbit_mlas.cpp` | isolated CompInt8 rate (two-phase pack) |
| `rebuild-ime-ab.sh` / `apply-ime-unpacked.sh` | board helpers |
| `apply-ime-m1pack.sh` / `run-e2e-m1pack.sh` | deploy m1pack + full e2e validation |
| `run_real_llm_ort.cpp` / `run-real-llm-ort.sh` | real Qwen2.5-0.5B int4 greedy decode |
