# onnx — int4 `MatMulNBits` on ONNX Runtime, SpaceMiT X60

Benchmark and root-cause writeup for int4 (`MatMulNBits`, 4-bit weights,
`BlkLen=32`) LLM-FFN inference on the SpaceMiT X60 (RISC-V, RVV + IME
`smt.vmadot`) through ONNX Runtime's MLAS `SQNBit` path.

**Headline:** `accuracy_level=4` selects CompInt8 IME; the shipped m1pack
backend reaches **~10 GOP/s** on BlkLen=32 FFN microbench, **~17×** faster
Qwen2.5-0.5B int4 decode (BlkLen=128) vs CompFp32, and **SQ8Bit + Q8×16**
panels for int8 (`bits=8`, BlkLen=32) — Qwen int8 ~241/159 ms decode (1t/4t).
Full log: [`MLAS_IME_IMPROVE.md`](MLAS_IME_IMPROVE.md).

## The workload

A Llama-7B-proportioned stack of FFN/projection blocks: `hidden=4096`,
`ffn=11008`, 8 layers, **16 `MatMulNBits` nodes** (com.microsoft), 4-bit
symmetric weights, `block_size=32`. This is where real LLM decode spends its
time — the big 4-bit weight GEMMs. Measured with `onnxruntime_perf_test`
(`-m times`, `-r 8`) at decode shape **M=1**.

Also: **AMD Qwen2.5-0.5B-Instruct int4** ONNX (`block_size=128`, fp16 acts) —
real greedy decode via `run_real_llm_ort.cpp`.

## Results (synthetic FFN, stock CompInt8 before m1pack)

| Configuration        | Before (CompFp32 fallback) | After (IME CompInt8) | Speedup    |
|----------------------|---------------------------:|---------------------:|:----------:|
| Single thread (`-x1`)| 31,956 ms                  | 3,522 ms             | **9.1×**   |
| 8 threads (`-x8`)    | 6,074 ms                   | 590 ms               | **10.3×**  |

**RV2 re-verify** (2026-08-20, same models/build, log
`~/logs/onnx-acc4-ab-20260820-191335.log`): x1 **29,851 → 3,532 ms = 8.5×**;
x8 **6,569 → 960 ms = 6.8×**. CompInt8 path confirmed; x8 was softer than the
original board run (CPU usage logged ~99% on ACC4_x8 vs ~199% previously).

- Peak RSS: ~1023 MB → **842 MB** (no fp32 dequant buffers).
- P50 latency sits on the mean (x1: min 3.520 / P50 3.522 s) — stable board.
- x1→x8 thread scaling on the fixed path: 3522/590 = **~6×** across 8 cores.

## Root cause

ORT selects the int4 compute variant in `matmul_nbits.cc`:

```cpp
// accuracy_level == 4  ->  SQNBIT_CompInt8   (the X60 IME kernel)
// anything else        ->  SQNBIT_CompFp32
```

The RISC-V/IME backend registers **only** a `CompInt8` kernel — no `CompFp32`
(unlike NEON/AVX2/AVX512/LASX). So when the model omits `accuracy_level=4`:

1. ORT picks `CompFp32`.
2. IME backend has no such kernel → `MatMulNBits` falls out of MLAS.
3. ORT's generic dequantize-to-fp32 + SGEMM fallback runs → **~32 s**.

The generated model had **zero** `accuracy_level` attributes across all 16
nodes. The generator's docstring *claimed* the CompInt8 path; the bytes
disagreed.

### How it was confirmed (not guessed)

- **Probe:** a one-shot `fprintf` in both the IME kernel's fast path and tile
  path — at M=1, **neither fired**. The kernel under "optimization" never ran.
- **Roofline:** X60 STREAM triad = 3.85 GB/s (1T) / ~10.6 GB/s (8T); int4 B
  traffic ≈ 361 MB/inference → bandwidth floor ~0.09 s (x1). Measured latency
  was **178–341× above** the floor → not bandwidth-bound, i.e. wrong code path,
  not slow code path.
- **Bytes:** `grep -c accuracy_level model.onnx` → 0 across 16 `MatMulNBits`.

## The fix

Set `accuracy_level=4` on every `MatMulNBits` node at generation time:

```python
helper.make_node(
    "MatMulNBits", ...,
    bits=4, block_size=block_size, K=K, N=N,
    accuracy_level=4,   # selects SQNBIT_CompInt8 -> X60 IME kernel.
                        # Without it: CompFp32 -> generic fallback (~9x slower).
)
```

For an already-built model with no `numpy`/`onnx` available (riscv64), the
attribute can be appended directly at the protobuf level — see
[`patch_accuracy_level.py`](patch_accuracy_level.py), a dependency-free walker
that adds `accuracy_level=4` (INT) to each `MatMulNBits` node and recomputes the
enclosing length prefixes. Verified by structural re-parse: 16/16 nodes patched,
+368 bytes (16 × 23), clean append.

After the fix, the probe fires and the kernel runs:

```
[PROBE] FASTPATH hit CountM=1 CountN=128 BlockCountK=128
First inference time cost: 4554 ms      (was ~32,000 ms)
```

## A cautionary sub-result: the kernel micro-opt was a regression

The original goal was an `M<4` RVV gemv "fast path" inside the IME
`SQ4BitGemmKernel_CompInt8`. Once the correct path actually executed, a fair A/B
(M=1, corrected model, probe-free) showed:

| Kernel (M=1)                 | Latency     |
|------------------------------|------------:|
| Hand-written RVV fast path   | 4,521 ms    |
| Stock IME `smt.vmadot` tile  | **3,522 ms**|

The bespoke fast path was **28 % slower** than the code it meant to beat — the
`smt.vmadot` tile path (see [`../ime`](../ime)) handles the gemv shape better.
It was reverted; the shipped kernel is byte-identical to upstream.

## Isolated kernel microbenchmark (`bench_qnbit_mlas.cpp`)

To measure the MLAS `SQNBit` int4 `CompInt8` kernel in isolation — without ORT's
graph, threadpool, or the fp32-fallback confound — [`bench_qnbit_mlas.cpp`](bench_qnbit_mlas.cpp)
links directly against the build tree's `libonnxruntime_mlas.a` and calls the
public MLAS entry points exactly as `matmul_nbits.cc` does:

- `MlasQNBitGemmPackQuantBData` to pack B (nbits=4, `BlkLen=32`, no zero-point),
- `MlasQNBitGemmBatch<float>` for the GEMM,

mirroring the convention of [`../ime/bench_i8i4.cpp`](../ime). It measures a
**single-thread** kernel rate (`thread_pool=nullptr`), reports GOP/s
(min/median/max over N reps), and guards every output element with an
`isfinite` check so a wrong pack layout can't masquerade as a fast run.

**Single-thread results** (X60, one core; end-to-end 8-thread scaling lives in
the perftest table above, not here):

| Shape (M×K×N)        | Kernel rate | packed-B |
|----------------------|------------:|---------:|
| 1 × 4096 × 11008     | 0.47 GOP/s  | 22.5 MB  |
| 1 × 11008 × 4096     | 0.39 GOP/s  | 22.5 MB  |
| 1 × 4096 × 4096      | 0.48 GOP/s  | 8.4 MB   |

These are deliberately *single-core* numbers isolating kernel efficiency; the
production path fans this across 8 cores (~6× — see the perftest table).

### m1pack IME (ship)

See [`MLAS_IME_IMPROVE.md`](MLAS_IME_IMPROVE.md). Dual-path CompInt8:

| BlkLen | Pack / kernel | M=1 microbench |
|-------:|---------------|---------------:|
| **32** | Q4_0×16 panels + llama M1 asm | **~10.4 GOP/s** |
| **128** | column-major + RVV gather + IME | Qwen path |

Synthetic FFN e2e (`accuracy_level=4`, BlkLen=32): x1 **3522 → 139 ms**.

**Qwen2.5-0.5B int4** (AMD export, BlkLen=128, `model_acc4.onnx`):

| Step | CompFp32 (acc0) | m1pack CompInt8 |
|------|----------------:|----------------:|
| Prefill 15 tok | ~18.4 s | **3.9 s** |
| Decode | ~16.0 s/tok | **~0.93 s/tok (~17×)** |

**Int8 (BlkLen=32, SQ8 + Q8×16 panels)** — Qwen decode ~241 ms (1t) / ~159 ms
(4t); SmolLM2-360M int8 ~140 ms @4t. SQ4 regression held (SmolLM2 int4 ~80 ms,
TinyLlama int4 ~156 ms @4t).

Deploy on board: `bash apply-ime-m1pack.sh` then rebuild `onnxruntime_mlas`
+ `onnxruntime`. Patches live under [`vendor/`](vendor/).

### One backend gotcha worth recording

The BlkLen=32 m1pack path uses `SQ4BitGemmPackQuantBDataAndBlkSum` (Q4×16
panels). BlkLen=128 uses a `memcpy` pack with external scales. The isolated
microbench (`bench_qnbit_mlas.cpp`) two-phase packs weights then scales for
BlkLen=32.

```sh
# build + run on the X60 (see Makefile header for the GCC 14 toolchain note)
make board CXX=$GCC14/bin/g++
LD_LIBRARY_PATH=$GCC14/lib64:$LD_LIBRARY_PATH ./qnbit-mlas-bench 1 4096 11008 50
```

## Reproduce

```sh
# 1) generate (or patch) the model so MatMulNBits carries accuracy_level=4
python3 patch_accuracy_level.py int4_ffn.onnx int4_ffn_acc4.onnx
# also rewrites accuracy_level=0 → 4 (AMD Qwen export)

# 2) apply m1pack IME to ORT tree on the X60, rebuild, e2e
bash apply-ime-m1pack.sh
# then: make -C $ORT_BUILD onnxruntime_mlas onnxruntime

# 3) synthetic FFN
onnxruntime_perf_test -e cpu -I -m times -r 8 -x 1 int4_ffn_acc4.onnx

# 4) real Qwen (after model_acc4.onnx exists)
bash run-real-llm-ort.sh   # or MODEL_DIR=... pointing at model_acc4.onnx
```

Environment: ONNX Runtime 1.29.0, `foss/2025b`, X60 `smt.vmadot` (XsmtVdot v1.0)
MLAS backend, `-march=rv64gcv_zvl256b_zfh_zvfh`.

## Takeaways

1. **Prove the code runs before optimizing it.** A one-shot probe would have
   saved the whole kernel-tuning detour.
2. **Roofline early.** STREAM + traffic math killed the bandwidth hypothesis in
   minutes and pointed at "wrong path."
3. **Comments aren't evidence** — `grep` the artifact, not the intent.
4. **Config beats kernels first** (`accuracy_level=4`); then BlkLen must match
   the IME pack path (32 Q4×16 / 128 column-major).
5. **Profile the real model.** Qwen decode was 99.7% MatMulNBits; fp16 attention
   RVV was a red herring until acc4 + BlkLen=128 IME landed.