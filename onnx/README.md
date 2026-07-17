# onnx — int4 `MatMulNBits` on ONNX Runtime, SpaceMiT X60

Benchmark and root-cause writeup for int4 (`MatMulNBits`, 4-bit weights,
`BlkLen=32`) LLM-FFN inference on the SpaceMiT X60 (RISC-V, RVV + IME
`smt.vmadot`) through ONNX Runtime's MLAS `SQNBit` path.

**Headline:** a single missing ONNX node attribute (`accuracy_level=4`) kept ORT
on a generic fp32 dequant+SGEMM fallback instead of the X60 IME int8 kernel
(the same `smt.vmadot` core benchmarked in [`../ime`](../ime)). Setting it cut
latency **~9–10×** with no kernel code change.

## The workload

A Llama-7B-proportioned stack of FFN/projection blocks: `hidden=4096`,
`ffn=11008`, 8 layers, **16 `MatMulNBits` nodes** (com.microsoft), 4-bit
symmetric weights, `block_size=32`. This is where real LLM decode spends its
time — the big 4-bit weight GEMMs. Measured with `onnxruntime_perf_test`
(`-m times`, `-r 8`) at decode shape **M=1**.

## Results

| Configuration        | Before (CompFp32 fallback) | After (IME CompInt8) | Speedup    |
|----------------------|---------------------------:|---------------------:|:----------:|
| Single thread (`-x1`)| 31,956 ms                  | 3,522 ms             | **9.1×**   |
| 8 threads (`-x8`)    | 6,074 ms                   | 590 ms               | **10.3×**  |

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

## Reproduce

```sh
# 1) generate (or patch) the model so MatMulNBits carries accuracy_level=4
python3 patch_accuracy_level.py int4_ffn.onnx int4_ffn_acc4.onnx

# 2) benchmark M=1 decode on the X60 (RVV+IME toolchain)
onnxruntime_perf_test -e cpu -I -m times -r 8 -x 1 int4_ffn_acc4.onnx   # 1 thread
onnxruntime_perf_test -e cpu -I -m times -r 8 -x 8 int4_ffn_acc4.onnx   # 8 threads
```

Environment: ONNX Runtime 1.29.0, `foss/2025b`, X60 `smt.vmadot` (XsmtVdot v1.0)
MLAS backend, `-march=rv64gcv_zvl256b_zfh_zvfh`.

## Takeaways

1. **Prove the code runs before optimizing it.** A one-shot probe would have
   saved the whole kernel-tuning detour.
2. **Roofline early.** STREAM + traffic math killed the bandwidth hypothesis in
   minutes and pointed at "wrong path."
3. **Comments aren't evidence** — `grep` the artifact, not the intent.
4. **Config beats kernels.** One attribute → 9–10×; the hand kernel → −28 %.
