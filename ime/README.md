# IME1 scale-build prefill optimization (SpaceMiT X60 / RISC-V)

An assembly-level optimization to the SpaceMiT **IME1** `int8×int4` Q4_0 GEMM
microkernel in llama.cpp (`ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp`),
measured on an Orange Pi RV2 (SpaceMiT X60, 8× cluster, RVV 1.0, 1.6 GHz,
`performance` governor).

## TL;DR

The patch replaces the scale-combine sequence (`LOAD_SCALE_4x16_FP16`) at the two
`gemm_kernel_i8i4` block sites with a vectorized variant (`LOAD_SCALE_4x16_FP16_OPT`)
that builds the combined `As[row] × Bs[col]` fp16→fp32 scale matrix with
`vfmul.vv` across lanes instead of the per-lane `vfmul.vf` chain.

- **+4.3% pp512 microkernel throughput**, bit-exact output.
- Effect is ~9× the board's round-to-round noise; **zero distribution overlap**
  over 30 interleaved A/B rounds.

## What changed

`llama-ime1-scalebuild-opt.patch`:
- Adds `LOAD_SCALE_4x16_FP16_OPT` — loads the four fp16 A-scale vectors + the two
  B-scale lanes, widens with `vfwcvt.f.f.v`, and produces the 8-vector combined
  scale (`v8..v15`) via `vfmul.vv` rather than eight `vfmul.vf` ops.
- Redirects both `LOAD_SCALE_4x16_FP16` call sites in `gemm_kernel_i8i4` (the two
  block-inner tails) to the `_OPT` macro. Exactly **2 call sites**, so the patch
  applies to a clean upstream tree with zero porting.

## Build

Built with EasyBuild on EESSI (`2025.06-001`), GCC from EESSI, on top of upstream
llama.cpp `ad8d821` (`foss-2025b`). Two patches are layered:

1. `llama.cpp-x60-ime-upstream-binutils.patch` — gates `ime2_kernels.cpp` behind
   `RISCV64_SPACEMIT_IME2` (the X60 is IME1-only), renames `vmadot`→`smt.vmadot`
   and injects `.option arch,+xsmtvdot` so the SpaceMiT IME1 opcode assembles with
   the `xsmtvdot` binutils (2.46.1) via the `-B.../xsmtvdot-as-only/` symlink dir.
2. `llama-ime1-scalebuild-opt.patch` — the optimization above.

The two EasyBuild modules form the canonical A/B pair:
- `llama.cpp/ad8d821-foss-2025b-x60-ime` — **stock** (arm A)
- `llama.cpp/ad8d821-foss-2025b-x60-ime-scaleopt` — **patched** (arm B)

The `.so`s are byte-different (distinct sha256; `cmp` diverges at byte 153),
confirming the patch is present in the binary and not a no-op.

## Verification

### Correctness — bit-exact

`bench_i8i4 M N K 0 0 1` drives `ime1::gemm_kernel_i8i4` through the exact
no-TCM tiled call path with deterministic, quantizer-bypassing inputs and prints a
`sum / sumsq / max` signature. Stock and scaleopt produce **identical signatures**
across three shapes:

| shape (M×N×K) | sum | sumsq | max |
|---|---|---|---|
| 512×512×512  | -5.614377e+07 | 1.362898e+11 | 1.056097e+04 |
| 128×512×4096 | -1.109233e+08 | 4.374616e+11 | 1.931147e+04 |
| 16×4096×512  | -1.235156e+07 | 2.914980e+10 | 1.099525e+04 |

Different binaries, identical numerics → the patch changes the compute path but
not the result.

### Performance — isolated interleaved A/B

30 rounds, each running **both** variants back-to-back (swapping the active
`libggml-cpu.so`) so slow thermal/frequency drift cancels in the per-round paired
delta. Pinned to cluster-0 (`taskset -c 0-3`), `performance` governor @ 1.6 GHz,
idle board, pp512 shape (M=N=K=512), median of 25 inner reps per invocation.

| | stock | scaleopt |
|---|---|---|
| median GOP/s | 22.50 | 23.47 |
| sd | 0.077 | 0.097 |
| range | 22.4 – 22.7 | 23.2 – 23.6 |

- Paired delta (scaleopt − stock): **+0.97 GOP/s (+4.3%)**, sd 0.11, paired t = 50.3 (df=29).
- scaleopt faster in **30/30** rounds; **zero overlap** (max stock 22.7 < min scaleopt 23.2).

The board noise floor (sd ≈ 0.08–0.10 GOP/s, ~0.4%) is an order of magnitude
below the effect, so the win is real, not measurement artifact.

## Files

- `llama-ime1-scalebuild-opt.patch` — the optimization patch.
- `llama.cpp-x60-ime-upstream-binutils.patch` — IME1 assembler/build prerequisite.
- `llama.cpp-ad8d821-foss-2025b-x60-ime-scaleopt.eb` — EasyBuild recipe (patched build).
- `bench_i8i4.cpp` — microkernel correctness + throughput harness.
- `ab_pp512.sh` — interleaved A/B driver.
- `ab_pp512.tsv` — raw 30-round per-round medians.
