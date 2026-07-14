# ime-bench — reusable int8 (s8s8s32) GEMM core for the SpaceMiT X60 IME

A standalone `int8 × int8 → int32` matrix-multiply microkernel built on the X60
IME instruction `smt.vmadot`, plus a harness that checks it **bit-exactly**
against a scalar reference and times it against a plain **RVV** int8 baseline.

This is the shared core to lift into a framework backend (MLAS / XNNPACK / ruy):
the packing here produces the same `B`-is-`N×K` (pre-transposed) tile layout that
`mmt4d` / ggml-repack feed to the matrix unit.

## What it computes

`C = A · Bᵀ`, i.e. `C[i][j] = Σ_k A[i][k]·B[j][k]`, with `A` = `M×K`, `B` = `N×K`
(both row-major int8) and `C` = `M×N` int32.

## The IME tile

One `smt.vmadot vd, vs1, vs2` does, at VLEN=256 with `vl=32, e8`:

```
C(4×4 int32) += A(4×8 int8) · B(4×8 int8)ᵀ      # M0=4, N0=4, K0=8
```

`vs1`/`vs2` hold row-major 4×8 int8 tiles; `vd` is an even `vd:vd+1` pair holding
the 4×4 int32 accumulator (EMUL=2). We pack A/B into contiguous 32-byte tiles,
accumulate across K, then scatter the result to `C`.

### Register blocking (the main kernel)

One `vmadot` per pair of loads is latency-bound — the store of each 4×4 tile
waits on its accumulate. The main path `ime_block_8x16` instead holds an **8×16
output** in 8 accumulator pairs (`v16..v31`): each K-step loads 2 A-tiles + 4
B-tiles and issues **8 `vmadot`s**, so every load feeds several MACs and the 8
independent accumulators hide the unit's latency. The 8 result tiles are written
**straight to `C`** (vectorized, no scalar scratch copy). The plain 4×4
`ime_tile` remains for the M/N block edges. An L2 panel loop keeps the reused
B-panel (~128 KB) resident in the 512 KB cluster L2 across the M sweep.

| File | Role | Portable? |
|---|---|---|
| `gemm_ref.c` | scalar reference + **shared packing** + packed scalar path | yes |
| `gemm_ime.c` | `smt.vmadot` kernels: register-blocked 8×16 + 4×4 edge | X60 only |
| `gemm_rvv.c` | RVV int8 baseline (`vwmul`+widening reduce) | X60 only |
| `bench.c` | fill / cross-check / GOP-s timing (+ peak mode) | yes |

Because the packing and tile-index math live in the **portable** `gemm_ref.c`
(`gemm_packed_ref` uses the exact same layout as `gemm_ime`), everything except
the two asm inner loops is verifiable off-target.

## Build & run

The IME kernel emits `smt.vmadot` as a **raw instruction word** (`.insn 4,
0xe200302b|…`, letting the assembler compute the register fields), so it needs no
`xsmtvdot`-aware toolchain — any RVV-capable binutils builds it. That is what
lets `make board` work on the RV2's stock **binutils 2.42**.

```bash
# Portable self-test (x86, or any host): verifies packing + harness, no RISC-V.
make host && ./ime-bench-host 256 256 128

# X60 build: scalar + RVV baseline + IME kernels. Pin to an IME-capable core.
make board && taskset -c 0 ./ime-bench 512 512 512

# X60 peak mode: 4th arg = rep count. The scalar oracle runs once for the
# bit-exact check, then N back-to-back IME reps report min/median/max GOP/s.
taskset -c 0 ./ime-bench 768 768 512 50

# X60, RVV baseline only (skips the IME kernel entirely):
make board-rvv && taskset -c 0 ./ime-bench
```

Dims must satisfy `M%4=0, N%4=0, K%8=0`. On the K1/M1 the IME lives on one 4-core
cluster — pin to a core in it (`taskset -c 0`; cluster 0 = cores 0-3).

## Results (measured — Orange Pi RV2 / X60, single core 0, 1.6 GHz, performance gov)

Built with `make board` (stock binutils 2.42, raw `.insn`). Every path is `ok`,
bit-exact vs the scalar reference at every size tested.

Throughput is **single-core GOP/s** (`2·M·N·K / t`). The board is multi-tenant
and cores 0-3 share one 512 KB L2, so a neighbouring core polluting L2 mid-run
adds noise. The figures below are the **clean-layout peak** — the max over ≥5
process launches, each internally 15-50 reps (contention only ever lowers a run,
so the max is the true single-core capability). See the aliasing note for the
spread.

| M×N×K | scalar | RVV int8 | **IME (8×16 blocked)** | IME/RVV | prev 4×4-only |
|---|--:|--:|--:|--:|--:|
| 256×256×256   | 0.39 | 5.05 | **33** | 6.5× | 22.0 |
| 512×512×512   | 0.39 | 5.22 | **39** | 7.5× | 25.8 |
| 768×768×512   | 0.39 | ~5.2 | **42** | 8.1× | —    |
| 1024×1024×512 | 0.39 | 5.19 | **32** | 6.2× | 26.7 |
| 2048×2048×512 | 0.39 | ~5.2 | **34** | 6.5× | —    |

GOP/s. Register blocking (8 accumulators, straight-to-`C` stores) lifts the IME
kernel **+20–50 %** over the earlier single-4×4-tile path and pushes the
IME-vs-RVV ratio from ~5× to **~7–8×** — peak **42 GOP/s at 768³**, ~108× the
scalar reference. The clean peak crests at 768³ (~42) then settles to ~32–34 at
1024³–2048³ as the working set outgrows the 512 KB L2 (at 1024³ the `C` matrix
alone is 4 MB): the large sizes are turning memory-bound, the main tuning lead
from here.

### Cache-set aliasing (buffer placement)

Every size here runs **bimodal**: some launches hit the peak above, others sit
~35 % lower (512³: 25 vs 39; 768³: 27 vs 42; 1024³: 22 vs 32; 2048³: 24 vs 34).
The mode is *fixed within a process* (all reps in one launch agree to ±3 %) but
*varies between launches* — a `malloc`-placement effect, not thermal/DVFS (the
governor is `performance`, pinned at 1.6 GHz). The L2 is **16-way, 512 sets, 64 B
lines → a 32 KB way-span** (two addresses share a set when congruent mod 32 KB).
At K=512 every buffer (`A`,`B` = M·512 / N·512 B; `C` = M·N·4 B) is an exact
multiple of 32 KB whenever M,N are multiples of 64 — true for **all** sizes here,
not just the powers of two — so when `malloc` lays the packed-A / packed-B / C
streams 32 KB-congruent they map to the same 16 ways and thrash. 768³ aliases
just like 512³; the effect is placement- not power-of-two-specific. The share of
"bad" launches drifts with heap and board state (~⅓–½ observed). A production
backend pads leading dimensions so the streams can't stay congruent; this
prototype leaves them unpadded.

## Comparison with llama.cpp's shipping IME kernel

llama.cpp already ships a SpaceMiT IME backend (`ggml-cpu/spacemit`,
`GGML_CPU_RISCV64_SPACEMIT`). Its kernel is a *block-scaled* Q4_0 GEMM
(`spacemit_kernels::ime1::gemm_kernel_i8i4`: int8 activations × 4-bit weights,
per-32-block fp16 scale, fp32 output) — not the pure `s8s8s32` this benchmark
measures. [`bench_i8i4.cpp`](bench_i8i4.cpp) links that kernel straight out of a
prebuilt `libggml-cpu.so` and times it the same way, on the same X60 core.
(Throughput is data-independent; the rate cross-validates against `llama-bench`
pp512 ≈ 20 GOP/s/thread. The harness paths are board-specific.)

**(A) The block-scale tax.** The `i8i4` kernel tops out at **~28 GOP/s ≈
0.6–0.7× our raw-int8 ceiling** — the per-block int32→fp convert + scale is real
work `s8s8s32` never does:

| M×N×K | llama.cpp `i8i4` | this `s8s8s32` | ratio |
|---|--:|--:|--:|
| 512×512×512   | 22.8 | 39 | 0.58 |
| 768×768×512   | 27.9 | 42 | 0.66 |
| 1024×1024×512 | 20.6 | 32 | 0.64 |
| 2048×2048×512 | 20.5 | 34 | 0.60 |

**(B) Aliasing — and a partial fix.** Unlike ours, the `i8i4` kernel is *not*
randomly bimodal (its 16×16 streaming tiles dodge the `malloc`-placement
lottery), but it pays a **deterministic power-of-two penalty on the C
write-stream**. Padding the output leading dimension by one tile (`ldc=N+16`)
recovers it — **+21 % at 512³** even after the scratch→dst copy — but the gain
**collapses as K grows** (+8 % at K=1024, +2 % at K=2048), because the aliasing
is a per-output C-write cost that large-K GEMMs amortise away. Real transformer
weight-matmuls have large K, so the practical payoff is low single digits (and
zero on non-power-of-two dims). Real and free, but not a broad IME speed-up.

**End-to-end** (`llama-bench`, Qwen2.5-0.5B Q4_0, this X60):

| build / threads | pp512 (t/s) | tg128 (t/s) |
|---|--:|--:|
| IME1 `-t4` (cluster 0) | 79.0 | 7.54 |
| IME1 `-t8`             | 56.7 | 3.71 |
| RVV  `-t8`             | 90.7 | 11.16 |

IME is **1.51× RVV at `-t4`** on prompt processing but is confined to the 4-core
cluster 0 — it *regresses* at `-t8`, and well-threaded RVV wins overall; IME
never helps decode. **Takeaway:** the real IME headroom is the ~40 % block-scale
gap in the kernel's fp path (needs a kernel rewrite, not a driver tweak), and
even closing it is capped by the cluster-0 constraint on this part.

## Cutting the block-scale tax (kernel optimization)

Following up on that "needs a kernel rewrite": the tax is **measurable and
reducible**. Rebuilding `libggml-cpu.so` from a patched `ime1_kernels.cpp` and
A/B-benchmarking on the X60 — correctness gated bit-for-bit by
[`bench_i8i4.cpp`](bench_i8i4.cpp)'s `chk` mode (a sum/sumsq/max signature vs the
stock kernel on identical varied inputs, since throughput can't catch a wrong
rewrite):

**Where the tax lives** (strip parts of the per-block FP reduction, M4 no-zp path):

| M×N×K | stock | −scale-build | −all FP (vmadot only) |
|---|--:|--:|--:|
| 512³ | 22.5 | 29.6 | 32.7 |
| 768³ | 27.5 | 36.7 | 41.3 |
| 512×512×2048 | 27.5 | 38.7 | 43.6 |

The ~31–37 % FP tax is **dominated by the per-block scale build**
(`LOAD_SCALE_4x16_FP16`: 16 masked `vfmul.vf` + widening + copies), not the
`vfcvt`+`vfmacc` convert/accumulate (~10 %). Removal scales ≈ linearly with
op-count, so the X60 does **not** overlap IME (`smt.vmadot`) with RVV FP — cutting
FP op-count cuts time.

**The fix** ([`llama-ime1-scalebuild-opt.patch`](llama-ime1-scalebuild-opt.patch)):
rebuild the combined `A·B` scale with **8 `vfmul.vv`** against two prebuilt
row-scale vectors instead of **16 split/masked `vfmul.vf` + 4 `vmv`** (also fewer
`vsetvli` toggles). Identical `v8..v15` layout → **bit-identical** output.

| kernel path | used by | 512³ | 768³ |
|---|---|--:|--:|
| M4 no-zero-point | Q4_0 prefill | 23.0→24.1 (+4.8 %) | 27.6→29.2 (+5.8 %) |
| M4 zero-point | Q4_1-style prefill | 20.6→21.7 (+5.3 %) | 24.7→26.0 (+5.3 %) |
| M1 / GEMV | decode (`tg`) | 12.1 — N/A | — |

**~5 % on both prefill GEMM paths, verified bit-exact.** M1/GEMV has no equivalent
tax (its single-row scale build is already 4 `vfmul.vf`, no masking) and is
memory-bound. The residual block-scale cost is fundamental here: the bigger lever
— software-pipelining the FP reduction under `vmadot` — is register-blocked (the
fp output, int accumulator, and scale vectors already fill the register file), and
IME stays cluster-0-capped end-to-end regardless.

## Caveats

- **qemu-user cannot run this.** It does not emulate the SpaceMiT custom
  `vmadot` — the IME path only executes on real X60 silicon.
- The RVV baseline is correct and representative but **not** cache-tuned (it
  reloads B rows); it is the honest "plain RVV int8" floor the IME kernel beats,
  not a state-of-the-art RVV GEMM.
- int32 accumulator is exact for full int8 while `K ≲ 133000`.
- Dims must be multiples of the tile (`M%4=N%4=0`, `K%8=0`): the 8×16 kernel
  handles M/N *block* edges via the 4×4 `ime_tile`, but there is no sub-tile
  remainder path. Large-N throughput is L2-bound — the next tuning target.

SPDX-License-Identifier: MIT
