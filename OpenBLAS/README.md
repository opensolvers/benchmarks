# OpenBLAS on RISC-V - verification & microbenchmarks

Small, self-contained programs used to check the **performance** and
**numerical correctness** of a BLAS backend, and - when something is wrong - to
localize *which* kernel is broken. Written while bringing up the RISC-V RVV
(SpaceMiT X60 / K1) OpenBLAS build, which surfaced two distinct kernel bugs: a
`gemv_n` vector kernel that silently returned `NaN`, and a set of `_rvv_v1`
TRSM kernels that were not VLEN-agnostic and corrupted results on non-`x280`
targets.

| File | What it does |
|---|---|
| `bench_dgemm.c` | Times square `C = A*B` (level-3 `dgemm`, 3 reps), reports GFLOP/s, prints `C[0]` so garbage is visible. |
| `difftest.c` | `dlopen`s a BLAS `.so` and runs level-1/2/3 routines (incl. `dgemv`, `dgemm`, `dtrsm`) on fixed inputs, printing `sum / sumsq / nan / inf` per routine for differential comparison. |
| `verify_ctrsm.c` | Full-parameter TRSM correctness sweep ({Left,Right} x {Upper,Lower} x {No,T,ConjT} x {Non,Unit} x size grid), recomputes `op(A)*X` and checks max residual. Shown for the complex-single (`ctrsm`) case; S/D/Z are structurally identical (swap type + `cblas_?trsm` + element generator). |

All are single C files, MIT-licensed. `bench_dgemm` / `difftest` select the
backend at runtime via FlexiBLAS (`FLEXIBLAS=...`) or OpenBLAS
(`OPENBLAS_CORETYPE`, `OPENBLAS_NUM_THREADS`) - no recompilation needed to A/B
two backends.

## Build

```bash
gcc -O2 bench_dgemm.c   -o bench_dgemm   -lflexiblas   # or -lopenblas
gcc -O2 difftest.c      -o difftest      -ldl -lm
gcc -O2 verify_ctrsm.c  -o verify_ctrsm  -I<cblas_include> -lopenblas -lm
```

## Run

```bash
# performance (pick threads + backend via env)
OPENBLAS_NUM_THREADS=8 ./bench_dgemm 4096

# correctness of a specific backend .so
./difftest /path/to/libopenblas.so

# TRSM correctness sweep (exit 0 = all pass)
./verify_ctrsm
```

## A/B a backend with FlexiBLAS (no rebuild)

```bash
# scalar baseline
OPENBLAS_CORETYPE=RISCV64_GENERIC ./bench_dgemm 2048
OPENBLAS_CORETYPE=RISCV64_GENERIC ./difftest  /path/to/libopenblas.so
# vector / alternative backend
FLEXIBLAS=/path/to/libopenblas.so ./bench_dgemm 2048
./difftest                        /path/to/vector-libopenblas.so
```

## Example - the OpenBLAS 0.3.30 RISC-V `gemv_n` NaN bug (SpaceMiT X60)

`difftest` run against the **stock** EESSI-CVMFS OpenBLAS 0.3.30 (a
`DYNAMIC_ARCH` build that dispatches the `ZVL256B` RVV kernels on the X60), the
same lib forced to scalar, and a **patched** RVV build:

| backend / dispatch | `dgemv` nan | `dgemm` nan | `dtrsm` nan | `dgemv` sum |
|---|--:|--:|--:|---|
| stock CVMFS, **default RVV** (`ZVL256B`) | **192** | 0 | 0 | 198.94 (wrong) |
| stock CVMFS, forced scalar (`RISCV64_GENERIC`) | 0 | 0 | 0 | 42.06549 (reference) |
| **patched RVV** (`gemv_n` fix) | 0 | 0 | 0 | 42.06549 (matches reference) |

The differential test pins the fault to `dgemv` alone - `dgemm` and `dtrsm` are
correct even on the broken RVV build. That is exactly why a plain `dgemm`
benchmark looks fine while **HPL fails with a NaN residual**: HPL's panel
factorization leans on `dgemv`. The patched vector build reproduces the scalar
reference bit-for-bit.

Performance on the same machine (`bench_dgemm`, 1 core, N=2048, via FlexiBLAS):

| backend | GFLOP/s | `C[0]` |
|---|--:|--:|
| scalar (`RISCV64_GENERIC`) | 1.16 | 245.24 |
| patched RVV (`ZVL256B`) | 2.62 | 245.24 |

i.e. ~2.3x faster and numerically identical. (Root cause: OpenBLAS 0.3.30
`kernel/riscv64/gemv_n_vector.c` zeroes an *uninitialized* vector accumulator;
fixed upstream after 0.3.30.)

## Cross-board confirmation - Banana Pi BPI-F3 (same K1 / X60 SoC)

The [Banana Pi BPI-F3](https://www.banana-pi.org/) uses the same SpaceMiT K1
(8x X60 @ 1.6 GHz, RVV 1.0 VLEN=256) as the Orange Pi RV2 above, so the same
bug and fix reproduce on a second board - here on EESSI `2025.06-001`, patched
OpenBLAS 0.3.30 ([easyconfigs #26444](https://github.com/easybuilders/easybuild-easyconfigs/pull/26444))
via FlexiBLAS.

`difftest` - bit-identical to the RV2:

| backend / dispatch | `dgemv` nan | `dgemm` nan | `dtrsm` nan | `dgemv` sum |
|---|--:|--:|--:|---|
| stock CVMFS, **default RVV** (`ZVL256B`) | **192** | 0 | 0 | 198.94 (wrong) |
| stock CVMFS, forced scalar (`RISCV64_GENERIC`) | 0 | 0 | 0 | 42.06549 (reference) |
| **patched RVV** (`gemv_n` fix) | 0 | 0 | 0 | 42.06549 (matches reference) |

`bench_dgemm` (1 core, N=2048, via FlexiBLAS):

| backend | GFLOP/s | `C[0]` |
|---|--:|--:|
| scalar (`RISCV64_GENERIC`) | 1.26 | 245.24 |
| patched RVV (`ZVL256B`) | 2.96 | 245.24 |

~2.35x faster and numerically identical - confirming the RV2 result on a second
K1 board. (Threaded: **17.71 GFLOP/s** at 8 cores, N=4096.)

## Example - the RVV `_rvv_v1` TRSM VLEN bug (OpenMathLib/OpenBLAS#5928)

A second, independent kernel bug on the same X60 build: the RVV triangular-solve
kernels `trsm_kernel_{LN,LT,RN,RT}_rvv_v1.c` are **not VLEN-agnostic**. They tile
the packed `A` panel by the *runtime* `VSETVL_MAX` (16 fp32 elements at LMUL=2 on
the X60's VLEN=256), but the GEMM driver they link against packs `A` by the
*compile-time* `GEMM_UNROLL_M`. Those two agree only when `GEMM_UNROLL_M == 16`
(the `x280` / `ZVL256B` target). On any build where `GEMM_UNROLL_M = 8`
(`ZVL128B`, the default for a plain RVV target) the tile stride `16 != 8`
diverges from how the panel was packed, and TRSM silently returns wrong results.

`verify_ctrsm` (and its S/D/Z siblings) run the full parameter sweep. On a
`ZVL128B` build (`GEMM_UNROLL_M = 8`):

| build | STRSM | DTRSM | CTRSM | ZTRSM | total |
|---|--:|--:|--:|--:|--:|
| stock `_rvv_v1` (pre-fix) | fails | **408 / 1600 fail** | fails | fails | broken |
| **fixed** (`GEMM_UNROLL_M`-tiled) | 0 fail | 0 fail | 0 fail | 0 fail | **8000 / 0** |

worst residuals on the fixed build: STRSM `1.4e-7`, DTRSM `3.3e-16`,
CTRSM `3.0e-7`, ZTRSM `5.5e-16`. The fix also passes `8000 / 0` on a `ZVL256B`
build (`GEMM_UNROLL_M = 16`), confirming it is genuinely VLEN-agnostic rather
than re-hardcoded to the other width.

**The fix:** replace the `VSETVL_MAX` outer tiling with generic
`GEMM_UNROLL_M`-tiled outer control plus a power-of-2 `M`-remainder path, keeping
the vectorized RVV `solve()` inner kernel unchanged.

### Performance - no regression

`benchmark/trsm.c` (from the OpenBLAS tree), single-thread, `L/U/N/N`,
`OPENBLAS_LOOPS=20`, best-of-3 on the X60, both libs `ZVL128B`. Delta = fixed vs
original `_rvv_v1`, MFlops:

| size | STRSM | DTRSM | CTRSM | ZTRSM |
|---|--:|--:|--:|--:|
| 8   | +4.3%  | +4.3% | -3.0%  | +0.3% |
| 16  | +32.6% | +0.2% | +31.9% | +2.6% |
| 32  | +40.9% | +0.4% | +29.2% | +1.6% |
| 64  | +35.6% | +1.4% | +21.4% | +1.5% |
| 128 | +26.0% | ~0%   | +8.6%  | -2.6% |
| 256 | +17.3% | -3.9% | +6.6%  | -4.2% |
| 400 | +11.5% | -6.9% | -0.8%  | -1.5% |

- **STRSM / CTRSM**: consistent speedups (+10-40%), largest at small/mid sizes -
  `GEMM_UNROLL_M` tiling reuses the packed diagonal block better than per-call
  `VSETVL_MAX` retiling.
- **DTRSM / ZTRSM**: within measurement noise (+-~4%) on this loaded board; a
  6-rep re-measure at DTRSM `m=128` gave overlapping distributions (orig
  1273-1412, fixed 1257-1371; equal medians), so the scattered negatives are
  jitter, not a systematic slowdown.

> **Caveat on the baseline:** on `ZVL128B` the original `_rvv_v1` kernels are
> *numerically broken* (that is the bug), so their timings are not a legitimate
> speed baseline - you cannot "regress" against a kernel that returns garbage.
> The fix makes `ZVL128B` correct for the first time while being at least as
> fast, usually faster. On `ZVL256B` / `x280` (`VSETVL_MAX == GEMM_UNROLL_M ==
> 16`) the tile width is unchanged, so the rewrite is a structural no-op for
> throughput there.
