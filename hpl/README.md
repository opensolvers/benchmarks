# HPL (High-Performance Linpack) - BLAS backend A/B

Configs and a runner to A/B a BLAS backend end-to-end with HPL, by swapping only
the BLAS implementation (via FlexiBLAS) under one unchanged `xhpl` binary. HPL
itself is not vendored here - it comes from your HPL module (e.g.
`HPL/2.3-foss-2025b`); this directory ships the `.dat` configs, the A/B runner,
and the expected results.

## Contents

| File | Grid | N | Purpose |
|---|---|---|---|
| `HPL.dat` | 1x8 | 8000 | quick A/B (~1 min/run) |
| `HPL_big.dat` | 1x8 | 28672 | large run (needs ~6.6 GB RAM) |
| `HPL-sweep.dat` | 2x4 | 20000 | near-peak, squarer grid |
| `run-hpl-ab.sh` | - | - | scalar-vs-RVV A/B driver |

All use NB=256, pure-MPI (`-np 8`, 1 thread/rank).

## Run

```bash
module load HPL/2.3-foss-2025b
# scalar only:
./run-hpl-ab.sh HPL.dat
# scalar + vector (swap in any OpenBLAS .so as the backend):
RVV_LIB=/path/to/vector/libopenblas.so ./run-hpl-ab.sh HPL.dat
```

The runner copies the chosen config to `./HPL.dat` (what `xhpl` reads), runs the
scalar baseline (`OPENBLAS_CORETYPE=RISCV64_GENERIC`), then - if `RVV_LIB` is set
- the vector backend (`FLEXIBLAS=$RVV_LIB`), and greps the result/PASSED line.

## Example results - RISC-V SpaceMiT X60 (Orange Pi RV2, K1, 8x X60 @ ~1.6 GHz)

Same `xhpl`, BLAS backend swapped via FlexiBLAS:

| config | scalar (`RISCV64_GENERIC`) | vector RVV (`ZVL256B`) | speedup |
|---|--:|--:|--:|
| `HPL.dat` (N=8000, 1x8) | 6.41 GFLOP/s | 11.55 GFLOP/s | **1.80x** |
| `HPL_big.dat` (N=28672, 1x8) | 7.38 GFLOP/s | 13.41 GFLOP/s | **1.82x** |

All PASSED (residual well under threshold).

### Correctness: the vector backend must be a *fixed* one

The stock EESSI-CVMFS OpenBLAS 0.3.30 is a `DYNAMIC_ARCH` build that dispatches
the `ZVL256B` RVV kernels on the X60 - and its RVV `gemv_n` has a NaN bug, so the
**stock vector HPL fails**:

```
... nan ... FAILED
```

Swapping in an OpenBLAS with the `gemv_n` fix backported gives residual
4.04e-03, **PASSED**. See [`../OpenBLAS/`](../OpenBLAS) for the differential test that
isolated the bug to `dgemv` (HPL's panel factorization leans on `dgemv`, which is
why HPL NaNs while a plain `dgemm` benchmark looks fine).

### Process-grid shape matters

Near peak, a squarer grid beats `1x8`: on the fixed build, `HPL-sweep.dat`
(N=20000, `2x4`) reached ~10.5 GFLOP/s vs ~9.7 at `1x8`. Keep the matrix within
RAM - N=28672 needs ~6.6 GB, and an 8 GB board with no swap OOMs above ~N=25000.

## Cross-board confirmation - Banana Pi BPI-F3 (same K1 / X60 SoC)

Same `xhpl`, `HPL.dat` (N=8000, `1x8`), EESSI `2025.06-001` on a
[Banana Pi BPI-F3](https://www.banana-pi.org/) (SpaceMiT K1, 8x X60 @ 1.6 GHz):

| backend | GFLOP/s | residual | result |
|---|--:|---|---|
| scalar (`RISCV64_GENERIC`) | 6.52 | 4.63e-03 | PASSED |
| patched RVV (`ZVL256B`, `gemv_n` fix) | 11.52 | 4.04e-03 | PASSED |
| stock RVV (`ZVL256B`, unpatched) | 11.64 | **nan** | **FAILED** |

**1.77x** scalar->vector, matching the RV2's 1.80x; the patched residual
(`4.04326757e-03`) is bit-identical to the RV2. The stock vector run is "fast"
(11.64 GFLOP/s) but NaN - correctness needs the fixed build, exactly as on the
RV2. `HPL_big.dat` (N=28672, ~6.6 GB) and `HPL-sweep.dat` (N=20000, ~3.2 GB) were
skipped here: they exceed this BPI-F3's 3.7 GB RAM.
