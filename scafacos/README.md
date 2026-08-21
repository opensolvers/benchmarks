# ScaFaCoS — P3M Coulomb solver FFT A/B

Whole-app FFT probe through [ScaFaCoS](http://www.scafacos.de/) 1.0.4 **P3M**
(Particle–Particle Particle–Mesh). Companion to [`../espresso`](../espresso) /
[`../gromacs`](../gromacs) / [`../fftw`](../fftw): swap double-precision
`libfftw3.so.3` under an unchanged binary via `LD_PRELOAD`.

ScaFaCoS itself is not vendored — load `ScaFaCoS/1.0.4-foss-2025b` from
`dev.eessi.io/riscv`. The EESSI package is library-only (no shipped tests).

## Contents

| File | Purpose |
|---|---|
| `scafacos_bench.c` | lattice of alternating charges → timed `fcs_run` |
| `armci_stubs.c` | tiny stubs so we can link without GlobalArrays (FMM pulls `armci_*`) |
| `Makefile` | links via cmake `SCAFACOS_LIBRARIES` + stubs |
| `run-scafacos-fft-ab.sh` | scalar vs r5v `libfftw3` A/B |

```bash
module load ScaFaCoS/1.0.4-foss-2025b
# needs ~/fftwbuild/{src-scalar,src-r5v}/.libs/libfftw3.so.3.6.10
NP=1 NSIDE=32 REPS=10 bash run-scafacos-fft-ab.sh
```

## Results — Orange Pi RV2 (2026-08-21)

`ScaFaCoS/1.0.4-foss-2025b`, `OPENBLAS_CORETYPE=RISCV64_GENERIC`.

### np=1 (serial FFTW path — cleanest `LD_PRELOAD`)

N=32³=32768, 10 timed runs. Log `~/logs/scafacos-p3m-ab-20260821-151135.log`:

| backend | per-run wall | E (½ Σ q·φ) |
|---|--:|--:|
| scalar `libfftw3` | 5.248 s | −916228.021 |
| r5v (RVV) `libfftw3` | 5.290 s | −916228.024 |
| **speedup** | **0.99×** | Δ ~4e−3 |

Energies match; **no meaningful FFT win** on this P3M size/shape (same lesson as
small-mesh GROMACS/ESPResSo: near-field + bookkeeping dilute the FFT).

### np=4 (mixed — `libfftw3_mpi` stays stock CVMFS)

N=24³=13824, 10 runs. Log `~/logs/scafacos-p3m-ab-20260821-150812.log`:

| backend | per-run wall | E |
|---|--:|--:|
| scalar preload | 0.738 s | −289900.273 |
| r5v preload | 0.820 s | −289900.273 |
| **speedup** | **0.90×** | match |

MPI-distributed P3M uses `libfftw3_mpi` from EESSI (no r5v MPI build on board),
so this row is not a pure FFT A/B — recorded for completeness.

## Build note

CMake `SCAFACOS_LIBRARIES` pulls FMM, which needs GlobalArrays/`libarmci`. The
board module does not ship GA; `armci_stubs.c` satisfies the linker so the **P3M**
path can run. Do not use `method=fmm` with these stubs.
