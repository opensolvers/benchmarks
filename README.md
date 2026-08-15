# benchmarks — RISC-V (SpaceMiT X60 / K1) HPC kernel & application A/Bs

A collection of small, self-contained benchmarks and A/B harnesses used while
bringing up the HPC / AI software stack on the **SpaceMiT X60 (K1)** RISC-V CPU
(Orange Pi RV2 / BPI-F3, RVV 1.0, VLEN=256), under the
[EESSI](https://www.eessi.io/) software stack.

Each directory is independent, ships its own `README.md` with the full
build/run/replicate steps, and follows one guiding principle:

> **Change exactly one variable, hold everything else constant, and verify
> correctness before trusting any speed number.**

Most benchmarks are *backend A/Bs* — one unchanged binary, one swapped
implementation (BLAS via FlexiBLAS, or FFT via `LD_PRELOAD`) — so a measured
delta is attributable to that single backend and nothing else. Numerical
correctness (finite / bit-identical results) is checked in every case.

Where both boards share the same K1 / X60 silicon, many directories also include a
**Banana Pi BPI-F3 cross-board confirmation** (3.7 GB RAM) next to the Orange Pi
RV2 numbers.

## What's here

| Dir | What it measures | Axis swapped | Kind |
|---|---|---|---|
| [`cores/`](cores) | X60 and SiFive U74 ISA / codegen notes (what the silicon implements, how to target it) | — | reference |
| [`gcc-14.3/`](gcc-14.3) | GCC 14.3 EasyBuild SpacemiT X60 patch + RV2 mtune A/Bs (canaries, OpenBLAS DGEMM, HPL) | **mtune** (`spacemit-x60` vs `generic-ooo`) | patch + results |
| [`gcc-15.2/`](gcc-15.2) | GCC 15.2 EasyBuild SpacemiT X60 patch + RV2 mtune A/Bs (canaries, OpenBLAS DGEMM, HPL) | **mtune** (`spacemit-x60` vs `generic-ooo`) | patch + results |
| [`OpenBLAS/`](OpenBLAS) | OpenBLAS on RISC-V: DGEMM + differential correctness + TRSM sweep; localizes two broken RVV kernels (`gemv_n` NaN, `_rvv_v1` TRSM VLEN bug) | BLAS | microbench + verification |
| [`BLIS/`](BLIS) | BLIS (FLAME) RVV build + DGEMM A/B vs OpenBLAS (RV2 + F3) | BLAS | microbench + verification |
| [`numpy/`](numpy) | BLAS/LAPACK backend as seen through NumPy/SciPy | BLAS | application proxy |
| [`hpl/`](hpl) | High-Performance Linpack, end-to-end (incl. HPL-on-BLIS) | BLAS (FlexiBLAS / static BLIS) | application A/B |
| [`elpa/`](elpa) | Dense real-symmetric eigensolver (ELPA, 1-stage) | BLAS | microbench |
| [`scalapack/`](scalapack) | Distributed dense eigensolver (ScaLAPACK `PDSYEV`, pure-MPI) | BLAS | microbench (MPI) |
| [`petsc/`](petsc) | PETSc Jacobi-CG + dense MatMult/CG + MUMPS/SuperLU/UMFPACK FlexiBLAS A/B | BLAS | dense **~1.70×**; stock RVV NaN on dense/direct |
| [`qe/`](qe) | Quantum ESPRESSO `pw.x` plane-wave DFT SCF, end-to-end | **BLAS** (FlexiBLAS) | real-application A/B |
| [`fftw/`](fftw) | FFTW 3.3.10 RVV (`r5v`) vs scalar, **and** the FFT axis inside a QE SCF (RV2 + F3) | **FFT** (`--enable-r5v`, `LD_PRELOAD`) | microbench + real-application A/B |
| [`gromacs/`](gromacs) | GROMACS `mdrun` PME: FFT-axis A/B + hand-written RVV `Force` backend (RV2 + F3 FFT) | **FFT** / **SIMD Force** | real-application A/B |
| [`kokkos/`](kokkos) | Kokkos (via LAMMPS) on X60: OpenMP/Serial, no RVV SIMD abi; Pair hot path; hand-RVV LJ + EAM results | Pair / RVV | learnings + results |
| [`lammps/`](lammps) | RVV-Kokkos whole-app MD (5 upstream benches × serial/Kokkos/MPI, RV2 + F3) + hand RVV `lj/cut` / `eam` plugins | Pair / parallel back-end | whole-app + plugin A/B |
| [`openfoam/`](openfoam) | OpenFOAM v2506 motorBike / `simpleFoam`: auto-vec + hand RVV Amul / GS A/B (RV2) | sparse Amul / GS | real-application A/B (negative) |
| [`walberla/`](walberla) | waLBerla 7.2: BasicLBM ISA A/B, HeatEquation / UniformGrid auto-vec, hand RVV simd + SoA microbenches (RV2) | ISA / auto-vec / simd | HeatEq **1.64×**; UG collide **1.54×**; hand simd loses |
| [`ime/`](ime) | int8 (`s8s8s32`) GEMM on X60 **IME** (`smt.vmadot`) vs RVV (RV2 + F3) | int8 kernel | microkernel + verification |
| [`llamacpp/`](llamacpp) | llama.cpp Q4_0 / Q4_K_M end-to-end: IME vs RVV (model validation + m1gemv study) | IME / RVV | application A/B |
| [`onnx/`](onnx) | int4 `MatMulNBits` LLM-FFN inference via ONNX Runtime MLAS | int4 kernel | application + root-cause writeup |
| [`gpu/`](gpu) | PowerVR BXE-2-32 GPU compute: vendor stack closed + deferred open Mesa `pvr` route | GPU | characterization / negative result |
| [`papers/`](papers) | Longer-form writeups (e.g. X60 IME block-scale optimization) | — | prose / PDF |
| [`todo.md`](todo.md) | Tracking list for remaining RISC-V bring-up work | — | planning |

### Two views of the same question

Several directories deliberately pair a **microbenchmark** with a
**real-application** measurement of the *same* backend, because they often
disagree — and that disagreement is the interesting result:

- **BLAS axis:** [`OpenBLAS`](OpenBLAS)/[`BLIS`](BLIS)/[`numpy`](numpy)/[`elpa`](elpa) (kernel level)
  → [`hpl`](hpl)/[`qe`](qe) (whole application).
- **FFT axis:** [`fftw`](fftw) standalone microbench (RVV wins **1.06–1.60×**)
  → the same RVV FFTW dropped into a Quantum ESPRESSO SCF (**~0% end-to-end**,
  documented in [`fftw/README.md`](fftw/README.md)), and into a
  [`gromacs`](gromacs) PME MD run (RVV wins **1.14–1.23×** on the isolated
  `PME 3D-FFT`, but that step is a small fraction of a scalar-`Force`-dominated
  run). Hand-vectorizing `Force` moves the whole app (**~3.31×** on RV2).
- **int8/int4 axis:** [`ime`](ime) microkernel → [`onnx`](onnx) / [`llamacpp`](llamacpp)
  real inference.
- **MD Pair axis:** stock Kokkos Pair (portable OpenMP, no RVV SIMD abi) → hand
  RVV plugins in [`lammps/rvv-lj`](lammps/rvv-lj) / [`lammps/rvv-eam`](lammps/rvv-eam)
  and learnings in [`kokkos`](kokkos).

## How to use / replicate

There is no top-level build. **Each directory is standalone** — `cd` into the
one you want and follow its `README.md`, which lists the exact modules, build
command, run command, and expected output.

Common ground for reproducing any of these:

1. **Hardware:** a SpaceMiT X60 / K1 board — **Orange Pi RV2** (8 GB) or
   **Banana Pi BPI-F3** (3.7 GB). Primary numbers are usually from the RV2;
   F3 sections confirm the same SoC (and note RAM limits for large HPL configs).
2. **Toolchain:** the [EESSI](https://www.eessi.io/) 2025.06 stack (or the
   [`dev.eessi.io/riscv`](https://www.eessi.io/docs/repositories/dev.eessi.io-riscv/)
   overlay), giving GCC 14.3.0 + external FFTW / FlexiBLAS / OpenBLAS / OpenMPI
   modules. Each README names the exact modules it loads.
3. **The A/B pattern:** swap **one** backend and keep the rest fixed —
   - **BLAS:** `FlexiBLAS` selects the backend at runtime under one unchanged
     binary — see [`hpl`](hpl), [`qe`](qe), [`numpy`](numpy).
   - **FFT:** `LD_PRELOAD` a specific `libfftw3(f).so.3` (r5v vs scalar) under
     one unchanged binary — see [`fftw`](fftw), [`gromacs`](gromacs).
4. **Always check correctness first** — every harness verifies finite /
   bit-identical results across the A/B before comparing timings.

> **RISC-V gotcha (bites everywhere):** on these boards `module load` does *not*
> put the module's `lib/` on the loader path in non-interactive shells, and the
> lmod `module` function returns nonzero / reads unbound vars — so it must run
> **before** any `set -euo pipefail`, or a script dies silently with empty
> output. Set `LD_LIBRARY_PATH` explicitly and load modules before `set -e`.
> Details in [`fftw/README.md`](fftw/README.md).

## Headline findings so far

- **GCC 14.3 SpacemiT X60 mtune** (EasyBuild patch in [`gcc-14.3`](gcc-14.3)):
  scheduler canaries **−5.0%** / **−6.7%** on `fma_chain` / `div_mix`; OpenBLAS
  DGEMM **−3…−7%** vs `generic-ooo` on this run; modest HPL **+6.8%** (local RV2).
- **GCC 15.2 SpacemiT X60 mtune** (EasyBuild patch in [`gcc-15.2`](gcc-15.2)):
  scheduler canaries **−8.7%** / **−7.7%** on `fma_chain` / `div_mix`; OpenBLAS
  DGEMM **+2–4%**; modest HPL **+0.8%** (local RV2 proof, not EESSI yet).
- **OpenBLAS RVV `gemv_n`** on stock EESSI returns NaN → HPL / ELPA / QE / NumPy
  fail; ScaLAPACK **hangs**. The patched build restores correctness (HPL
  **~10.5–11.5 GFLOP/s** on X60).
- **FFTW RVV codelets are real** and win **1.06–1.60×** in isolation (RV2 and
  F3) — but the win is *largely a planner effect* and **evaporates to ~0%**
  inside a real QE SCF (`FFTW_ESTIMATE`, many small transforms).
- **GROMACS:** FFT swap **~1.14–1.23×** on `PME 3D-FFT`; hand RVV `Force`
  backend **~3.31×** whole-app on RV2.
- **LAMMPS:** RVV-Kokkos whole-app scales to **~6–7×** across 8 cores (parallel
  scaling, not RVV-vs-scalar); hand RVV Pair: LJ micro **~1.6×**, EAM in-app
  **1.27×** (still behind `eam/opt`).
- **OpenFOAM motorBike:** GCC `-ftree-vectorize` **~0%**; hand RVV Amul
  **~50% slower** than scalar (solve **~3–4% worse**); hand RVV GS face loops
  also a mild regression — see [`openfoam`](openfoam).
- **waLBerla:** ISA-tag BasicLBM is only ~**1–4%**; real auto-vec wins are
  HeatEquation Jacobi **1.64×** (np1) and UniformGrid `--not-fused` WALL
  **1.30×** / collide **1.54×** (stream flat). Hand `simd::double4_t` RVV is
  slower than FORCE_SCALAR; plain SoA auto-vec microbench ~**2.4×** vs novec
  (~**9×** vs hand simd). Collide/stream split BasicLBM loses to fused — see
  [`walberla`](walberla).
- **IME** (`smt.vmadot`) int8 peaks **~42–45 GOP/s** (~6–7× RVV int8) on both
  boards; end-to-end int4 wins live in [`onnx`](onnx) / [`llamacpp`](llamacpp).
- **GPU (BXE-2-32):** vendor GPGPU path is **closed** (BXM-only DDK); open Mesa
  deferred — see [`gpu`](gpu).

See each directory's `README.md` for the numbers, methodology notes, and the
traps encountered along the way. Site write-ups: [opensolvers.com](https://www.opensolvers.com).
