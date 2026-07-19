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

## What's here

| Dir | What it measures | Axis swapped | Kind |
|---|---|---|---|
| [`OpenBLAS/`](OpenBLAS) | OpenBLAS on RISC-V: DGEMM performance + differential correctness + TRSM sweep; localizes two broken RVV kernels (`gemv_n` NaN, `_rvv_v1` TRSM VLEN bug) | BLAS | microbench + verification |
| [`numpy/`](numpy) | BLAS/LAPACK backend as seen through NumPy/SciPy | BLAS | application proxy |
| [`hpl/`](hpl) | High-Performance Linpack, end-to-end | BLAS (FlexiBLAS) | application A/B |
| [`elpa/`](elpa) | Dense real-symmetric eigensolver (ELPA, 1-stage) | BLAS | microbench |
| [`scalapack/`](scalapack) | Distributed dense eigensolver (ScaLAPACK `PDSYEV`, pure-MPI) | BLAS | microbench (MPI) |
| [`qe/`](qe) | Quantum ESPRESSO `pw.x` plane-wave DFT SCF, end-to-end | **BLAS** (FlexiBLAS) | real-application A/B |
| [`fftw/`](fftw) | FFTW 3.3.10 RVV (`r5v`) vs scalar, **and** the FFT axis inside a QE SCF | **FFT** (`--enable-r5v`, `LD_PRELOAD`) | microbench + real-application A/B |
| [`gromacs/`](gromacs) | GROMACS `mdrun` PME molecular dynamics, end-to-end | **FFT** (single-prec `libfftw3f`, `LD_PRELOAD`) | real-application A/B |
| [`ime/`](ime) | int8 (`s8s8s32`) GEMM microkernel on the X60 **IME** (`smt.vmadot`) vs RVV | int8 kernel | microkernel + verification |
| [`onnx/`](onnx) | int4 `MatMulNBits` LLM-FFN inference via ONNX Runtime MLAS | int4 kernel | application + root-cause writeup |
| [`papers/`](papers) | Longer-form writeups (e.g. X60 IME block-scale optimization) | — | prose / PDF |

### Two views of the same question

Several directories deliberately pair a **microbenchmark** with a
**real-application** measurement of the *same* backend, because they often
disagree — and that disagreement is the interesting result:

- **BLAS axis:** [`OpenBLAS`](OpenBLAS)/[`numpy`](numpy)/[`elpa`](elpa) (kernel level)
  → [`hpl`](hpl)/[`qe`](qe) (whole application).
- **FFT axis:** [`fftw`](fftw) standalone microbench (RVV wins **1.06–1.60×**)
  → the same RVV FFTW dropped into a Quantum ESPRESSO SCF (**~0% end-to-end**,
  documented in [`fftw/README.md`](fftw/README.md)), and into a
  [`gromacs`](gromacs) PME MD run (RVV wins **1.23×** on the isolated
  `PME 3D-FFT`, but that step is a small fraction of a scalar-`Force`-dominated
  run). A microbenchmark speedup is not an application speedup.
- **int8/int4 axis:** [`ime`](ime) microkernel → [`onnx`](onnx) real inference.

## How to use / replicate

There is no top-level build. **Each directory is standalone** — `cd` into the
one you want and follow its `README.md`, which lists the exact modules, build
command, run command, and expected output.

Common ground for reproducing any of these:

1. **Hardware:** a SpaceMiT X60 / K1 board (Orange Pi RV2 or Banana Pi BPI-F3).
   Results are reported per-board; the FFTW/QE numbers here are from the
   Orange Pi RV2.
2. **Toolchain:** the [EESSI](https://www.eessi.io/) 2025.06 stack (or the
   `dev.eessi.io` riscv overlay), giving GCC 14.3.0 + external
   FFTW / FlexiBLAS / OpenBLAS / OpenMPI modules. Each README names the exact
   modules it loads.
3. **The A/B pattern:** swap **one** backend and keep the rest fixed —
   - **BLAS:** `FlexiBLAS` selects the backend at runtime (`FLEXIBLAS=OpenBLAS`
     vs another) under one unchanged binary — see [`hpl`](hpl), [`qe`](qe),
     [`numpy`](numpy).
   - **FFT:** `LD_PRELOAD` a specific `libfftw3.so.3` (r5v vs scalar) under one
     unchanged binary — see [`fftw/run-qe-fft-ab.sh`](fftw/run-qe-fft-ab.sh).
4. **Always check correctness first** — every harness verifies finite /
   bit-identical results across the A/B before comparing timings.

> **RISC-V gotcha (bites everywhere):** on these boards `module load` does *not*
> put the module's `lib/` on the loader path in non-interactive shells, and the
> lmod `module` function returns nonzero / reads unbound vars — so it must run
> **before** any `set -euo pipefail`, or a script dies silently with empty
> output. Set `LD_LIBRARY_PATH` explicitly and load modules before `set -e`.
> Details in [`fftw/README.md`](fftw/README.md).

## Headline findings so far

- **FFTW RVV codelets are real** (bit-accurate) and win **1.06–1.60×** in
  isolation — but the win is *largely a planner effect* and **evaporates to ~0%**
  inside a real Quantum ESPRESSO SCF, which plans with `FFTW_ESTIMATE` over
  thousands of small mixed-radix transforms.
- On the X60, **neither the BLAS axis nor the FFT axis meaningfully moves a real
  QE DFT run** with today's drop-in vectorized libraries.
- The X60 **IME** (`smt.vmadot`) int8 path is where the real integer-GEMM wins
  live — see [`ime`](ime) and [`onnx`](onnx).

See each directory's `README.md` for the numbers, methodology notes, and the
traps encountered along the way.
