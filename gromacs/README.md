# GROMACS - real-MD-application FFT backend A/B (PME 3D-FFT)

One runner that A/Bs the **FFT backend** end-to-end through **GROMACS**
`mdrun` (PME molecular dynamics), swapping only the single-precision FFTW
library (`libfftw3f.so.3`) under one unchanged binary via `LD_PRELOAD`. GROMACS
itself is not vendored here - it comes from your GROMACS module (e.g.
`GROMACS/2026.2-foss-2025b`); this directory ships the system generator and the
A/B runner.

This is the **FFT-axis** companion to [`../qe`](../qe)'s BLAS-axis probe: where
QE isolates the *BLAS* backend inside a real DFT SCF (and leaves its FFT half
untouched), GROMACS isolates the *FFT* backend inside a real MD run. On this
build GROMACS reports **`SIMD instructions: None`** - its own force kernels are
scalar - so the mesh part of PME, computed by `libfftw3f`'s 3D-FFT, is the one
cleanly isolated variable when the rest of the binary is held constant.

## Why GROMACS as an FFT probe

PME (Particle-Mesh Ewald) splits the long-range electrostatics into a real-space
part (the dominant `Force` term, scalar here) and a reciprocal-space **mesh**
part whose cost is a pair of 3D FFTs per step. Swapping `libfftw3f` under a fixed
`.tpr` (fixed step count => identical FFT work) moves only the `PME 3D-FFT` /
`PME mesh` rows of GROMACS's own cycle-accounting table; every other row -
`Force`, `Neighbor search`, `Constraints` - is a built-in control that must not
move. That makes GROMACS the FFT analogue of the `dgemm`->QE BLAS spectrum: a
whole real application in which exactly one numerical backend is A/B-swapped.

## Contents

| File | Purpose |
|---|---|
| `gen-water-box.sh` | build the SPC water PME system (solvate + EM + `md.tpr`) |
| `run-gmx-fft-ab.sh` | scalar-vs-r5v(RVV) **FFT** A/B swapping `libfftw3f` via `LD_PRELOAD`, with per-activity PME timers + potential-energy correctness check |

## Setup

```bash
# 1. a GROMACS module providing gmx + FFTW (EESSI riscv dev repo):
module load GROMACS/2026.2-foss-2025b
# 2. two single-precision libfftw3f.so.3 builds to A/B (scalar vs RVV/r5v):
#    ./configure --enable-shared --enable-single --disable-fortran \
#                --disable-doc --disable-mpi [--enable-r5v] \
#                CC=gcc CFLAGS="-O3 -march=rv64imafdcv_zvl256b"
#    (GROMACS mixed-precision links the *single*-prec libfftw3f.so.3,
#     NOT the double-prec libfftw3.so.3 used by the QE/FFTW BLAS work.)
# 3. generate the PME system (writes em.mdp/md.mdp/topol.top/md.tpr):
./gen-water-box.sh
```

Point the two `SCALAR=` / `R5V=` paths at the top of `run-gmx-fft-ab.sh` at your
two `libfftw3f.so.3.6.10` builds.

## Run

```bash
# scalar vs r5v FFT A/B on the PME system (serial, CPU PME):
bash run-gmx-fft-ab.sh v1
```

The runner holds everything except the FFT constant: BLAS pinned
(`FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC`), `OMP_NUM_THREADS=1`,
`mdrun -ntmpi 1 -ntomp 1 -nb cpu -pme cpu -nsteps 2000`, and swaps **only**
`LD_PRELOAD` between the scalar and r5v `libfftw3f`. It greps each `md.log` for
the `PME 3D-FFT` / `PME mesh` cycle rows and captures the average potential
energy via `gmx energy` for a correctness check.

## Correctness - the RVV FFT reproduces the scalar MD

Same `mdrun`, same `md.tpr`, `libfftw3f` swapped:

| FFT backend | avg potential energy | Δ vs scalar |
|---|--:|--:|
| scalar (`libfftw3f`, ~944 KB, ~0 RVV instr) | **-236755 kJ/mol** | - |
| r5v / RVV (`libfftw3f`, ~14 MB, 282k RVV instr) | **-236592 kJ/mol** | **0.069 %** |

The two backends agree to <0.07 % on the average potential energy - the expected
single-precision float-rounding divergence between two different FFT codelet
sets, i.e. physically the same trajectory. The RVV `libfftw3f` does not break the
MD.

## Performance - SPC water PME, 2000 steps (`run-gmx-fft-ab.sh v1`)

GROMACS's own end-of-run `REAL CYCLE AND TIME ACCOUNTING` (WALL), serial,
17.2k-atom SPC water box, 48x48x48 PME grid:

| activity | scalar | r5v (RVV) | speedup | kind |
|---|--:|--:|--:|---|
| **`PME 3D-FFT`** | 21.095 s | 17.152 s | **1.23x** | **FFT (the swapped axis)** |
| `PME mesh` (total) | 73.281 s | 69.792 s | 1.05x | FFT + spread/gather/solve |
| `PME solve Elec` | 9.116 s | 9.139 s | 1.00x | scalar (control) |
| `Force` | 821.35 s | 819.97 s | 1.00x | **scalar kernels (control)** |
| `Neighbor search` | 6.731 s | 6.714 s | 1.00x | scalar (control) |
| `Constraints` | 5.660 s | 5.651 s | 1.00x | scalar (control) |

The **`Force` row is the built-in control**: at 90 % of the run and untouched by
the FFT swap, it reads 821.35 vs 819.97 s (0.17 % - noise), confirming the A/B
isolated exactly one variable. On that isolated variable the RVV `libfftw3f`
delivers a **~1.23x speedup on the 3D-FFT** step (21.1 s -> 17.2 s), and ~1.05x
on the full PME mesh (the non-FFT spread/gather/solve sub-steps are scalar and do
not move).

### Where GROMACS sits on the backend-dilution spectrum

| probe | axis A/B'd | isolated-kernel speedup | whole-app effect |
|---|---|--:|---|
| [`../dgemm`](../dgemm) | BLAS | ~2.3x | (pure kernel) |
| [`../qe`](../qe) (DFT SCF) | BLAS | ~1.5-2.0x on BLAS routines | ~1.2-1.3x (FFT half untouched) |
| **GROMACS** (this dir, MD) | **FFT** | **~1.23x on `PME 3D-FFT`** | small (Force = 90 %, scalar) |

GROMACS is the mirror image of QE: QE speeds up the BLAS and is diluted by its
FFT; GROMACS speeds up the FFT and is diluted by its scalar `Force` kernels
(`SIMD: None` on this build). Both show the same lesson - a whole real
application only moves by the fraction its swapped backend actually owns.

## Gotchas

- **GROMACS mixed precision links the *single*-precision `libfftw3f.so.3`**, not
  the double-precision `libfftw3.so.3`. A separate `--enable-single` FFTW build
  is required; reusing the double-prec lib from the BLAS work will not be picked
  up by PME.
- **`SIMD instructions: None`.** This GROMACS build has no RVV force kernels, so
  the 90 %-of-runtime `Force` term is scalar and unaffected - which is exactly
  what makes `PME 3D-FFT` the one clean variable, but also caps the
  whole-application speedup.
- **Fixed `-nsteps` matters.** The A/B relies on identical FFT work per run; a
  fixed step count (not an energy-minimization / convergence-based stop) keeps
  the `PME 3D-FFT` call count identical (4002 here) between backends.
- **The RVV `libfftw3f` build is slow to compile** (~2 h on one X60: ~2906 dft +
  ~766 rdft SIMD codelets at `-O3`); the scalar variant is ~15 min. This is
  build-time cost, not run-time.
- **`/usr/bin/time` is not present** on the board - rely on GROMACS's own cycle
  table, not an external timer.

## Files

- `gen-water-box.sh` - SPC water PME system generator (solvate + EM + `md.tpr`).
- `run-gmx-fft-ab.sh` - scalar-vs-r5v FFT A/B with per-activity PME timers and a
  potential-energy correctness check.

## Measured on

RISC-V SpaceMiT X60 (Orange Pi RV2, K1 / "Ky(R) X1", 8x X60 @ ~1.6 GHz, 7.7 GB
RAM), GROMACS 2026.2 / foss-2025b (EESSI), CPU FFT library fftw-3.3.10,
`SIMD instructions: None`. FFT backends A/B'd = two single-precision FFTW 3.3.10
`libfftw3f.so.3.6.10` builds from identical sources, one scalar
(`RISCV64_GENERIC`, ~944 KB) and one RVV (`--enable-r5v`,
`-march=rv64imafdcv_zvl256b`, ~14 MB, 282k vector instructions), swapped under
one unchanged `gmx mdrun` via `LD_PRELOAD`.
