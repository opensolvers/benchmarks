# Quantum ESPRESSO - real-DFT-application BLAS backend A/B

Two runners that A/B a BLAS backend end-to-end through **Quantum ESPRESSO**
`pw.x` (plane-wave DFT SCF), swapping only the BLAS implementation (via
FlexiBLAS) under one unchanged binary. QE itself is not vendored here - it comes
from your QE module (e.g. `QuantumESPRESSO/7.5-foss-2025b`); this directory ships
the inputs, a supercell generator, and the A/B runners.

This is the **real-application** end of the RVV-OpenBLAS bring-up on the SpaceMiT
X60 / K1: where [`../dgemm`](../dgemm) isolates a single kernel, and
[`../hpl`](../hpl) / [`../elpa`](../elpa) probe one HPC solver each, a full DFT
SCF mixes level-3 GEMM (`calbec`, subspace rotation), dense LAPACK
diagonalization, latency-bound BLAS-2, MPI, **and a large FFT fraction** - so it
shows both whether a buggy vector BLAS breaks a production code, and what
fraction of a real run the RVV speedup actually moves.

## Why QE as a probe

`dgemm` gives the BLAS-3 peak; an eigensolver (ELPA) mixes BLAS-3 with BLAS-2;
QE goes one step further and embeds that dense linear algebra inside a real
plane-wave DFT loop whose single biggest cost is the FFT (applying the local
potential to every band). That makes QE the most representative - and most
*diluted* - whole-application number: the BLAS routines still speed up ~2x, but
the FFT half of the run is untouched by the backend swap.

## Contents

| File | Purpose |
|---|---|
| `si-scf.in` | 2-atom Si, gamma-point LDA SCF - tiny, for the **correctness** A/B |
| `si-super-64.in` | 64-atom Si supercell, gamma - GEMM-heavy, for the **performance** A/B |
| `gen_si_supercell.py` | generate an `N`x`N`x`N` Si supercell input (`N [ecut] [maxstep]`) |
| `run-qe-ab.sh` | 3-backend **correctness** A/B (stock-RVV / scalar / patched-RVV) |
| `run-perf-ab.sh` | scalar-vs-RVV **performance** A/B with per-routine timers |

## Setup

```bash
# 1. a QE module providing pw.x + mpirun + FlexiBLAS (EESSI riscv dev repo):
module load QuantumESPRESSO/7.5-foss-2025b
# 2. the Si pseudopotential the inputs reference (pseudo_dir='.'):
curl -O https://pseudopotentials.quantum-espresso.org/upf_files/Si.pz-vbc.UPF
# 3. running as root (e.g. on a dev board) needs OpenMPI 5 / PRRTE opt-in:
export MPIRUN_FLAGS=--allow-run-as-root      # omit if running as a normal user
```

## Run

```bash
# correctness: does the stock vector BLAS break the SCF? (np=4)
RVV_LIB=/path/to/patched/libopenblas.so ./run-qe-ab.sh si-scf.in 4

# performance: how much faster is patched RVV vs scalar on a GEMM-heavy cell?
RVV_LIB=/path/to/patched/libopenblas.so ./run-perf-ab.sh si-super-64.in 4

# bigger / more BLAS-dominated cell (N^3*8 atoms; more bands => more GEMM):
./gen_si_supercell.py 3 22 4 > si-super-216.in
```

Both runners force `FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC` for the
scalar baseline and `FLEXIBLAS=$RVV_LIB` for the vector backend. `run-perf-ab.sh`
also passes `pw.x -ndiag 1` so the subspace diagonalization stays on serial
LAPACK->OpenBLAS (otherwise a linked ELPA/ScaLAPACK would hide the swap).

## Correctness - the stock RVV `gemv_n` NaN bug breaks a real DFT SCF

`run-qe-ab.sh si-scf.in 4` on the SpaceMiT X60, same `pw.x`, BLAS backend swapped:

| backend / dispatch | result | total energy |
|---|---|---|
| **[A]** stock CVMFS OpenBLAS 0.3.30, default RVV (`ZVL256B`) | `Error in routine inverse_s (1): stopping` -> MPI_ABORT | *(none)* |
| **[B]** scalar (`OPENBLAS_CORETYPE=RISCV64_GENERIC`) | converged, 7 iterations | **-14.57861334 Ry** |
| **[C]** patched RVV (`gemv_n` fix backported) | converged, 7 iterations (identical trace) | **-14.57861334 Ry** |

The stock vector backend aborts in `inverse_s` - the overlap-matrix inversion /
Lowdin orthonormalization, which leans on `dgemv` - exactly the `gemv_n` kernel
that [`../dgemm`](../dgemm)'s differential test pinned as broken. So the same
OpenBLAS 0.3.30 RVV bug that NaNs HPL and ELPA also **fatally breaks a production
DFT code**. The patched vector build reproduces the scalar SCF bit-for-bit
(identical energy *and* identical per-iteration convergence trace).

## Performance - 64-atom Si supercell (`run-perf-ab.sh si-super-64.in 4`)

Same `pw.x`, `-ndiag 1`, gamma-point, np=4; scalar vs patched RVV via FlexiBLAS.
QE's own end-of-run timer breakdown (WALL), high-bands variant (`nbnd=272`):

| routine | scalar | patched RVV | speedup | kind |
|---|--:|--:|--:|---|
| `calbec` (`<beta\|psi>`) | 6.25 s | 3.18 s | **1.97x** | DGEMM |
| subspace rotation (in `regterg`) | 33.2 s | 16.6 s | **2.0x** | DGEMM |
| `rdiaghg` (dense diag) | 28.2 s | 18.0 s | **1.57x** | LAPACK->BLAS3 |
| `vloc_psi` (apply V to bands) | 52.6 s | 52.4 s | 1.00x | **FFT (untouched)** |
| `fftw` | 54.5 s | 53.8 s | 1.01x | **FFT (untouched)** |
| **`PWSCF` total** | **144.8 s** | **110.3 s** | **1.31x** | whole SCF |

At the physical default band count (`nbnd=136`) the same cell gives
**67.6 s -> 57.0 s = 1.19x** overall (`calbec` 1.90x, `rdiaghg` 1.53x). The BLAS
routines consistently speed up ~1.5-2.0x; the FFT half of the run does not move
(FlexiBLAS only swaps BLAS), which is what caps the whole-application number.

Two knobs raise the BLAS fraction (and the overall speedup): **more bands**
(136 -> 272 took it 1.19x -> 1.31x) and **bigger supercells** (GEMM scales ~N^3
vs FFT ~N^2 log N). Memory was never the limit here - the 64-atom cell used only
~292 MB across 4 ranks - so there is headroom on the standard 4-8 GB X60 boards.

### Where QE sits on the BLAS-dilution spectrum (same X60, patched RVV vs scalar)

| probe | vector-vs-scalar speedup | why |
|---|--:|---|
| [`../dgemm`](../dgemm) (pure level-3) | ~2.3x | all BLAS-3 |
| [`../hpl`](../hpl) (Linpack) | ~1.8x | BLAS-3 + `dgemv` panel factorization |
| [`../elpa`](../elpa) (eigensolver) | ~1.58x | BLAS-3 + BLAS-2 tridiagonalization |
| **QE** (this dir, full DFT SCF) | **~1.2-1.3x** | BLAS + ~40-50% FFT + MPI |

Each step down adds more non-BLAS / latency-bound work, diluting the BLAS-3 peak.
QE is the most realistic whole-application figure - and the reason a raw `dgemm`
A/B *overstates* what a real code sees.

## Gotchas

- **`-ndiag 1`** keeps the O(nbnd^3) subspace diagonalization on serial
  LAPACK -> OpenBLAS. Without it a QE built against ELPA/ScaLAPACK routes that
  work elsewhere and the FlexiBLAS backend swap barely shows in `rdiaghg`.
- **FFT is not BLAS.** `vloc_psi`/`fftw` are unchanged between backends; don't
  expect a `dgemm`-sized speedup from a whole SCF.
- **Standard QE benchmarks are too big for small boards.** AUSURF-112
  (`nbnd=800`), PSIWAT-586 and GRIR443 (QE's `test-suite/benchmarks/pw/`) need
  many GB; a Si supercell from `gen_si_supercell.py` is the memory-fitting proxy.
- **OpenMPI 5 / PRRTE refuses to run as root** - set
  `MPIRUN_FLAGS=--allow-run-as-root` (and `PRTE_ALLOW_RUN_AS_ROOT[_CONFIRM]=1`)
  on a single-board machine where you are root.

## Files

- `si-scf.in`, `si-super-64.in` - QE inputs (2-atom correctness; 64-atom perf).
- `gen_si_supercell.py` - N^3*8-atom Si supercell generator (MIT licensed).
- `run-qe-ab.sh` - 3-backend correctness A/B.
- `run-perf-ab.sh` - scalar-vs-RVV performance A/B with per-routine timers.

## Measured on

RISC-V SpaceMiT X60 (Banana Pi BPI-F3, K1, 8x X60 @ ~1.6 GHz, 3.7 GB RAM),
QuantumESPRESSO 7.5 / foss-2025b (EESSI), FlexiBLAS 3.4.5, OpenBLAS 0.3.30.
Patched vector backend = OpenBLAS 0.3.30 with the RISC-V `gemv_n` NaN fix
backported (`libopenblas_riscv64_zvl256bp`), the same "good" build used by
[`../dgemm`](../dgemm), [`../hpl`](../hpl) and [`../elpa`](../elpa).
