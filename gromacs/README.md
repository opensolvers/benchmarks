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

> **Companion Force-axis result:** the scalar `Force` kernels that make this FFT
> A/B clean are themselves the bottleneck (90 % of runtime). [`rvv-backend/`](rvv-backend)
> ships a hand-written RISC-V RVV SIMD backend for GROMACS 2026.2 that vectorizes
> them — `SIMD: RISCV_RVV`, **4.38x on `Force`**, 3.31x whole-app. See "The RVV
> `Force` backend, now built" under "Can the `Force` kernel be optimized?" below.

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
| [`rvv-backend/`](rvv-backend) | hand-written RISC-V **RVV `Force` SIMD backend** for GROMACS 2026.2 (patch + apply/build script) — the Force-axis counterpart to the FFT A/B |

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
| **GROMACS** (this dir, MD) | **`Force` SIMD** (`rvv-backend/`) | **~4.38x on `Force`** | **~3.31x** (Force = 90 %, now vectorized) |

GROMACS is the mirror image of QE: QE speeds up the BLAS and is diluted by its
FFT; GROMACS's *FFT* swap is diluted by its scalar `Force` kernels. That last row
is the exception that proves the rule — because `Force` owns 90 % of the run,
vectorizing *it* (the `rvv-backend/` port, see below) moves the whole application
3.31x, whereas the FFT swap that owns ~2 % moves it barely at all. Same lesson: a
whole real application only moves by the fraction its swapped backend actually
owns — so the backend worth porting is the one that owns the most runtime.

## Can the `Force` kernel be optimized?

The FFT swap only moves ~2 % of the run because `Force` (90 %) is scalar. Two
distinct levers exist for that scalar bottleneck - one free, one a real port:

**1. Threading (free, ~6.9x).** The A/B runs serial (`-ntmpi 1 -ntomp 1`) *by
design*, to isolate the FFT. For actual throughput that wastes 7 of 8 cores. A
thread scan on the same system (SPC water PME, this box's 8 X60 cores):

| layout | ns/day | note |
|---|--:|---|
| 1 rank x 1 omp | 0.377 | the serial A/B baseline |
| 1 rank x 8 omp | 2.595 | ~6.9x |
| 8 rank x 1 omp | **2.610** | fastest, but within noise of 1x8 |
| 2 rank x 4 omp | 2.389 | hybrid splits ~8 % slower |
| 4 rank x 2 omp | 2.404 | " |

So for real runs just use all 8 cores (`-ntomp 8` is simplest); the rank/thread
decomposition is layout-insensitive here (endpoints tie at ~2.60 ns/day). This
is a run-parameter fix, not a kernel fix.

**2. RVV `Force` kernels (a genuine upstream port).** GROMACS 2026.2 has **no
RISC-V RVV SIMD backend** - `cmake/gmxManageSimd.cmake` accepts only x86
(SSE/AVX), ARM (NEON/SVE), IBM VSX, REFERENCE, NONE. The shipped
`libgromacs_mpi.so` contains **0 vector instructions** and is built for base
`rv64...c` (no `v` extension). So `SIMD: None` is not a misconfigured flag - the
RVV kernels do not exist. GROMACS's nonbonded kernels are written against an
abstract SIMD wrapper, so adding RVV means implementing an `impl_riscv_rvv/`
backend (ARM SVE is the closest VLA template). **This port is now done** — see
"The RVV `Force` backend, now built" below and [`rvv-backend/`](rvv-backend): 13
files / ~1545 lines added, float (mixed) precision, measured **4.38x on `Force`**.
It is the only path to vectorizing the `Force` term (auto-vec via `-march`, next
subsection, does not work).

### Why not just compile with `-march=...v`? (measured: it doesn't help)

Before committing to the ~6-10 week port, we tested the cheap alternative: does
simply telling the compiler the vector unit exists let GCC **auto-vectorize** the
scalar `Force` kernels? We rebuilt GROMACS 2026.2 from source on the X60 with
EESSI **GCC 14.3** and `-O3 -march=rv64gcv_zvl256b -funroll-all-loops` (vs the
stock EESSI build's base `rv64...c`), then A/B'd it against the stock binary on
the same `md.tpr`, serial (1 rank x 1 thread), `-nb cpu -pme cpu`.

**The compiler did emit RVV - abundantly:**

| binary | RVV instrs in `libgromacs.so` | in `kernel_ref_4x4` (Force hot path) |
|---|--:|--:|
| stock EESSI (`rv64...c`) | **0** | **0** |
| rebuilt (`rv64gcv_zvl256b`, GCC 14.3) | **164 519** | **4 834** |

**...but it ran ~2 % *slower*, not faster:**

| metric (1000 steps, 1 core, `SIMD: None`) | stock | auto-vec | Δ |
|---|--:|--:|--:|
| `Force` wall time (501 calls) | 205.7 s | 209.6 s | **+1.9 % (slower)** |
| total wall | 228.1 s | 232.8 s | +2.1 % |
| performance | 0.380 ns/day | 0.372 ns/day | -2.1 % |
| conserved energy | -196 667 | -196 670 | correct (matches to 5 s.f.) |

Conserved energy agrees to 5 significant figures, so the auto-vectorized build is
**numerically correct** - the win simply isn't there. GROMACS's `kernel_ref_4x4`
is not a clean SIMD loop: it is gather-heavy `j`-atom access with per-pair cutoff
branches and exclusion masks. GCC vectorizes the surrounding arithmetic (hence
the 4 834 RVV instrs) but wraps it in setup/gather/scatter overhead that cancels
the gain on the in-order X60. **A compiler flag cannot usefully vectorize this
kernel.** This is the empirical confirmation that a meaningful `Force` speedup
requires the hand-written `impl_riscv_rvv/` backend above, not `-march`.

### The RVV `Force` backend, now built (lever 2, measured: 4.38x on Force)

The hand-written backend the two sections above kept pointing at **now exists**:
[`rvv-backend/`](rvv-backend) ships it as a patch against GROMACS `v2026.2`
(`da9e013`) — 6 new `impl_riscv_rvv/` headers (~1483 LOC, ARM SVE as the VLA
template) plus 7 build-system files that register a first-class
`GMX_SIMD=RISCV_RVV`. Built on this same X60 under EESSI GCC 14.3, `gmx --version`
now reports **`SIMD instructions: RISCV_RVV`** — no longer `None` — and the float
SIMD unit test suite passes **158/158** (`SimdFloatingpoint*`, `SimdMath*` incl.
the full `frexp`/`ldexp`-dependent log/exp/pow/erf/trig/pme cascade, `Simd4*`,
`SimdVectorOperations*`).

Re-running the **same serial protocol** (SPC water PME, `md.tpr`, 2000 steps,
`-ntmpi 1 -ntomp 1 -nb cpu -pme cpu`) against the scalar baseline:

| metric (2000 steps, 1 core) | scalar (`SIMD: None`) | RVV (`SIMD: RISCV_RVV`) | speedup |
|---|--:|--:|--:|
| **`Force` wall** | 821.35 s | **187.37 s** | **4.38x** |
| total wall | 910.33 s | **275.40 s** | **3.31x** |
| performance | 0.380 ns/day | **1.256 ns/day** | **3.31x** |
| avg potential energy | -236755 kJ/mol | -236653 kJ/mol | Δ 0.043 % (correct) |

The hand-written backend converts the vector unit into a real **4.38x on the
`Force` term** — exactly the win auto-vectorization (the `-march` subsection
above, ~2 % *slower*) could not reach — and **3.31x whole-application**, also beating the auto-vec build's
0.372 ns/day by **3.38x**. Energy is within the single-precision band (0.043 %).

**The profile inverts.** With `Force` cut ~4.4x, PME mesh (FFT, unvectorized here)
becomes the new leading cost:

| activity | scalar share | RVV share |
|---|--:|--:|
| `Force` | 90.2 % | 68.0 % |
| `PME mesh` | 8.0 % | 28.3 % |

So the two axes compose: the RVV `Force` backend (this section) makes the FFT axis
(the rest of this README) *matter more* — `PME 3D-FFT` grows from ~2 % to ~28 % of
the run, i.e. the FFT-swap lever the top of this doc measures at ~1.23x now acts on
a 3.5x-larger slice. Next optimization target is the PME mesh / FFT half.

### Lever 3: the upstream 2026.3 auto-vectorizable 1x1 kernel (measured: 1.74x, still 2.3x behind the hand-written backend)

GROMACS **2026.3** adds a third, upstream-native lever that sits between the two
above: an **auto-vectorizer-friendly reference kernel**. Commits `c6908b2` /
`0cfe97f` gate a plain-C **1x1** NBNXM kernel behind
`-DGMX_ENABLE_NBNXM_CPU_VECTORIZATION=on` and annotate its `j`-loop with
`#pragma clang loop vectorize(assume_safety)` (`kernel_ref_outer.h:302`), so a
sufficiently modern compiler can vectorize the *reference* kernel with **no
hand-written SIMD backend at all**. It requires **clang ≥ 19** (the pragma is
clang-specific) and is selected at runtime by `GMX_NBNXN_PLAINC_1X1=1`. This is a
fundamentally different mechanism from lever 2: no `impl_riscv_rvv/`, no
`GMX_SIMD` backend — GROMACS still reports `SIMD instructions: None`; the vectors
come entirely from the compiler.

Built on the X60 under **EESSI clang 20.1.8** (`llvm-compilers/20.1.8`, targeting
`riscv64-unknown-linux-gnu`) with `-DGMX_SIMD=None
-DGMX_ENABLE_NBNXM_CPU_VECTORIZATION=on` and `-O3 -march=rv64gcv_zvl256b`. Clang's
`-Rpass=loop-vectorize` remarks confirm it vectorized the 1x1 `j`-loop at
`kernel_ref_outer.h:304`, and the kernel object `kernel_ref_1x1.cpp.o` carries
**3830 RVV instructions** (1155 `vfmul.vv`, 188 `vfmacc.vv`, 916 `vsetvli`, 632
`vluxei64.v` gathers, …). At runtime the log confirms the path:
`Using plain-C-1x1 1x1 nonbonded short-range kernels`.

**Same-binary A/B** (clang-20 2026.3, `md.tpr`, 2000 steps, `-ntmpi 1 -ntomp 1
-nb cpu -pme cpu`), toggling **only** `GMX_NBNXN_PLAINC_1X1`, so the autovec
1x1 path is isolated against the default plain-C 4x4 kernel of the *same* binary:

| metric (2000 steps, 1 core, clang 20) | default plain-C-4x4 | autovec 1x1 (`GMX_NBNXN_PLAINC_1X1=1`) | speedup |
|---|--:|--:|--:|
| **`Force` wall** | 995.11 s | **480.14 s** | **2.07x** |
| total wall | 1107.86 s | **636.55 s** | **1.74x** |
| performance | 0.312 ns/day | **0.543 ns/day** | **1.74x** |
| avg potential energy | -236568 kJ/mol | -236673 kJ/mol | Δ 0.044 % (correct) |

So the upstream autovec 1x1 kernel is a **real** win — **2.07x on `Force` /
1.74x whole-app** vs the same binary's 4x4 default, and **1.43x** vs the stock
scalar 4x4 baseline (0.543 vs 0.380 ns/day). Unlike the GCC `-march` auto-vec
attempt on the 4x4 kernel (lever "Why not just compile with `-march=...v`",
~2 % *slower*), the **1x1 layout is the key**: one i-atom against a scalable
vector of j-atoms is a clean vectorizable loop, where the 4x4 layout's per-pair
cutoff/exclusion branching defeated the vectorizer.

**But the hand-written backend still wins by ~2.3x.** Placing all three Force
levers side by side (all 2000 steps, 1 core, same system):

| Force lever | GROMACS | mechanism | `Force` wall | ns/day | vs scalar |
|---|---|---|--:|--:|--:|
| scalar 4x4 (baseline) | 2026.2 | none (`SIMD: None`) | 821.35 s | 0.380 | 1.00x |
| GCC `-march=...v` 4x4 autovec | 2026.2 | compiler, 4x4 | ~836 s* | 0.372 | 0.98x (slower) |
| **upstream 1x1 autovec** (lever 3) | **2026.3** | **compiler + `#pragma`, 1x1** | **480.14 s** | **0.543** | **1.43x** |
| **hand-written `impl_riscv_rvv/`** (lever 2) | **2026.2** | **`GMX_SIMD=RISCV_RVV`, 4x4** | **187.37 s** | **1.256** | **3.31x** |

<sub>*4x4-autovec Force scaled from its 1000-step run (209.6 s / 501 calls) to the 2000-step protocol; ns/day is directly comparable.</sub>

The dedicated backend is **2.56x faster on `Force`** (187.37 vs 480.14 s) and
**2.31x faster whole-app** (1.256 vs 0.543 ns/day) than the upstream autovec 1x1
kernel. The intuition: the hand-written backend packs a full RVV register of
i-atoms *and* controls the gather/scatter/transpose data movement
(`impl_riscv_rvv_util_float.h`), whereas the autovec 1x1 kernel vectorizes only
the inner j-loop of a single i-atom and leans on compiler-emitted
`vluxei64.v`/`vsuxei64.v` gathers with no layout control.

**The reviewer's portability point (valid, and it favours autovec).** The
hand-written backend is compiled to a **fixed VLEN** (`-mrvv-vector-bits=zvl`,
256-bit on the X60); a different-VLEN core needs a recompile. The autovec 1x1
kernel is **VLEN-agnostic**: clang emitted it as scalable RVV — confirmed in the
binary as `vscale x 4` vectors with runtime `vsetvli`, so the *same* object runs
correctly on any RVV VLEN. That is the autovec path's genuine advantage: **~44 %
of the hand-written backend's throughput** (0.543 / 1.256), zero backend code to
maintain, upstream-native, and portable across VLENs — a compelling default until
(or unless) the dedicated backend is upstreamed. For maximum performance on a
known target, the hand-written backend remains 2.3x ahead.

## Gotchas

- **GROMACS mixed precision links the *single*-precision `libfftw3f.so.3`**, not
  the double-precision `libfftw3.so.3`. A separate `--enable-single` FFTW build
  is required; reusing the double-prec lib from the BLAS work will not be picked
  up by PME.
- **`SIMD instructions: None`.** *This FFT-axis build* has no RVV force kernels,
  so the 90 %-of-runtime `Force` term is scalar and unaffected - which is exactly
  what makes `PME 3D-FFT` the one clean variable, but also caps the
  whole-application speedup. Rebuilding with `-march=rv64gcv_zvl256b` (GCC 14.3)
  *does* make GCC auto-emit RVV into the Force kernels (164k vector instrs) but
  yields **no speedup** (~2 % slower) - see "Why not just compile with
  `-march=...v`?" above. `SIMD: None` reports because *stock* GROMACS has no
  hand-written RVV backend to register, regardless of the compiler `-march`. The
  [`rvv-backend/`](rvv-backend) patch fixes exactly this: a build with that patch
  reports `SIMD instructions: RISCV_RVV` and runs `Force` 4.38x faster (see "The
  RVV `Force` backend, now built" above).
- **Fixed `-nsteps` matters.** The A/B relies on identical FFT work per run; a
  fixed step count (not an energy-minimization / convergence-based stop) keeps
  the `PME 3D-FFT` call count identical (4002 here) between backends.
- **The RVV `libfftw3f` build is slow to compile** (~2 h on one X60: ~2906 dft +
  ~766 rdft SIMD codelets at `-O3`); the scalar variant is ~15 min. This is
  build-time cost, not run-time.
- **`/usr/bin/time` is not present** on the board - rely on GROMACS's own cycle
  table, not an external timer.
- **The 2026.3 autovec 1x1 kernel needs clang ≥ 19 *and* the runtime env var.**
  The `#pragma clang loop vectorize` is clang-specific (GCC ignores it), and the
  build only *offers* the kernel — it is dormant until `GMX_NBNXN_PLAINC_1X1=1`
  selects it at runtime (default is still plain-C-4x4). Confirm activation via the
  `Using plain-C-1x1 1x1 nonbonded short-range kernels` line in `md.log`, not just
  the build flag. The EESSI clang build also needs a longer runtime
  `LD_LIBRARY_PATH` than the GCC one: GCC's `libstdc++` (`CXXABI_1.3.15`) **plus**
  `$EBROOTLLVM/lib/riscv64-unknown-linux-gnu` (`libomp.so`, `libclang_rt`) **plus**
  the FFTW/FlexiBLAS/OpenBLAS module `lib` dirs.

## Files

- `gen-water-box.sh` - SPC water PME system generator (solvate + EM + `md.tpr`).
- `run-gmx-fft-ab.sh` - scalar-vs-r5v FFT A/B with per-activity PME timers and a
  potential-energy correctness check.
- `rvv-backend/` - RISC-V RVV `Force` SIMD backend patch for GROMACS 2026.2
  (`gromacs-2026.2-riscv-rvv-float.patch` + `apply-and-build.sh`), the Force-axis
  port that delivers `SIMD: RISCV_RVV` / 4.38x on `Force`. See its own README.

## Measured on

RISC-V SpaceMiT X60 (Orange Pi RV2, K1 / "Ky(R) X1", 8x X60 @ ~1.6 GHz, 7.7 GB
RAM), GROMACS 2026.2 / foss-2025b (EESSI), CPU FFT library fftw-3.3.10,
`SIMD instructions: None`. FFT backends A/B'd = two single-precision FFTW 3.3.10
`libfftw3f.so.3.6.10` builds from identical sources, one scalar
(`RISCV64_GENERIC`, ~944 KB) and one RVV (`--enable-r5v`,
`-march=rv64imafdcv_zvl256b`, ~14 MB, 282k vector instructions), swapped under
one unchanged `gmx mdrun` via `LD_PRELOAD`.

The **Force-axis** levers were measured on the same X60: the hand-written
[`rvv-backend/`](rvv-backend) built under EESSI **GCC 14.3** (`GMX_SIMD=RISCV_RVV`,
2026.2); the upstream **1x1 autovec** kernel (lever 3) built under EESSI **clang
20.1.8** (`llvm-compilers/20.1.8`) from **GROMACS 2026.3** (tag `v2026.3`,
`-DGMX_ENABLE_NBNXM_CPU_VECTORIZATION=on -DGMX_SIMD=None`, clang-built
FFTW/FlexiBLAS/OpenBLAS), selected at runtime with `GMX_NBNXN_PLAINC_1X1=1`. All
Force runs use the identical serial protocol (`md.tpr`, 2000 steps, `-ntmpi 1
-ntomp 1 -nb cpu -pme cpu -pin off`, `FLEXIBLAS=OPENBLAS
OPENBLAS_CORETYPE=RISCV64_GENERIC OMP_NUM_THREADS=1`).
