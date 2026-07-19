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
| `run-hpl-ab.sh` | - | - | scalar-vs-RVV A/B driver (FlexiBLAS backend swap) |
| `Make.rv64_blis` | - | - | HPL make config linking `xhpl` against static BLIS |
| `build-hpl-blis.sh` | - | - | build a BLIS-linked `xhpl` (no FlexiBLAS) |
| `run-hpl-blis.sh` | - | - | run the BLIS-linked `xhpl` end-to-end |

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

---

# HPL on BLIS - end-to-end validation

The A/B above swaps OpenBLAS backends under one prebuilt `xhpl` via FlexiBLAS.
BLIS is a *different BLAS library*, and there is **no FlexiBLAS on the RV2**, so
BLIS can't be swapped in at run time - it must be **linked into a dedicated
`xhpl` at build time**. This validates that the RVV BLIS build (see
[`../BLIS/`](../BLIS)) drives a real end-to-end Linpack solve correctly, not just
a standalone DGEMM microbenchmark.

Why this matters: a plain `dgemm` bench can look perfect while HPL still NaNs,
because HPL's panel factorization leans on `dgemv`/`dtrsv` - exactly how the
stock OpenBLAS RVV `gemv_n` bug was caught (above). Running HPL on BLIS
exercises those same level-2/3 paths in BLIS.

## Build

```bash
# 1. build + install BLIS first (RVV, CBLAS enabled)
cd ../BLIS && ./build-blis.sh          # -> $HOME/blis-install

# 2. build an xhpl linked against that libblis.a
cd ../hpl
module load OpenMPI                    # or the foss/HPL module that pulls in mpicc/mpirun
BLIS_HOME=$HOME/blis-install ./build-hpl-blis.sh
# -> $HOME/hpl-2.3/bin/rv64_blis/xhpl
```

`build-hpl-blis.sh` fetches HPL 2.3, installs [`Make.rv64_blis`](Make.rv64_blis)
(pointed at the HPL source), and builds. The make config links `xhpl` against
BLIS's **CBLAS** interface:

```make
HPL_OPTS = -DHPL_CALL_CBLAS
LAinc    = -I$(BLIS_HOME)/include/blis
LAlib    = $(BLIS_HOME)/lib/libblis.a -lm -lpthread
CCFLAGS  = ... -O3 -march=rv64imafdcv_zvl256b -fopenmp \
           -L$(GCC14)/lib -B$(GCC14)/lib -Wl,-rpath,$(GCC14)/lib
```

`-DHPL_CALL_CBLAS` routes HPL's `HPL_dgemm`/`HPL_dtrsv` wrappers to the
`cblas_*` symbols that `libblis.a` exports (it was configured
`--enable-cblas --enable-blas`). The `-fopenmp` + `-L`/`-B`/`-rpath $GCC14/lib`
flags are the same two EESSI GCC 14.3.0 link gotchas documented in
[`../BLIS/README.md`](../BLIS/README.md) (libgcc_s/libatomic via `-L`,
`libgomp.spec` via `-B`, `libgomp.so.1` at run time via rpath).

## Run

```bash
./run-hpl-blis.sh HPL.dat              # quick N=8000 1x8
./run-hpl-blis.sh HPL-sweep.dat        # near-peak N=20000 2x4
```

Pure-MPI: `-np 8` with `BLIS_NUM_THREADS=1` per rank, so 8 ranks fill the 8 X60
cores exactly (no thread/rank oversubscription). The runner greps the `W...`
result line and the `PASSED`/`FAILED` residual check.

## Results - RISC-V SpaceMiT X60 (Orange Pi RV2, K1, 8x X60 @ ~1.6 GHz)

Measured end-to-end on the board. `xhpl` links `libblis.a` statically (verified:
`ldd` shows no external BLAS `.so`; `nm` shows `cblas_dgemm` +
`bli_dgemm_rviv_asm_4vx4`). All runs **PASSED** (residual well under threshold).
Baseline for comparison: the patched-RVV OpenBLAS HPL above (`HPL.dat` = 11.55
GFLOP/s, `HPL-sweep.dat` ~10.5 GFLOP/s).

| config | BLIS `xhpl` GFLOP/s | residual | result | vs OpenBLAS-RVV |
|---|--:|---|---|--:|
| `HPL.dat` (N=8000, 1x8) | 4.02 | 4.12e-03 | PASSED | **0.35x** |
| `HPL-sweep.dat` (N=20000, 2x4) | 5.57 | 3.39e-03 | PASSED | **0.53x** |

The interesting question was: BLIS beat OpenBLAS by 1.2-1.3x on single-thread
DGEMM at large N but trailed ~0.8-0.9x at 8 threads (see [`../BLIS/`](../BLIS)).
HPL is pure-MPI (1 BLAS thread/rank), so it exercises BLIS's *single-thread* gemm
path across 8 ranks - the regime where BLIS was ahead on square DGEMM.

**It did not carry through: BLIS HPL runs ~0.35-0.53x of patched-RVV OpenBLAS.**
The gap is largest at `HPL.dat` (`1x8`, 0.35x) and narrows to 0.53x on the
squarer `HPL-sweep.dat` (`2x4`). The reason is that HPL is *not* a square-DGEMM
benchmark: its inner loop is dominated by thin rank-k updates (k=NB=256), `dtrsm`,
and the `dgemv`/`dtrsv` panel factorization - exactly the level-2 and skinny
level-3 shapes where BLIS's RVV kernels trail OpenBLAS's, not the large square
`dgemm` where BLIS wins. The standalone DGEMM microbenchmark advantage does *not*
predict Linpack throughput; correctness, however, holds fully (no NaN, residual
on par with OpenBLAS), which is the primary thing this validation set out to
prove for the RVV BLIS build.

### Full-memory run - process-grid sweep at the largest N that fits

The board is 8 GB with **no swap**. The OpenBLAS `HPL_big.dat` (N=28672, ~6.6 GB)
does **not** fit for the BLIS build - rank 7 was OOM-killed (`signal 9`) mid-run,
matching the documented "OOMs above ~N=25000" ceiling. The largest square problem
that fits is **N=25600** (~5.2 GB matrix + HPL workspace + MPI buffers in ~6 GB
usable), which ran with all 8 cores pinned (`-np 8`, 1 BLIS thread/rank) and ~0.9
GB headroom. Swept across all four `P x Q` grid shapes (`HPL_max*.dat`):

| grid (`P x Q`) | GFLOP/s | time (s) | residual | result |
|---|--:|--:|---|---|
| `2x4` | **5.87** | 1904.5 | 3.37e-03 | PASSED |
| `1x8` | 5.84 | 1916.0 | 2.84e-03 | PASSED |
| `4x2` | 5.23 | 2140.5 | 4.07e-03 | PASSED |
| `8x1` | 4.31 | 2592.9 | 5.75e-03 | PASSED |

**Grid shape matters, and wide beats tall.** The two wide grids (`2x4`, `1x8`)
land together at ~5.85 GFLOP/s; tilting the grid taller costs throughput
monotonically - `4x2` drops ~11% and `8x1` ~26% vs the `2x4` best. HPL's panel
factorization broadcasts down process *columns*, so a grid with more rows (`P`
large) lengthens the critical communication path; `Q >= P` is the usual rule and
it holds cleanly here. This mirrors the OpenBLAS A/B above (squarer/wider grid
beat `1x8` near peak) - the same grid physics, independent of the BLAS backend.
At the full-memory best (`2x4`, 5.87), BLIS still trails the patched-RVV OpenBLAS
`HPL_big` (13.41 GFLOP/s at N=28672) by ~0.44x, consistent with the smaller-N
results above: BLIS solves HPL *correctly* but its RVV kernels are not yet
competitive with OpenBLAS on Linpack's panel-heavy shape mix.

The four configs are committed as [`HPL_max.dat`](HPL_max.dat) (`1x8`),
[`HPL_max_2x4.dat`](HPL_max_2x4.dat), [`HPL_max_4x2.dat`](HPL_max_4x2.dat), and
[`HPL_max_8x1.dat`](HPL_max_8x1.dat) - identical except the `Ps`/`Qs` lines.

## Gotchas specific to HPL+BLIS

- **CBLAS, not Fortran BLAS.** `-DHPL_CALL_CBLAS` avoids Fortran name-mangling
  (`F2CDEFS` stays empty) and links cleanly against BLIS's `cblas_*` symbols. A
  Fortran-BLAS build would need `-lgfortran` and the right underscore convention.
- **Header path is `include/blis/`**, not `include/`, on this BLIS install -
  `cblas.h` and `blis.h` both live there.
- **Link `libblis.a` statically** (full path, not `-lblis`) to avoid picking up
  any stray shared BLAS on the board.
- **`mpicc` as linker** (not a Fortran linker) - the CBLAS path needs no Fortran
  runtime.
- **HPL 2.3's top Makefile is not `-j`-safe** in its `startup_dir`/`refresh_src`
  phase - a parallel `cp` races to populate `src/*/rv64_blis` before the dir
  exists (`cp: cannot create ... No such file`). `build-hpl-blis.sh` scaffolds
  serially (`make startup_dir`) first, then compiles with `-j`.
- **`-Wl,--allow-shlib-undefined` in `LINKFLAGS`.** EESSI OpenMPI 5.x
  `libmpi.so`/`libopen-pal.so` carry undefined refs to their *own* transitive
  deps (UCX `uct_*`/`ucp_*`, UCC, PMIx, hwloc). Those `.so`s are on
  `LIBRARY_PATH` and resolve at run time via `DT_NEEDED`, but `ld` refuses to
  leave them undefined while linking the executable; the flag defers them.
