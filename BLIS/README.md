# BLIS on RISC-V - build, validation & performance vs OpenBLAS

Builds [BLIS](https://github.com/flame/blis) (BLAS-like Library Instantiation
Software, from the FLAME group) for the **SpaceMiT X60 / K1** (Orange Pi RV2 /
BPI-F3, `rv64gcv`, RVV 1.0, VLEN=256) under the [EESSI](https://www.eessi.io/)
GCC 14.3.0 toolchain, verifies it is numerically correct, and A/Bs its BLAS
throughput against OpenBLAS.

BLIS ships **hand-written RVV assembly** level-3 microkernels under
[`kernels/rviv/3/`](https://github.com/flame/blis/tree/master/kernels/rviv/3)
(dynamic-VLEN via `get_vlenb()`), selected by the `rv64iv` config target - so
this is a real vector-vs-vector comparison, not vector-vs-scalar.

| File | What it does |
|---|---|
| `build-blis.sh` | Clones + configures (`rv64iv`, `--enable-cblas --enable-blas`) + builds BLIS with EESSI GCC 14.3.0. Emits `libblis.a` + CBLAS/BLAS symbols. |
| `run-ab.sh` | Links the backend-agnostic `bench_dgemm.c` against BLIS **and** OpenBLAS, sweeps sizes × threads, prints GFLOP/s side-by-side + a ratio. |
| `bench_dgemm.c` | Square `C=A*B` (level-3 `dgemm`, 3 reps), reports GFLOP/s, prints `C[0]` so a NaN/garbage backend is visible. *(shared with [`../OpenBLAS/`](../OpenBLAS))* |
| `difftest.c` | `dlopen`s a BLAS `.so` (or link a `.a`) and runs level-1/2/3 routines on fixed inputs, printing `sum / sumsq / nan / inf` per routine for differential comparison against a reference. |
| `verify_ctrsm.c` | Full-parameter TRSM correctness sweep (residual-checked); confirms BLIS's triangular solve matches the reference. |

## Why link, not FlexiBLAS

The other BLAS-axis dirs ([`../OpenBLAS`](../OpenBLAS), [`../hpl`](../hpl),
[`../numpy`](../numpy)) A/B by swapping the backend at runtime via FlexiBLAS.
**FlexiBLAS is not installed on the RV2**, so here the A/B is done by *linking*
the same unchanged `bench_dgemm.c` against each library in turn. This still
honours the repo principle - change exactly one variable (the BLAS
implementation), hold source and compiler flags constant.

## Build

```bash
# on the Orange Pi RV2 (native build); ~10-20 min for a full BLIS build
./build-blis.sh                       # installs to $HOME/blis-install
# or:  ./build-blis.sh /custom/prefix
```

`build-blis.sh` forces EESSI **GCC 14.3.0** ahead of the EESSI compat GCC 13.4.0
(see the gotcha below) and configures the RVV 1.0 target with OpenMP threading:

```bash
./configure --prefix=$PREFIX --enable-cblas --enable-blas \
            --enable-threading=openmp rv64iv
make -j$(nproc) && make install
```

> `rv64iv` = RVV 1.0 with **dynamic VLEN detection** (correct for the X60's
> VLEN=256). Do **not** use `sifive_rvv`, which defaults to VLEN=128.
> `--enable-threading=openmp` is required for BLIS to use all 8 cores; without
> it DGEMM stays single-threaded (~2.7 GFLOP/s regardless of thread count).

## Run

```bash
# performance A/B (BLIS vs OpenBLAS)
BLIS_PREFIX=$HOME/blis-install \
OPENBLAS_LIB=$HOME/trsm-pr5830/libopenblas.a \
  ./run-ab.sh
# tune the sweep:  SIZES="1024 2048 4096" THREADS="1 8" ./run-ab.sh

# correctness of the BLIS build (threaded lib needs -fopenmp + the GCC14 flags)
GCC14=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0
gcc -O3 -march=rv64imafdcv_zvl256b -fopenmp verify_ctrsm.c -o verify_ctrsm \
    -I$HOME/blis-install/include/blis -L$GCC14/lib -B$GCC14/lib \
    $HOME/blis-install/lib/libblis.a -lm -lpthread
OMP_NUM_THREADS=8 ./verify_ctrsm    # exit 0 = all pass
```

## A/B methodology

- **One variable**: identical `bench_dgemm.c`, identical `-O3
  -march=rv64imafdcv_zvl256b`, only the linked BLAS differs.
- **OpenBLAS baseline**: the patched RVV OpenBLAS from this repo's work
  (`$HOME/trsm-pr5830/libopenblas.a`, `ZVL128B`/`ZVL256B` RVV kernels).
- **Threads**: 1 (single-core kernel efficiency) and 8 (full X60 parallel).
- **Correctness first**: `verify_ctrsm` / `difftest` must pass before any speed
  number is trusted - a fast NaN is worthless.

## Results

Measured on the Orange Pi RV2 (SpaceMiT X60, 8 cores, VLEN=256), BLIS
`061c2eb` (`rv64iv`, OpenMP threading) vs the patched RVV OpenBLAS
`0.3.33.dev` (`zvl128bp`), both `-O3 -march=rv64imafdcv_zvl256b`, EESSI
GCC 14.3.0. DGEMM GFLOP/s (square, 3 reps, best):

| threads | N | BLIS | OpenBLAS | BLIS/OpenBLAS |
|--:|--:|--:|--:|--:|
| 1 | 1024 | 1.99 | 2.13 | 0.93x |
| 1 | 2048 | 2.73 | 2.25 | 1.21x |
| 1 | 4096 | 2.95 | 2.28 | 1.29x |
| 8 | 1024 | 8.93 | 10.13 | 0.88x |
| 8 | 2048 | 9.60 | 10.83 | 0.89x |
| 8 | 4096 | 9.55 | 11.94 | 0.80x |

Takeaways:

- **Single thread, large N**: BLIS's hand-written RVV assembly gemm microkernel
  (`kernels/rviv/3/`) **beats OpenBLAS by ~20-30%** once the problem is big
  enough to amortize packing (N>=2048). At N=1024 the two are within noise.
- **8 threads**: OpenBLAS scales slightly better (BLIS reaches 0.80-0.89x). Both
  get ~3.5-5x from 8 cores; BLIS's OpenMP path leaves some headroom vs
  OpenBLAS's threading.
- `C[0]=245.24` was identical for both backends at every size - no NaN/garbage.

Correctness (`verify_ctrsm`, BLIS `rv64iv`): **2400 cases, 0 fails,
worst_resid=2.55e-07** (single- and 8-thread). BLIS's triangular solve is
numerically correct on the X60 - unlike the stock OpenBLAS `_rvv_v1` TRSM kernel
this repo's [`../OpenBLAS`](../OpenBLAS) work localized.

## Gotcha - `module load` does not repath `gcc` on the Orange Pi RV2

Same trap as [`../fftw/README.md`](../fftw/README.md): `module load
GCCcore/14.3.0` returns rc=0 but does **not** put GCC 14 first on `PATH` - the
EESSI compat-layer GCC 13.4.0 keeps winning. Both scripts here force it
explicitly, *before* any `set -e` (the lmod `module` function reads unbound vars
and would kill a strict shell):

```bash
GCC14=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0
export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$GCC14/lib64:$GCC14/lib:$LD_LIBRARY_PATH"
```

## Gotcha - GCC14 runtime libs are off the default linker search path

The shared `libblis.so` link (and any `-fopenmp` link against the static lib)
fails with the EESSI GCC 14.3.0 unless you point the driver at `$GCC14/lib`,
where its `libgcc_s`, `libatomic`, and `libgomp.spec` actually live:

```
/usr/bin/ld: cannot find -lgcc_s / -latomic          # → needs -L$GCC14/lib
gcc: fatal error: cannot read spec file 'libgomp.spec'  # → needs -B$GCC14/lib
```

Both scripts pass `-L$GCC14/lib -B$GCC14/lib -Wl,-rpath,$GCC14/lib` on every
link. `-L` resolves the libs, `-B` lets the gcc driver read `libgomp.spec` for
`-fopenmp`, and the rpath lets the resulting binary find `libgomp.so.1` at run
time. (The static-linked `bench_dgemm` still needs the rpath because BLIS's
OpenMP calls pull in `libgomp` dynamically.)

## References

- BLIS RISC-V config: [`config_registry`](https://github.com/flame/blis/blob/master/config_registry) (`rv64iv`, `sifive_rvv`)
- RVV assembly gemm kernels: [`kernels/rviv/3/`](https://github.com/flame/blis/tree/master/kernels/rviv/3)
- BLIS RISC-V CI: [`ci/do_riscv.sh`](https://github.com/flame/blis/blob/master/ci/do_riscv.sh)
- Prior RISC-V work: PR [#737](https://github.com/flame/blis/pull/737) (SiFive X280), [#832](https://github.com/flame/blis/pull/832) (generic RVV, VLEN-tunable), [#868](https://github.com/flame/blis/pull/868) (SG2042, +49% sdgemm)
