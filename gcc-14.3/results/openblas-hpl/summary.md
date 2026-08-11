# OpenBLAS / HPL mtune A/B (GCC 14.3-x60 on Orange Pi RV2)

## Setup

- Host: `orangepi@192.168.1.37`, performance governor @ 1.6 GHz
- Compiler: `/home/orangepi/gcc-tune/install/gcc-14.3.0-x60/bin/gcc` (+ EESSI gfortran/libgomp)
- OpenBLAS: `0.3.33.dev` from `~/ob-rvv`, `TARGET=RISCV64_ZVL256B`, `NO_SHARED=1`, `NO_LAPACK=1`, `USE_OPENMP=1`
- Flags: `-O3 -march=rv64gcv_zba_zbb_zbc_zvl256b -mabi=lp64d` + `-mtune=generic-ooo` vs `-mtune=spacemit-x60`
- DGEMM: static-linked, `CLOCK_MONOTONIC_RAW`, `taskset -c 0`, `OPENBLAS_NUM_THREADS=1`
- Two interleaved passes (ooo→x60→x60→ooo); table averages both passes
- HPL 2.3: linked against each static OpenBLAS (CBLAS via `-DHPL_CALL_CBLAS`); HPL itself built with EESSI OpenMPI/GCC 14.3; N=3000, NB=192, P×Q=2×4, 8 ranks

## DGEMM (OpenBLAS)

| N | ooo mean GF/s | x60 mean GF/s | Δ% (x60 vs ooo) | ooo best | x60 best |
|---|---------------|---------------|-----------------|----------|----------|
| 512 | 2.279 | 2.122 | -6.88% | 2.298 | 2.130 |
| 1024 | 2.414 | 2.252 | -6.69% | 2.510 | 2.257 |
| 2048 | 2.449 | 2.375 | -3.02% | 2.482 | 2.417 |

Checksums matched across mtune variants for all sizes: **yes**.

## HPL

| mtune | N | NB | P×Q | Time (s) | Gflops | Residual |
|-------|---|----|-----|----------|--------|----------|
| ooo | 3000 | 192 | 2×4 | 3.42 | 5.2704e+00 | PASSED |
| x60 | 3000 | 192 | 2×4 | 3.20 | 5.6261e+00 | PASSED |

HPL Δ (x60 vs ooo): **+6.75%** (5.2704e+00 → 5.6261e+00 Gflops).

## vs GCC 15.2 (same harness)

| Metric | 14.3 Δ% (x60 vs ooo) | 15.2 Δ% |
|--------|----------------------|---------|
| DGEMM N=512 | -6.88% | +2.19% |
| DGEMM N=1024 | -6.69% | +2.25% |
| DGEMM N=2048 | -3.02% | +3.79% |
| HPL N=3000 | +6.75% | +0.79% |

## Caveats

- Static `NO_SHARED=1`; `make libs` only (skip broken utest).
- gcc-14.3.0-x60 install has no libgomp/gfortran; DGEMM/HPL linked against EESSI GCCcore 14.3 `-lgomp`/`gfortran`.
- Single-thread OpenBLAS DGEMM favors `generic-ooo` on this 14.3 run (opposite sign vs 15.2); modest HPL favors x60.
- HPL host code is EESSI mpicc; only the linked OpenBLAS differs by mtune.
- HPL N=3000 is a modest sanity run (~few seconds), not peak-system sizing.
- First HPL Make without `HPL_LIBS`/`-DHPL_CALL_CBLAS` failed link; rebuilt via rv64_blis-style Make (same fix as 15.2).
