# SpacemiT X100 BLIS gemm microkernel (`8x8` / `2vx8`)

RVV `sgemm`/`dgemm` microkernels + X100-tuned blocksizes for **VLEN ≥ 256**.

## What we changed

1. **`dgemm` `8x8` (LMUL=2 C intrinsics)** — OpenBLAS-style FMA density: 8
   LMUL=2 accumulators, k-unroll 8, builtin prefetch. Selected when `vlenb ≥ 32`.
2. **`sgemm` stays `2vx8` asm** — same tile shape (16×8 at VLEN=256).
3. **Fallback `4vx4`** for narrower VLEN (`vlenb/4 < 8`).
4. **Blocksizes** (double): `MC=32·mr` (=256), `KC=320`, `NC=4096`.
5. **OpenMP affinity**: `OMP_PROC_BIND=close OMP_PLACES=cores` — ~20%+ on 8T.

## X100 results (2026-09-27)

DGEMM GFLOP/s with `OMP_PROC_BIND=close`, same blocksizes:

| thr | N | **8x8** | 2vx8 asm | OpenBLAS | 8x8 vs 2vx8 | vs OB |
|----:|--:|--------:|---------:|---------:|-------------:|------:|
| 1 | 2048 | 8.12 | 7.46 | 10.66 | 1.09× | 0.76× |
| 1 | 4096 | **10.15** | 9.22 | 10.46 | **1.10×** | **0.97×** |
| 8 | 2048 | 43.66 | 41.52 | 66.28 | 1.05× | 0.66× |
| 8 | 4096 | **51.51** | 48.42 | 66.77 | **1.06×** | **0.77×** |

Progression @ 8T N=4096: stock `4vx4` ~22 → `2vx8` 38 → affinity+blocks 48 → **`8x8` 51.5 GF**.

## Remaining gap to OpenBLAS

Single-thread large-N is near parity (~3%). Multithreaded still ~23% behind — likely
generic BLIS `packm` (no rviv packer yet) and Goto-style paneling/prefetch in OpenBLAS.

## Apply

```bash
./build-blis-x100.sh          # on k3 → ~/blis-x100-install
# always run with:
OMP_PROC_BIND=close OMP_PLACES=cores BLIS_NUM_THREADS=8 …
```

Files: `bli_dgemm_rviv_8x8.c`, `bli_sdgemm_rviv_asm_2vx8.h`,
`bli_{s,d}gemm_rviv_{2vx8.c,asm_2vx8.S}`, `bli_kernels_rviv.h`,
`bli_cntx_init_rv64iv.c`.
