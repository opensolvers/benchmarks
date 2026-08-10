# rv2-gcc152-openblas-hpl-ab

OpenBLAS (+ HPL) mtune A/B on Orange Pi RV2 using GCC 15.2 with SpacemiT X60 tune.

See `summary.md` / `summary.csv` for numbers.

## Layout (as archived here)

- `raw/` — per-run DGEMM + HPL outputs
- `logs/bench.log`, `logs/hpl.log` — concise run logs
- `bin/bench_dgemm.c` — microbench source (rebuild with the patch + OpenBLAS)
- `HPL.dat`, `run_ab.sh` — config / harness

**Not archived:** static `libs/*.a`, ELF `bench-*` / `xhpl-*`, multi-MB build
logs (rebuildable; see upstream `spacemit-x60-gcc-tune`).
