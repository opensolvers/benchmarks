# rv2-gcc143-openblas-hpl-ab

OpenBLAS (+ HPL) mtune A/B on Orange Pi RV2 using GCC 14.3 with SpacemiT X60 tune.

See `summary.md` / `summary.csv` for numbers. Upstream detail:
`spacemit-x60-gcc-tune` → `measurements/results/rv2-gcc143-openblas-hpl-ab/`.

## Layout (as archived here)

- `raw/` — per-run DGEMM + HPL outputs
- `logs/bench.log`, `logs/hpl-retry.log` — concise run logs
- `bin/bench_dgemm.c` — microbench source (rebuild with the patch + OpenBLAS)
- `HPL.dat`, `run_ab.sh` — config / harness

**Not archived:** static `libs/*.a`, ELF `bench-*` / `xhpl-*`, multi-MB build
logs (rebuildable; see upstream `spacemit-x60-gcc-tune`).
