# HeatEquation RVV A/B — Orange Pi RV2

waLBerla **7.2** tutorial `02_HeatEquation` (stock D2Q5 Jacobi, **no VTK**, no
CODEGEN/Python, no handwritten SIMD) — `rv64gc` vs `rv64gcv` (+ true
`-fno-tree-vectorize` novec under gcv ISA).

Board: `orangepi@192.168.3.112` · builds `~/walberla-bench/build-gc` /
`build-gcv` · EESSI `2025.06-001` + `foss/2025b`.

## Problem

| Piece | Value |
|---|---|
| App | prm-driven fork of `apps/tutorials/pde/02_HeatEquation` (stock Jacobi path) |
| Domain (np=1) | `blocks <1,1,1>` · `cellsPerBlock <200,200,1>` · **40 000** cells |
| Domain (np=4) | `blocks <2,2,1>` · `cellsPerBlock <100,100,1>` · same total cells |
| Work | **10** timesteps × **2000** Jacobi iterations (VTK off) |
| Correctness | `checksum_u_sum=4373.0146557791086` (identical gc / gcv / novec, np=1 & 4) |

Prms: [`prm/02_HeatEquation_bench.prm`](../prm/02_HeatEquation_bench.prm),
[`prm/02_HeatEquation_bench_mpi4.prm`](../prm/02_HeatEquation_bench_mpi4.prm).
Bench source: [`apps/heat_equation_bench/`](../apps/heat_equation_bench/).
Script: [`scripts/heat-equation-rvv-ab.sh`](../scripts/heat-equation-rvv-ab.sh).

## WALL (primary, 200×200)

| Variant | np | WALL (s) | vs gc |
|---|---:|---:|---|
| **gc** (`-march=rv64gc -ftree-vectorize`) | 1 | **35.568** | — |
| **gcv** (`-march=rv64gcv -ftree-vectorize`) | 1 | **21.755** | **1.635×** (~39% faster) |
| **novec** (`-march=rv64gcv -fno-tree-vectorize`) | 1 | **37.607** | 0.95× (slightly slower than gc) |
| gc | 4 | **17.547** | — |
| gcv | 4 | **16.646** | **1.054×** (~5% faster) |
| novec | 4 | **19** | 0.92× (comm-bound; gap noisy) |

Raw log: [`heat-equation-rvv-ab.txt`](heat-equation-rvv-ab.txt).

**np=1 reading:** gcv is **~1.73×** faster than true novec on the same ISA tag —
the win is auto-vec / RVV codegen, not just `-march=rv64gcv` scheduling.
novec ≈ gc (slightly worse), as expected when vectorization is forced off.

**np=4:** per-rank 100×100 + ghost exchange; ISA gap shrinks to ~5%.

## Was Jacobi vectorized?

**Yes under gcv** (objdump of `JacobiIteration::operator()`):

| Binary | `vle64` | `vse64` | `vfdiv` | `vsetvli` |
|---|---:|---:|---:|---:|
| gc | 0 | 0 | 0 | 0 |
| gcv | 5 | 11 | 2 | 9 |
| true novec | 0 | 0 | 0 | 4 |

`-fopt-info` does **not** report a clean “loop vectorized” on the Jacobi
`WALBERLA_FOR_ALL_CELLS_XYZ` site (iterator / nest complexity); RVV still
appears in the Jacobi symbol via basic-block / inlined paths (`FieldIterator`
BB vectorization notes). Disassembly is the stronger evidence.

Caveat: cmake `make CXXFLAGS=…` does **not** override the embedded compile
rule — true novec requires compiling the TU with `c++ … -fno-tree-vectorize`
directly (script does this).

## Reproduce

```bash
# On board, after EESSI foss/2025b + LD_LIBRARY_PATH as in walberla/README:
bash ~/walberla-bench/scripts/heat-equation-rvv-ab.sh
# Rebuild already done:
SKIP_BUILD=1 bash ~/walberla-bench/scripts/heat-equation-rvv-ab.sh
```
