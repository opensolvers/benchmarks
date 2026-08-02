# waLBerla — RVV / auto-vec campaign on Orange Pi RV2

waLBerla **7.2** lattice-Boltzmann tutorial `01_BasicLBM` on the Orange Pi RV2
(SpaceMiT K1 / X60, RVV 1.0, VLEN=256), under the EESSI RISC-V overlay
(`dev.eessi.io/riscv` `2025.06-001`, `foss-2025b`).

> **Change one variable.** Same source, same OpenMPI/Boost stack, same prm —
> only the **ISA tag / `-march`** differs: stock EESSI binary (`rv64gc`, no `v`)
> vs a local rebuild with `-march=rv64gcv` (ELF tag includes `v1p0` + `zve*` /
> `zvl*`).

> **Bottom line:** BasicLBM ISA-tag alone is only ~**1–4%**. Contiguous auto-vec
> wins show up on HeatEquation (**1.64×** np1) and UniformGrid `--not-fused`
> (**1.30×** WALL / **1.54×** collide). Hand `simd::double4_t` RVV loses to
> FORCE_SCALAR; plain SoA auto-vec is ~**2.4×** vs novec (~**9×** vs hand simd).
> Splitting collide/stream for vectorizable collide costs more than it gains vs
> fused stock on bandwidth-bound D2Q9.

Raw on-board summary + run logs: [`results/`](results).

---

## Setup (what was measured)

| Piece | Value |
|---|---|
| Board | Orange Pi RV2 · `orangepi@192.168.3.112` · Ky X1 / X60 · 8× @ 1.6 GHz |
| Stack | EESSI `2025.06-001` + `foss/2025b` · CMake 4.0.3 · Boost.MPI 1.88.0 |
| App | `01_BasicLBM` (waLBerla 7.2 tutorial; official benches disabled in the EESSI build) |
| Domain (np=1) | `blocks <1,1,1>` · `cellsPerBlock <800,400,1>` · **320 000** cells · **2000** timesteps |
| Domain (np=4) | `blocks <2,2,1>` · `cellsPerBlock <400,200,1>` · same total cells · MPI 4 ranks |
| Timing | wall clock around the binary (`WALL` seconds in the summary) |

Parameter files: [`prm/01_BasicLBM_bench.prm`](prm/01_BasicLBM_bench.prm),
[`prm/01_BasicLBM_bench_mpi4.prm`](prm/01_BasicLBM_bench_mpi4.prm).

Board work dir: `~/walberla-bench/` · scripts:
[`scripts/build-gcv-and-bench.sh`](scripts/build-gcv-and-bench.sh),
[`scripts/resume-gcv-and-bench.sh`](scripts/resume-gcv-and-bench.sh).

---

## ISA tags (`readelf -A`)

| Binary | Path | `Tag_RISCV_arch` (abbrev.) |
|---|---|---|
| **stock-gc** | `/cvmfs/dev.eessi.io/riscv/.../waLBerla/7.2-foss-2025b/build/apps/tutorials/lbm/01_BasicLBM` | `rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_…` — **no `v`** |
| **gcv** | `~/walberla-bench/build-gcv/apps/tutorials/lbm/01_BasicLBM` | same + **`v1p0`** + `zve32f/x` + `zve64d/f/x` + `zvl32/64/128b` |

Stock is the generic EESSI `riscv64/generic` build (scalar `rv64gc`). Local rebuild
uses:

```text
-O2 -ftree-vectorize -march=rv64gcv -mabi=lp64d -fno-math-errno -fPIC
```

with `WALBERLA_BUILD_WITH_MPI=ON`, OpenMP/Python/tests/benchmarks off, tutorials on.

---

## Results (WALL)

| Variant | np | WALL (s) | vs stock |
|---|---|---|---|
| stock-gc | 1 | **129** | — |
| gcv | 1 | **127** | **~1.6%** faster (~1.02×) |
| stock-gc | 4 | **47** | — |
| gcv | 4 | **45** | **~4.3%** faster (~1.04×) |

Source: [`results/walberla-ab-summary.txt`](results/walberla-ab-summary.txt)
(completed `2026-08-02T09:26:42+00:00`). Progress-logger step rates in the run
logs track the same direction (gcv slightly ahead at each 10 s tick).

Rough throughput from WALL (320 k cells × 2000 steps):

| Variant | np | ≈ MLUPS |
|---|---|---|
| stock-gc | 1 | ~5.0 |
| gcv | 1 | ~5.0 |
| stock-gc | 4 | ~13.6 |
| gcv | 4 | ~14.2 |

MPI np=4 vs np=1 is ~**2.7–2.9×** on this small 2-D domain (communication /
domain-decomp overhead still large vs perfect 4×).

---

## Reading

This is the **contiguous-stencil** counterpart to OpenFOAM’s sparse Amul/GS
story ([`openfoam/`](../openfoam)): LBM streams/collides over dense cell arrays,
so `-march=rv64gcv` + auto-vec can help a little — unlike short irregular
gathers, which regressed there. The win is still small because:

1. Stock EESSI already ships a Release/`-O2` build; the axis is mostly **ISA
   enablement**, not a hand RVV kernel.
2. Classic D2Q9 LBM is **memory-bandwidth heavy**; vector ISA tags do not add
   DRAM bandwidth.
3. The tutorial kernel path is not a hand-tuned RVV collision/stream.

A larger RVV story here would need explicit vector kernels (or a code-gen
backend that emits them), plus a bandwidth-aware sizing study — not just
`-march=rv64gcv`.

---

## RVV `simd::double4_t` backend (2026-08-02)

Hand RVV backend for waLBerla’s fixed-width SIMD API, on the same board/build
(GCC / foss only for this A/B).

**GCC approach:** scalable `vfloat64m2_t` cannot be a struct member. Current
`RVV.h` stores a GNU fixed vector instead:

`typedef double f64x4_t __attribute__((vector_size(32)));` inside `double4_t`,
so arithmetic (`a.v + b.v`, …) can stay in registers; `-march=rv64gcv` lowers
to RVV. (Earlier POD `double[4]` + per-op vle/vse spilled heavily.)

| Check | Result |
|---|---|
| Smoke TU (`-march=rv64gcv`, `#include simd/SIMD.h`) | **COMPILE_OK** · `usedInstructionSet()=RVV` · arithmetic/hadd/sqrt/compare OK |
| RVV vs Scalar equiv harness | **EQUIV_OK** on prior POD backend (masks via bool/movemask) |
| `WALBERLA_SIMD_FORCE_SCALAR` | `usedInstructionSet()=scalar emulation` (needs `-std=c++20` when Scalar.h is pulled) |
| `01_BasicLBM` short (100 steps) | **WALL≈10.0 s** on existing `build-gcv` binary — still **scalar CellwiseSweep**; does **not** call `simd::double4_t` |

Patch + notes: [`patches/rvv-simd/`](patches/rvv-simd/) (`RVV.h`, `SIMD.h`, `FieldAllocator.h`).
Scripts: [`scripts/rvv-simd-smoke.sh`](scripts/rvv-simd-smoke.sh),
[`scripts/rvv-simd-equiv.sh`](scripts/rvv-simd-equiv.sh).
Results: [`results/rvv-simd-summary.txt`](results/rvv-simd-summary.txt).

---

## SoA `double4_t` microbench (RVV vs FORCE_SCALAR)

Standalone D2Q9-ish SoA collide TU exercising `walberla::simd::double4_t`
(VL=4): contiguous PDF arrays, `load_aligned` / mul/add / `sqrt` / `store_aligned`,
~18 MiB working set, 80 iters × 3 reps. GCC 14.3, `-march=rv64gcv`.

| Variant | `usedInstructionSet` | BEST WALL (s) | ≈ MLUPS |
|---|---|---|---|
| **RVV `vector_size(32)`** | `RVV` | **14.60** | 1.44 |
| **Scalar** (`-DWALBERLA_SIMD_FORCE_SCALAR`) | `scalar emulation` | **8.34** | 2.52 |
| RVV POD `double[4]` (previous) | `RVV` | **22.80** | 0.92 |

**vs POD RVV:** ~**1.56×** faster (14.60s vs 22.80s).  
**vs FORCE_SCALAR:** still ~**1.75×** slower (speedup RVV/scalar ≈ 0.57×). Checksums match.

Objdump `collideSoA` (hot):

| backend | vle64 | vse64 | vfadd | vfmul | vfmacc† | vfsqrt |
|---|---:|---:|---:|---:|---:|---:|
| POD RVV | 71 | 61 | 11 | 17 | ~0 | 1 |
| vector_size RVV | 48 | 18 | 22 | 20 | 14 | 2 |

† includes `vfmadd`. Mem ops ~132→66; vf*/mem ratio ~0.22→0.88 — fewer spill
stores, more live arithmetic / FMA.

**Caveats:** shuffle helpers (`hadd`, rotates, `invSqrt`) still lane-extract;
`double4_t fq[Q]` can spill; Scalar.h under the same `-march` auto-vecs and
remains the faster baseline on this bandwidth-ish kernel.

| Piece | Path |
|---|---|
| Source | [`microbench/soa_lbm_simd.cpp`](microbench/soa_lbm_simd.cpp) |
| Script | [`scripts/rvv-soa-microbench.sh`](scripts/rvv-soa-microbench.sh) |
| Results | [`results/rvv-soa-microbench.txt`](results/rvv-soa-microbench.txt), [`results/soa-microbench/`](results/soa-microbench/) |
| Board | `~/walberla-bench/microbench/`, `~/walberla-bench/results/soa-microbench/` |

```bash
# On board (after EESSI foss/2025b + LD_LIBRARY_PATH as above):
bash ~/walberla-bench/scripts/rvv-soa-microbench.sh
```

---

## Prefer GCC auto-vec SoA (plain `double*`) over hand `simd::`

Standalone collide TU with **no** `walberla::simd`: contiguous PDF arrays,
`__restrict__` pointers, `#pragma GCC ivdep`, straight-line FP. Same size as
above (nx=262144, 80 iters, best of 3). A/B: `-ftree-vectorize` vs
`-fno-tree-vectorize` under `-O2 -march=rv64gcv`.

| Variant | BEST WALL (s) | ≈ MLUPS |
|---|---:|---:|
| **AUTOVEC** (`-ftree-vectorize`) | **1.58** | **13.3** |
| **NOVEC** (`-fno-tree-vectorize`) | **3.76** | 5.57 |
| prior simd `vector_size(32)` | 14.60 | 1.44 |
| prior FORCE_SCALAR (+auto-vec via Scalar.h) | 8.34 | 2.52 |
| prior POD RVV | 22.80 | 0.92 |

**AUTOVEC ≈ 2.4×** over NOVEC, and **≈ 9×** over hand `double4_t` on this
kernel. Checksums match. Hot loop confirmed vectorized:

- `-fopt-info-vec-optimized`: `soa_lbm_autovec.cpp:62` — *loop vectorized using
  variable length vectors*
- objdump `collideSoA`: AUTOVEC `vle64=9 vse64=9 vfadd=22 vfmul=41 vfmacc=24
  vfsqrt=1`; NOVEC all zero

**Recommendation:** for GCC/RVV SoA collide work, prefer plain contiguous
`double*` + auto-vec over chasing `walberla::simd::double4_t`. Fixed VL=4
underperforms scalable auto-vec here; Scalar.h wrappers still add overhead vs
unwrapped SoA.

**What still blocks auto-vec in full LBM sweeps** (no quick patch applied):

1. `CellwiseSweep` — per-cell `filter` + `Field::get(x±Δ,y±Δ,z,f)` stream-pull;
   XYZ ghost iterator. Neighbor gathers are not contiguous along x for most
   directions.
2. Fused stream-collide — even with fzyx SoA, y±1 PDF rows break a simple
   vectorizable x-stride load of all Q components for the pull.
3. `SplitPureSweep` already has `WALBERLA_RESTRICT` temps and x-loops, but PDF
   I/O still goes through neighbor `get()` — same gather issue for streaming
   dirs. Collide-only contiguous rows (this microbench) vectorize; that is
   **not** the default `01_BasicLBM` path.

| Piece | Path |
|---|---|
| Source | [`microbench/soa_lbm_autovec.cpp`](microbench/soa_lbm_autovec.cpp) |
| Script | [`scripts/rvv-soa-autovec.sh`](scripts/rvv-soa-autovec.sh) |
| Results | [`results/rvv-soa-autovec.txt`](results/rvv-soa-autovec.txt), [`results/soa-autovec/`](results/soa-autovec/) |

```bash
bash ~/walberla-bench/scripts/rvv-soa-autovec.sh
```

---

## Collide-SoA then stream vs fused CellwiseSweep

Real-app follow-up to the auto-vec SoA microbench: fork of `01_BasicLBM` with
`sweepMode=stock|split` on the same Orange Pi `build-gcv` tree
(`-O2 -march=rv64gcv -ftree-vectorize`).

| Mode | Kernel path |
|---|---|
| **stock** | Fused `lbm::makeCellwiseSweep` (stream-pull + collide) |
| **split** | `CellwiseSweep::stream` (gather) **then** contiguous fzyx SoA collide (`SoaCollideKernels.cpp`, plain `double*`, no `walberla::simd`) |

Same problem as the ISA A/B: 320 k cells, 2000 steps, VTK off. Density checksum
identical for stock and split (`checksum_density_sum=320994`).

| Variant | np | WALL (s) | ≈ MLUPS | vs stock |
|---|---:|---:|---:|---|
| stock | 1 | **133.8** | 4.78 | — |
| split | 1 | **175.9** | 3.64 | **0.76×** (slower) |
| stock | 4 | **41.3** | 15.5 | — |
| split | 4 | **64.7** | 9.89 | **0.64×** (slower) |

**Collide auto-vec’d:** yes — GCC
`SoaCollideKernels.cpp:35: optimized: loop vectorized using variable length vectors`.

**Reading:** splitting unlocks a vectorizable collide, but the fused CellwiseSweep
already does stream+collide in **one** memory pass. Split pays an extra full-field
stream gather/scatter plus a second collide write; on this bandwidth-bound D2Q9
tutorial that dominates the collide auto-vec win. Stream remains gather-bound and
is not expected to auto-vec well.

| Piece | Path |
|---|---|
| App | [`apps/collide_stream_split/`](apps/collide_stream_split/) |
| Patch notes | [`patches/collide-stream-split/`](patches/collide-stream-split/) |
| Script | [`scripts/collide-stream-split-ab.sh`](scripts/collide-stream-split-ab.sh) |
| Results | [`results/collide-stream-split.txt`](results/collide-stream-split.txt), [`results/collide-stream-split/`](results/collide-stream-split/) |
| Board binary | `~/walberla-bench/build-gcv/apps/tutorials/lbm/01_BasicLBM_CollideStreamSplit` |

```bash
# On board (after EESSI foss/2025b + LD_LIBRARY_PATH as above):
bash ~/walberla-bench/scripts/collide-stream-split-ab.sh
# Rebuild already done:
SKIP_BUILD=1 bash ~/walberla-bench/scripts/collide-stream-split-ab.sh
```

---

## HeatEquation Jacobi A/B (`02_HeatEquation`)

Stock D2Q5 Jacobi tutorial (VTK off, no CODEGEN, no hand SIMD) — same
`build-gc` / `build-gcv` trees. Domain **200×200**, 10 steps × 2000 Jacobi.

| Variant | np | WALL (s) | vs gc |
|---|---:|---:|---|
| gc (`rv64gc`) | 1 | **35.57** | — |
| gcv (`rv64gcv`) | 1 | **21.76** | **1.64×** |
| novec (`rv64gcv -fno-tree-vectorize`) | 1 | **37.61** | 0.95× |
| gc | 4 | **17.55** | — |
| gcv | 4 | **16.65** | **1.05×** |
| novec | 4 | **19** | 0.92× |

Checksum identical across variants (`4373.0146557791086`). Jacobi
`operator()` contains RVV under gcv (`vle64`/`vse64`/`vfdiv`); none under gc /
true novec. Details: [`results/heat-equation/`](results/heat-equation/),
[`results/heat-equation-rvv-ab.txt`](results/heat-equation-rvv-ab.txt),
[`scripts/heat-equation-rvv-ab.sh`](scripts/heat-equation-rvv-ab.sh).

```bash
bash ~/walberla-bench/scripts/heat-equation-rvv-ab.sh
```

---

## UniformGrid `--not-fused` A/B (`UniformGridBenchmark`)

Official waLBerla 7.2 `UniformGridBenchmark` (enable
`WALBERLA_BUILD_BENCHMARKS=ON`). Defaults: **split + pure + fzyx + fused**;
`--not-fused` separates SoA `SplitPureSweep` collide from stream. Domain
**64³**/block · 200 steps (np=1) or 100 steps (np=4) · VTK off · SRT D3Q19.
No CODEGEN/PYTHON; GCC foss only.

| Mode | Variant | np | WALL (s) | collide (s)† | stream (s)† | fused kernel (s)† | ≈ MLUPS | vs gc |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **not-fused** | gc | 1 | **30** | **4.01** | 1.28 | — | 1.81 | — |
| **not-fused** | gcv | 1 | **23** | **2.61** | 1.34 | — | 2.32 | **1.30×** WALL · **1.54×** collide |
| fused | gc | 1 | 25 | — | — | 4.22 | 2.25 | — |
| fused | gcv | 1 | 21 | — | — | 3.33 | 2.70 | 1.19× WALL · 1.27× fused |
| **not-fused** | gc | 4 | 27 | 12.13 | 7.88 | — | 4.79 | — |
| **not-fused** | gcv | 4 | 26 | 9.76 | 8.07 | — | 5.33 | 1.04× WALL · 1.24× collide |

† TimingPool **total** for last outer block (40 steps np=1; 25 steps np=4).

**Collide auto-vec under gcv:** yes —
`SplitPureSweep.impl.h:485/608` *loop vectorized using variable length vectors*;
binary RVV counts (objdump): gcv `vle64=1822` vs gc `0`. Stream stays ~flat
(gather-bound). Narrative: between HeatEquation (~1.64×) and BasicLBM (~flat);
not-fused collide (~1.54×) is the RVV win; WALL diluted by stream/overhead.

Board needed a one-line null-check in `CartesianDistribution` (np=1 passed
`nullptr` process map). Details:
[`results/uniform-grid-rvv-ab.txt`](results/uniform-grid-rvv-ab.txt),
[`results/uniform-grid/`](results/uniform-grid/),
[`prm/UniformGrid_bench.prm`](prm/UniformGrid_bench.prm),
[`scripts/uniform-grid-rvv-ab.sh`](scripts/uniform-grid-rvv-ab.sh).

```bash
# After WALBERLA_BUILD_BENCHMARKS=ON and UniformGridBenchmark built:
BIN_GC=~/walberla-bench/build-gc/apps/benchmarks/UniformGrid/UniformGridBenchmark
BIN_GCV=~/walberla-bench/build-gcv/apps/benchmarks/UniformGrid/UniformGridBenchmark
$BIN_GC  ~/walberla-bench/prm/UniformGrid_bench.prm --not-fused
$BIN_GCV ~/walberla-bench/prm/UniformGrid_bench.prm --not-fused
```

---

## Reproduce (board)

```bash
export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL=$HOME/eessi-overlay
export EESSI_NO_MODULE_PURGE_ON_INIT=1
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load EasyBuild/5.3.1 EESSI-extend/2025.06-easybuild foss/2025b
module load CMake/4.0.3 Boost.MPI/1.88.0-gompi-2025b
export LD_LIBRARY_PATH="$EBROOTGCCCORE/lib64:${LD_LIBRARY_PATH:-}"

# Full wipe+configure+build+bench (slow):
#   ~/walberla-bench/build-gcv-and-bench.sh
# Resume existing build-gcv tree without wiping:
#   ~/walberla-bench/resume-gcv-and-bench.sh

STOCK=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/waLBerla/7.2-foss-2025b/build/apps/tutorials/lbm/01_BasicLBM
GCV=$HOME/walberla-bench/build-gcv/apps/tutorials/lbm/01_BasicLBM
readelf -A "$STOCK" | head -8
readelf -A "$GCV" | head -10

cd /tmp && "$STOCK" ~/walberla-bench/prm/01_BasicLBM_bench.prm
cd /tmp && "$GCV"   ~/walberla-bench/prm/01_BasicLBM_bench.prm
mpirun -np 4 "$STOCK" ~/walberla-bench/prm/01_BasicLBM_bench_mpi4.prm
mpirun -np 4 "$GCV"   ~/walberla-bench/prm/01_BasicLBM_bench_mpi4.prm
```

Do **not** `rm -rf ~/walberla-bench/build-gcv` unless you intend a full rebuild
(hours on the board at `-j4`).
