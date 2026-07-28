# Kokkos on RISC-V (SpaceMiT X60) — learnings & results

Working notes for **Kokkos** as used by **LAMMPS** on the Orange Pi RV2
(SpaceMiT X60 / K1, RVV 1.0, VLEN=256), under the EESSI RISC-V stack.

Companion kernels (hand RVV Pair plugins, not Kokkos SIMD):
[`../lammps/rvv-lj/`](../lammps/rvv-lj), [`../lammps/rvv-eam/`](../lammps/rvv-eam).

> **Change one variable.** Here the axes are: execution space (Serial / OpenMP),
> whether Pair uses hand RVV, and the EESSI toolchain (always GCC 14.3 for
> published numbers).

---

## EESSI RISC-V (required)

RISC-V apps are **not** in the production `software.eessi.io` catalogue the same
way as x86_64/aarch64. Use the development stack:

| Piece | Role |
| --- | --- |
| `/cvmfs/software.eessi.io` | Compat layer + **init / Lmod only** |
| `/cvmfs/dev.eessi.io/riscv` | **riscv64** software (`2025.06-001`, `riscv64/generic`) |

Docs: [dev.eessi.io/riscv](https://www.eessi.io/docs/repositories/dev.eessi.io-riscv/) ·
Repo: [EESSI/dev.eessi.io-riscv](https://github.com/EESSI/dev.eessi.io-riscv)

```bash
export EESSI_VERSION_OVERRIDE=2025.06-001
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
# "compatibility layer only" message is NORMAL on RISC-V.
module load GCC/14.3.0          # or foss/2025b for full LAMMPS
```

**Gotchas:** never pipe `source`/`module` into `tail`; prefer EESSI **GCC 14.3**
over board gcc 13.3; builds are `riscv64/generic` today; EasyBuild scratch must
leave `/tmp` tmpfs (`WORKING_DIR` on NVMe) when using `EESSI-extend`.

**LAMMPS is not on the RISC-V CVMFS easystack** (unlike ESPResSo 4.2.2). The RV2
install is a local `eessi-overlay` build (`22Jul2025_update4-foss-2025b-kokkos`).

---

## What Kokkos is doing in LAMMPS

Kokkos supplies portable **parallel dispatch** over Views:

| Primitive | Role in LAMMPS |
| --- | --- |
| `parallel_for` | Pair / Neigh / Integrate loops over atoms |
| `parallel_reduce` | Energies, virials, reductions |
| `parallel_scan` | Rare in Pair path |

**Execution spaces on our build:** **OpenMP + Serial** only  
**Arch:** `Kokkos_ARCH_EASYBUILD_GENERIC` (avoids `-mcpu=native`, which RISC-V
GCC 14.3 rejects; march comes from EESSI flags including `…cv…` / `zve*`).

**SIMD layer (Kokkos 4.6.2 bundled with LAMMPS):** backends are **AVX2 / AVX512 /
NEON / Scalar** only — **no RVV abi**. LAMMPS `PairComputeFunctor` does **not**
use `Kokkos::Experimental::simd`; it runs scalar FP inside `parallel_for`.
Any RVV in `liblammps.so` today is **GCC auto-vec** (`-ftree-vectorize` +
`-march=…cv…`), not a deliberate Kokkos SIMD path.

---

## Hot kernels (where time goes)

From board profiling (`in.lj` / `in.melt`, Kokkos OpenMP):

| Activity | Share | Notes |
| --- | ---: | --- |
| **Pair** (`PairComputeFunctor` / `PairLJCutKokkos`) | **~84–87%** | Dominant |
| **Neigh** | ~8–12% | List build / distance |
| Comm / Modify / Other | small | |

Leaf: `PairComputeFunctor<PairLJCutKokkos, …>` (~46% of samples in earlier gdb
sampling). Inner loop (`pair_kokkos.h`):

1. Indirect neighbor gather → `Δx,Δy,Δz`, `rsq`
2. Cutoff branch
3. `compute_fpair`: `r2inv → r6inv → LJ` (`lj1`/`lj2`)
4. Accumulate `f[i]`; Newton-on also scatters to `f[j]`

### Why stock Kokkos Pair is a weak RVV target *as written*

- **Gather/scatter** of positions/forces (irregular neighbor indices)
- Variable `jnum`, type-dependent params, `special_lj` → divergence
- Newton-on **atomics** on `f[j]` fight wide SIMD
- No Kokkos RVV SIMD backend to express a vector Pair cleanly

Auto-vec alone will not deliver OpenBLAS-like gains. The workable path is the
same spirit as GROMACS RVV Force: **hand layout + hand RVV** on the Pair math.

---

## RVV Pair direction (results so far)

### LJ/cut — [`../lammps/rvv-lj/`](../lammps/rvv-lj)

| Build | Scope | Result |
| --- | --- | --- |
| EESSI GCC 14.3 microbench | force-on-i vs naive scalar | **~1.61–1.64×** |
| `lj/cut/rvv` plugin vs stock `lj/cut` | in-LAMMPS Pair, 4000 atoms | **~1.02×** (near parity) |

In-app LJ is gather/scatter limited; stock is already `-march=…cv…` auto-vec’d.
See [`../lammps/rvv-lj/INTEGRATION.md`](../lammps/rvv-lj/INTEGRATION.md).

### EAM — [`../lammps/rvv-eam/`](../lammps/rvv-eam)

Cu `Cu_u3.eam`, 864 atoms, 100 steps, 1 core, force-only; forces **bit-exact** vs stock.

| Style | Pair time | vs `eam` |
| --- | ---: | ---: |
| `eam` | 0.780 s | 1.00× |
| **`eam/rvv`** | 0.614 s | **1.27×** |
| `eam/opt` | 0.574 s | 1.36× |

EAM is ~**96%** Pair — a better RVV target than LJ. Hand RVV beats stock `eam`
but still trails the OPT package (~7%).

---

## Board LAMMPS snapshot

| Item | Status |
| --- | --- |
| Version | 22 Jul 2025 Update 4 + Kokkos 4.6.2 |
| Binary | `~/eessi-overlay/.../LAMMPS/22Jul2025_update4-foss-2025b-kokkos/bin/lmp` |
| Lmod module | **Missing** (EB failed at `ctest`; manual install) |
| `in.melt` | **PASSED** (~42 steps/s, 1 MPI × 1 OpenMP in one check) |
| CVMFS | No `LAMMPS` in `dev.eessi.io/riscv` easystack |

Accelerator line from `lmp -h`:

```
KOKKOS package API: OpenMP Serial
OPENMP package API: OpenMP
FFT library = FFTW3
```

---

## Practical optimization backlog

1. **`eam/rvv` vs `eam/opt`** — close the ~7% gap (templating / less spill).
2. **Kokkos `SIMD_RVV` backend** upstream — unlocks SIMD-aware kernels later.
3. Neigh distance filter — lower ROI than Pair.
4. LJ further tuning — low ROI after ~1.02× in-app.

---

## Related

- LAMMPS handoff (build saga): `riscv-learnings` `docs/LAMMPS_RV2_STATUS.md`
- GROMACS RVV Force (analogous “hand SIMD the hot kernel” success):
  [`../gromacs/rvv-backend/`](../gromacs/rvv-backend)
- Benchmarks TODO: [`../todo.md`](../todo.md) (LAMMPS marked done for bring-up;
  RVV Pair is the follow-on)
