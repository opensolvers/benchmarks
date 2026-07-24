# LAMMPS - RVV-Kokkos whole-app MD benchmark (SpaceMiT X60)

The classic molecular-dynamics engine, built **RVV-enabled** on the Orange Pi RV2
(SpaceMiT X60, riscv64) and run across the five canonical LAMMPS `bench/`
workloads in three execution modes (serial / Kokkos-OpenMP / MPI). Unlike the
[`../gromacs`](../gromacs) FFT-backend A/B, this is a straight **whole-application
throughput** probe: how fast does a vectorized Kokkos LAMMPS actually integrate
each of the standard MD potentials on this in-order 1.6 GHz core, and how much do
its two parallel back-ends (Kokkos threads vs MPI ranks) recover over the serial
baseline.

LAMMPS is the MD analogue of the GROMACS `Force` probe: its pair-force kernels
are the hot path, and the Kokkos package is the vehicle that puts them on the
vector unit. The binary here is genuinely vectorized — its ELF advertises
`Tag_RISCV_arch = ...v1p0_zve64d`, and the disassembly carries **~53k `vsetvli`**
plus **~12k floating-point `vf*` FMAs** across `liblammps.so.0` — so the Kokkos
runs below exercise real RVV force kernels, not scalar fallback.

## Why these five benchmarks

They are the upstream LAMMPS `bench/` set — the reference workloads every LAMMPS
build is expected to reproduce, each stressing a different force style:

| bench | potential / physics | kernel character |
|---|---|---|
| `lj` | Lennard-Jones (Melt) | short-range pair, the canonical MD baseline |
| `eam` | embedded-atom metal (Cu) | many-body metal potential, heavier per-pair |
| `chain` | bead-spring polymer (FENE) | bonded + short-range, communication-bound |
| `chute` | granular chute flow | granular contact, gravity-driven |
| `rhodo` | rhodopsin protein (CHARMM + PPPM) | full biomolecular: bonds/angles/dihedrals + long-range Coulomb |

All five run at the LAMMPS default problem size (**32000 atoms, 100 steps**), so
the numbers are directly comparable across modes and against any other LAMMPS
build on the same inputs.

## Contents

| File | Purpose |
|---|---|
| `run-lammps-bench.sh` | runs the 5 `bench/` inputs x 3 modes (serial / kokkos8 / mpi8), parses LAMMPS's own `Loop time` + `Performance:` lines into a CSV |

The five inputs and their data files (`in.lj`, `in.eam`, `in.chain`, `in.chute`,
`in.rhodo`, `Cu_u3.eam`, `data.{chain,chute,rhodo}`) come straight from the LAMMPS
source tree's `bench/` directory and are not re-vendored here.

## Setup

```bash
# 1. an RVV-Kokkos LAMMPS providing lmp + liblammps.so + an MPI launcher.
#    On the RV2 this was built from a custom easyconfig against
#    dev.eessi.io/riscv 2025.06-001 (foss-2025b) — see "Building on RISC-V" below.
module load LAMMPS/22Jul2025_update4-foss-2025b-kokkos     # once upstreamed
#    (until then, point $LMP / $MPIRUN at the manually-installed prefix)

# 2. stage the five bench inputs + data files in the run directory:
cp <lammps-src>/bench/in.{lj,eam,chain,chute,rhodo} .
cp <lammps-src>/bench/Cu_u3.eam <lammps-src>/bench/data.{chain,chute,rhodo} .
```

## Run

```bash
# defaults: LMP=$(which lmp), MPIRUN=$(which mpirun), 8 threads / 8 ranks
./run-lammps-bench.sh

# or point it explicitly at a manual build:
LMP=/path/to/lmp MPIRUN=$EBROOTOPENMPI/bin/mpirun ./run-lammps-bench.sh
```

Each benchmark is run serial, then Kokkos/OpenMP (`-k on t 8 -sf kk`,
`OMP_NUM_THREADS=8`), then 8-rank MPI (`mpirun -np 8`); the throughput of record
is **katom-step/s** parsed from LAMMPS's own `Performance:` line, with loop time
from `Loop time of`.

## Performance - 5 bench inputs, 32000 atoms, 100 steps

LAMMPS's own end-of-run `Loop time` and `Performance:` throughput, on 8x X60 @
~1.6 GHz. `speedup` is relative to that benchmark's serial run.

| benchmark | mode | loop (s) | katom-step/s | speedup vs serial |
|---|---|--:|--:|--:|
| **lj**    | serial   | 15.48  | 206.7  | 1.00x |
|           | kokkos8  | 2.496  | 1282   | **6.20x** |
|           | mpi8     | 3.003  | 1065   | 5.15x |
| **eam**   | serial   | 41.69  | 76.8   | 1.00x |
|           | kokkos8  | 5.784  | 553.3  | **7.21x** |
|           | mpi8     | 7.559  | 423.4  | 5.52x |
| **chain** | serial   | 8.950  | 357.5  | 1.00x |
|           | kokkos8  | 2.213  | 1446   | 4.04x |
|           | mpi8     | 1.578  | 2028   | **5.67x** |
| **chute** | serial   | 6.491  | 493.0  | 1.00x |
|           | kokkos8  | 1.547  | 2069   | 4.20x |
|           | mpi8     | 1.325  | 2414   | **4.90x** |
| **rhodo** | serial   | 321.79 | 9.9    | 1.00x |
|           | kokkos8  | 59.10  | 54.1   | 5.44x |
|           | mpi8     | 54.17  | 59.1   | **5.94x** |

**Reading the matrix.** Two clear regimes on 8 cores:

- **Compute-bound pair potentials (`lj`, `eam`)** favour **Kokkos/OpenMP** — the
  RVV force kernels plus shared-memory threading hit **6.2x / 7.2x**, beating MPI
  (5.2x / 5.5x). `eam`'s heavier many-body metal potential is the best threaded
  scaler of the set (7.21x), because there is more vectorizable per-pair work to
  amortize the parallel overhead.
- **Communication / bonded-heavy workloads (`chain`, `chute`, `rhodo`)** favour
  **MPI** — spatial domain decomposition wins where the kernel is lighter and the
  bottleneck is neighbor/ghost exchange. `chain` (bead-spring FENE) and the full
  biomolecular `rhodo` (bonds + PPPM long-range) both top out under 8 ranks
  (5.67x / 5.94x).

`rhodo` is the absolute-cost outlier: at 321.8 s serial it is ~20x slower than
the `lj` melt for the same atom count, reflecting the full CHARMM force field
(bonds/angles/dihedrals) plus PPPM Ewald — and even so the vector build brings it
to a usable ~54 s under either back-end.

### Cross-check against x86

As a sanity anchor, the same `rhodo` input on the x86 build host runs ~14.2 s
serial vs the RV2's 321.8 s — a ~23x single-core gap, which is the expected order
for a 1.6 GHz in-order RISC-V core against a modern out-of-order x86 part. The
point of this page is not that the X60 is fast in absolute terms; it is that the
RVV-Kokkos back-end **scales cleanly** (up to 7.2x on 8 cores) and that a fully
vectorized whole-app MD stack builds and runs correctly on the board.

## Building on RISC-V (custom easyconfig, foss-2025b)

There is no upstream LAMMPS module in the `dev.eessi.io/riscv` stack yet, so the
binary here was built from a **custom easyconfig** (LAMMPS 22Jul2025 update4,
Kokkos 4.6.2, GCC 14.3.0, OpenMPI 5.0.8) against `2025.06-001 (foss-2025b)` via
EESSI-extend + EasyBuild. Five RISC-V-specific easyconfig fixes were required —
none are default LAMMPS behaviour:

1. **Kokkos arch — `kokkos_arch = 'EASYBUILD_GENERIC'`.** The default `NATIVE`
   emits `-mcpu=native/-mtune=native`, which RISC-V GCC 14.3 rejects (`unknown
   CPU 'native'`). `GENERIC` adds no `-m` flags, so Kokkos inherits the correct
   `-march=rv64imafdcv…` from EasyBuild's CXXFLAGS.
2. **CVMFS `readlinkat` storm (two fixes).** Over slow FUSE, CMake's `find_*`
   re-canonicalises every CVMFS path segment per probe → effectively
   non-terminating. Fix (a): feed a bounded `-B$EBROOTGCCCORE/lib/` + per-dir
   `-L/-rpath-link/-rpath` into `CMAKE_{C,CXX,Fortran}_FLAGS_INIT` (only the
   TryCompile phase honours it). Fix (b): replace the ~80-entry Lmod
   `CMAKE_PREFIX_PATH` with a short explicit allowlist, disable
   env/system/registry find paths, and set `CMAKE_FIND_ROOT_PATH_MODE_*=NEVER`.
3. **Pin binutils.** With `SYSTEM_ENVIRONMENT_PATH` off, explicitly pin
   `CMAKE_AR/RANLIB/NM/OBJCOPY/STRIP/READELF/ADDR2LINE` or the Kokkos SIMD
   static-lib archive link fails with `CMAKE_AR-NOTFOUND`.
4. **`-DUSE_SPGLIB=OFF`.** The PHONON `phana` tool (pulled by `BUILD_TOOLS=on`)
   fetches spglib from GitHub and runs a nested `try_compile` with host
   `/usr/bin/ld`, which lacks our OpenMPI transitive flags → undefined refs.
   `PKG_PHONON=OFF` does **not** gate it.
5. **Remove `MDI`** from `general_packages` (a bare `-DPKG_MDI=OFF` is overridden
   by the easyblock's per-package loop). MDI links PRIVATE to the `lammps` target
   and leaves undefined `MDI_*` refs in the unittest `test_*` binaries; it is not
   needed for a compute-only `lmp`.

## Gotchas

- **`ctest` blocks the install: 3/571 tests fail (99% pass), and EasyBuild aborts
  on any ctest failure.** The three are *not* compute defects: `KimCommands` is a
  GoogleTest infra bug ("Only one stdout capturer can exist at a time"), and
  `ewald_conp_charge` / `pppm_conp_charge` are CONP constant-potential electrode
  KSpace tests. Use `--skip-test-step` (or `make install` the already-built tree
  manually) — note `-DENABLE_TESTING=OFF` in configopts is silently overridden by
  the framework's trailing `-DENABLE_TESTING=on`, so tests build and run
  regardless.
- **Detached shells need a login shell for the module env.** `nohup`/background
  runs must source the EESSI/Lmod env via `bash -lc`; otherwise `module` is
  undefined. In the harness, put `set -u` **after** sourcing the env script +
  `module load foss/2025b` (the env script trips unbound-variable errors), and
  default `${PATH:-}` / `${LD_LIBRARY_PATH:-}`.
- **`mpirun` is unreliable on `PATH`.** Derive it from `$EBROOTOPENMPI/bin/mpirun`
  (reliably set after `module load foss/2025b`), which is what the run script's
  `MPIRUN` default points at on the board.
- **Input parametrization is `-var x/y/z`, not `-var xx`.** The `bench/` inputs
  scale the fcc lattice by `x`/`y`/`z` multipliers; the defaults (20x20x20) give
  the 32000-atom / 100-step problem used here.
- **`data.*` files must accompany their inputs.** `in.chain`, `in.chute`,
  `in.rhodo` read `data.chain/chute/rhodo`; `in.eam` needs `Cu_u3.eam`. Stage all
  of them in the run directory.

## Files

- `run-lammps-bench.sh` — 5-input x 3-mode benchmark driver; emits a
  `benchmark,mode,atoms,steps,loop_s,timesteps_per_s,katom_step_s` CSV and logs
  per-run progress to stderr.

## Measured on

RISC-V SpaceMiT X60 (Orange Pi RV2, K1 / "Ky(R) X1", 8x X60 @ ~1.6 GHz, 7.7 GB
RAM), LAMMPS 22Jul2025 update4 / foss-2025b (EESSI `dev.eessi.io/riscv`,
custom easyconfig), Kokkos 4.6.2 (OpenMP + Serial), GCC 14.3.0, OpenMPI 5.0.8.
The `liblammps.so.0` is RVV-vectorized (`Tag_RISCV_arch` includes `v1p0`/`zve64d`;
~53k `vsetvli`, ~12k `vf*` FMAs in disassembly). All runs: 32000 atoms, 100
steps; Kokkos = `-k on t 8 -sf kk` / `OMP_NUM_THREADS=8`; MPI = `mpirun -np 8`;
throughput = LAMMPS's own `Performance:` katom-step/s.
