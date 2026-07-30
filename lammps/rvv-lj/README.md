# LAMMPS LJ/cut — hand RVV Pair microkernel (step 1)

Prototype for an RVV-optimized **force-on-i** LJ/cut loop matching the hot path in
LAMMPS `PairComputeFunctor` (`pair_kokkos.h`), without full LAMMPS linkage yet.

Measured on Orange Pi RV2 (SpaceMiT X60): **~1.62–1.64×** vs scalar with
**EESSI GCC 14.3.0**, max |Δf| ~1e-14.

## Idea

1. Scalar gather of neighbor Δr into **SoA tiles** (`dx[]/dy[]/dz[]`)
2. RVV (e64 / m1): `rsq`, masked cutoff, `r2inv` / `r6inv` / `fpair`
3. Reduce `Δ·fpair` into `f[i]`

Newton-on scatter to `f[j]` is deferred (needs atomics / FULL list redesign).

## EESSI on RISC-V (do not skip)

RISC-V software is **not** in the production catalogue the same way as
x86_64/aarch64. Use the **RISC-V development stack**:

| Piece | Role |
| --- | --- |
| `/cvmfs/software.eessi.io` | Compat layer + **init / Lmod only** on RISC-V |
| `/cvmfs/dev.eessi.io/riscv` | Actual **riscv64** apps/toolchains (`2025.06-001`, `riscv64/generic`) |

Docs: [dev.eessi.io/riscv](https://www.eessi.io/docs/repositories/dev.eessi.io-riscv/)  
Easystack / build recipes: [EESSI/dev.eessi.io-riscv](https://github.com/EESSI/dev.eessi.io-riscv)

**Init (required):**

```bash
export EESSI_VERSION_OVERRIDE=2025.06-001
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
# Expected: "compatibility layer" message — that is normal on RISC-V.
# Software resolves from /cvmfs/dev.eessi.io/riscv/...
module load GCC/14.3.0
```

**Gotchas (RISC-V EESSI):**

- Never `source …/init/… | tail` / pipe `module load` — breaks Lmod in the parent shell.
- Prefer **EESSI GCC 14.3** over board `/usr/bin/gcc` 13.3 (fixed-length RVV /
  `-mrvv-vector-bits=zvl` and the rest of the X60 stack assume 14.3).
- Builds are **`riscv64/generic`** today (no CPU-optimized EESSI targets yet).
- For EasyBuild *extensions* of the stack, use `EESSI-extend` + `WORKING_DIR` on
  NVMe (not `/tmp` tmpfs) — see [building on EESSI](https://www.eessi.io/docs/using_eessi/building_on_eessi/).
  This microbench only needs `GCC/14.3.0`, not extend.

## Build / run on RV2

```bash
export EESSI_VERSION_OVERRIDE=2025.06-001
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load GCC/14.3.0
make CC=gcc
taskset -c 0 ./lj_pair_bench [nlocal=2048] [rounds=20]
```

`-march=rv64gcv_zvl256b` via the Makefile `ARCH` flag.

## Cross-board confirmation - Banana Pi BPI-F3

Same microbench on a [Banana Pi BPI-F3](https://www.banana-pi.org/) (SpaceMiT
K1, 8× X60), EESSI GCC 14.3.0 (`-L/-B` for `libgcc_s`):

```
n=2048 … nnz=98426 rounds=20
max|f_scalar - f_rvv| = 6.750e-14 OK
scalar: 58.765 ns/pair
rvv:    37.417 ns/pair  speedup 1.57x
```

Matches the RV2 ~1.62× within board noise.

## Next

See [`INTEGRATION.md`](INTEGRATION.md). **In-LAMMPS verify is done** via
[`plugin/`](plugin/) (`lj/cut/rvv`, bit-exact vs stock on melt A/B).
