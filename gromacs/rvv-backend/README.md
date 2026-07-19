# GROMACS 2026.2 — RISC-V RVV SIMD backend (float / mixed precision)

A hand-written `impl_riscv_rvv/` SIMD backend for **GROMACS 2026.2**, targeting
the **SpaceMiT X60 / K1** (RVV 1.0, VLEN=256, `zvl256b`) under EESSI GCC 14.3.
This is the port the [`../README.md`](../README.md) "Can the `Force` kernel be
optimized?" section concluded was the *only* path to a real `Force` speedup —
now implemented, built, and measured.

> **Status: captured here as a patch for reproduction and review. No upstream PR
> yet.** Float / mixed-precision path only (double not yet ported).

## What's here

| File | Purpose |
|---|---|
| `gromacs-2026.2-riscv-rvv-float.patch` | the complete backend as one diff vs `v2026.2` (commit `da9e013`) — 13 files, +1545 lines |
| `apply-and-build.sh` | apply the patch to a pristine v2026.2 checkout + configure + build `gmx`/`simd-test` on the X60 |

## What the patch contains

**6 new backend headers** (`src/gromacs/simd/include/gromacs/simd/impl_riscv_rvv/`, ~1483 LOC) implementing the GROMACS abstract-SIMD wrapper against RVV 1.0 intrinsics, using **ARM SVE as the VLA template** (the only other variable-length backend in-tree):

| header | LOC | contents |
|---|--:|---|
| `impl_riscv_rvv.h` | 64 | umbrella include / router |
| `impl_riscv_rvv_definitions.h` | 106 | `GMX_SIMD*` capability macros, widths, alignment |
| `impl_riscv_rvv_general.h` | 61 | init / prefetch / arch glue |
| `impl_riscv_rvv_simd_float.h` | 620 | `SimdFloat` + `SimdFInt32`: arith, logical, compare/blend/mask, `frexp`/`ldexp`, `rsqrt`/`rcp` (+`maskz`), reductions |
| `impl_riscv_rvv_simd4_float.h` | 356 | fixed-width `Simd4Float` (4-lane: dotProduct, transpose, reduce) |
| `impl_riscv_rvv_util_float.h` | 276 | gather/scatter/transpose load-store (`vluxei32`/`vsuxei32`/`vrgather_vv`), the nonbonded-kernel data movement API |

**7 modified tracked files** wiring the backend into the build system and dispatch:

| file | change |
|---|---|
| `cmake/gmxManageSimd.cmake` | new `RISCV_RVV` branch: `GMX_SIMD_RISCV_RVV_LENGTH` (128/256/512/1024), flag detection, `SimdType::Riscv_Rvv` |
| `cmake/gmxSimdFlags.cmake` | `gmx_find_simd_riscv_rvv_flags()` → `-march=rv64gcv_zvl256b -mrvv-vector-bits=zvl` |
| `CMakeLists.txt` | `RISCV_RVV` added to the `GMX_SIMD` multichoice |
| `src/config.h.cmakein` | `GMX_SIMD_RISCV_RVV` + length `#cmakedefine`s |
| `src/gromacs/hardware/simd_support.h` | `SimdType::Riscv_Rvv` enum value |
| `src/gromacs/hardware/simd_support.cpp` | detection / name string |
| `src/gromacs/simd/include/gromacs/simd/simd.h` | `#elif GMX_SIMD_RISCV_RVV → #include impl_riscv_rvv/…` |

## Apply & build

```bash
# on the X60 board, with a pristine GROMACS v2026.2 git checkout:
./apply-and-build.sh /path/to/gromacs-2026.2
# → configures GMX_SIMD=RISCV_RVV, builds gmx + simd-test, prints the SIMD banner
```

The patch applies cleanly to `v2026.2` (`git apply --check` verified). If the
`impl_riscv_rvv/` directory already exists in your tree, remove it first (the
patch *creates* those files).

### Toolchain gotcha (bites at runtime, not build)

The built binaries need **GCC 14.3.0's** `libstdc++` (`CXXABI_1.3.15`). EESSI's
*compat* libstdc++ is GCC 13 and lacks it — running `gmx`/`simd-test` then fails
with `version 'CXXABI_1.3.15' not found`. Prepend the real one:

```bash
GCCLIB=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="$GCCLIB:$LD_LIBRARY_PATH"
```

`apply-and-build.sh` does this for you.

## Verification (measured on the X60)

- **SIMD unit tests: 158/158 float tests pass** — `SimdFloatingpointTest`,
  `SimdFloatingpointUtilTest`, `SimdMathTest` (log/exp/pow/erf/trig/pme, the full
  cascade that depends on the added `frexp`/`ldexp`), `SimdVectorOperationsTest`,
  `Simd4FloatingpointTest`, `Simd4MathTest`, `Simd4VectorOperationsTest`.
- **Real `gmx mdrun` (SPC water PME, serial, 2000 steps): 3.31× whole-app / 4.38× on `Force`** vs stock scalar, energy-correct. Full table and methodology in [`../README.md`](../README.md).

## Scope / caveats

- **Float (mixed) precision only.** `GMX_DOUBLE=OFF`. The double-precision SIMD
  API (`SimdDouble` etc.) is **not** ported yet.
- **VLEN=256 fixed** (`-mrvv-vector-bits=zvl`, X60). The CMake option exposes
  128/256/512/1024, but only 256 has been built/tested.
- Built serial (`GMX_MPI=OFF`, `GMX_GPU=OFF`), `GMX_FFT_LIBRARY=fftpack`. The FFT
  backend is an orthogonal axis (see the FFT A/B in `../README.md`).
- Base: GROMACS `v2026.2` (`da9e013175bae98b31b34384f6b4864ff29f65a5`).
