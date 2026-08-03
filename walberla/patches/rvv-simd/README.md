# RVV SIMD backend for waLBerla (Orange Pi RV2)

## Problem
GCC rejects scalable RVV types as struct members:
`member variables cannot have RVV type 'vfloat64m2_t'`.
The first backend stored `alignas(32) double d[4]` and did
load→compute→store around every op (heavy spill).

## Fix (current): GCC `vector_size(32)`
`src/simd/RVV.h`: `double4_t` holds a GNU fixed-size vector:

```cpp
typedef double f64x4_t __attribute__((vector_size(32)));
struct double4_t { f64x4_t v; };
```

Arithmetic is `a.v + b.v` (etc.). With `-march=rv64gcv`, GCC lowers these
to RVV and can keep values in registers / form FMAs. Lane order matches
`Scalar.h` / SSE. Compares use GCC vector relational ops (all-ones / zero
integer lanes, reinterpreted as `f64`), which match SSE mask bit patterns
for `blendv` / `movemask` well enough for the microbench.

`usedInstructionSet()` → `"RVV"`.

## Companion patches
- `SIMD.h` — detect `__riscv_vector`/`__riscv_v`, select RVV before scalar fallback;
  honor `WALBERLA_SIMD_FORCE_SCALAR`.
- `FieldAllocator.h` — `SIMDAlignment()` returns `32u` under RVV.

## Apply on board
```bash
SRC=~/walberla-bench/src/walberla-7.2
cp RVV.h          $SRC/src/simd/RVV.h
cp SIMD.h         $SRC/src/simd/SIMD.h   # or merge the RVV blocks
# FieldAllocator.h: ensure the __riscv_vector branch returns 32u
```

Smoke / equiv / SoA scripts: `walberla/scripts/rvv-simd-*.sh`,
`walberla/scripts/rvv-soa-microbench.sh`.

## Caveats
- GNU `vector_size` is GCC-oriented (this A/B used foss/GCC 14.3 only).
- Shuffle-heavy helpers (`hadd`, rotates, `blend<mask>`, `invSqrt`) still
  use lane extracts; hot SoA collide path is mostly `+/-/*/`, `sqrt`, load/store.
- `double4_t fq[Q]` arrays can still spill some vectors to the stack; the win
  is fewer per-op vle/vse than the POD backend, not zero memory traffic.
