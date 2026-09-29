# EasyBuild / GCCcore — SpacemiT patches (GCC 14.3.0)

## Patches (apply in order)

| Order | Patch | Purpose |
|-------|-------|---------|
| 1 | [`GCC-14.3.0-spacemit-x60.patch`](GCC-14.3.0-spacemit-x60.patch) | X60 DFA + `xsmtvdot` / Layer B costs (RV2) |
| 2 | [`GCC-14.3.0-spacemit-x100-a100.patch`](GCC-14.3.0-spacemit-x100-a100.patch) | K3 `-mcpu=spacemit-x100` / `-mcpu=spacemit-a100` |

Both apply with `patch -p1` on stock **GCC 14.3.0** (`--fuzz=0` verified).

### X60 (patch 1)

Layer A then Layer B; **`type=shadd` deferred**. Finished RV2 semantics:
**atomic@12**, **memory_cost=4**, **vector_cost** wired, **clmul@2**, **no**
`type=shadd`.

### X100 / A100 (patch 2)

| `-mcpu` / `-mtune` | VLEN | Pipeline | Cost table |
|--------------------|------|----------|------------|
| `spacemit-x100` | 256 | `generic_ooo` | LLVM/trunk X100 (`issue_rate=4`) |
| `spacemit-a100` | 1024 | `spacemit_x60` | K3-measured A100 (`issue_rate=2`) |

Shared single-binary tip: prefer `-mcpu=spacemit-x100` (safe on A100).
A100-only / hetero A100 ranks: `-mcpu=spacemit-a100`.

## EasyBuild snippet

```python
# In a GCCcore-14.3.0 / GCC-14.3.0 easyconfig (illustrative — not submitted):
patches = [
    'GCC-14.3.0-spacemit-x60.patch',
    'GCC-14.3.0-spacemit-x100-a100.patch',
]
```

Place the patches next to the easyconfig (or in EasyBuild’s patch path).

## Still separate / still missing for EESSI

- **Binutils** IME encode: `patches/binutils/binutils-2.46.1_add-spacemit-xsmtvdot.patch`
  (GCC alone never encodes `smt.vmadot`).
- **`EASYBUILD_OPTARCH`**: only pass `-mtune=spacemit-x100` (or `-x60`) once
  this patched GCCcore is what hosts use; until then keep march-only (see
  `notes/eessi-wiring.md` in `spacemit-x60-gcc-tune`).
- **GCC 15.2**: sibling unified patch in [`../gcc-15.2/`](../gcc-15.2/).
- **`type=shadd`**: still deferred (`patches/deferred/0005b-…`).

Do **not** treat local RV2 / K3 proof as an EESSI PR.
