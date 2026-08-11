# EasyBuild / GCCcore — single SpacemiT X60 patch (GCC 14.3.0)

## Patch

`GCC-14.3.0-spacemit-x60.patch` — one file for stock **GCC 14.3.0** (EESSI
current GCCcore line).

Composition (Layer A then Layer B; **`type=shadd` deferred**):

| Layer | Contents |
|-------|----------|
| A | tune/DFA (`spacemit-x60`) + table-form `xsmtvdot` / `xsmtvdotii` |
| B | 0001 → 0001b → 0002 → 0003 → 0005(**clmul-only**) → 0004 → 0006 |

Finished RV2 semantics: **atomic@12**, **memory_cost=4**, **vector_cost**
wired, **clmul@2**, **no** `type=shadd`.

## Apply

From the extracted `gcc-14.3.0` source root (EasyBuild does this via
`patches = [...]` with `patch -p1`):

```bash
patch -p1 < GCC-14.3.0-spacemit-x60.patch
```

Proof (fresh stock tree, `--fuzz=0`):
`results/easybuild-unified-14.3-verify.log` (layered:
`results/pristine-apply-14.3.log`).

## EasyBuild snippet

```python
# In a GCCcore-14.3.0 / GCC-14.3.0 easyconfig (illustrative — not submitted):
patches = [
    'GCC-14.3.0-spacemit-x60.patch',
]
```

Place the patch next to the easyconfig (or in EasyBuild’s patch path).

## Still separate / still missing for EESSI

- **Binutils** IME encode: `patches/binutils/binutils-2.46.1_add-spacemit-xsmtvdot.patch`
  (GCC alone never encodes `smt.vmadot`).
- **`EASYBUILD_OPTARCH`**: only pass `-mtune=spacemit-x60` once this patched
  GCCcore is what hosts use; until then keep march-only (see
  `notes/eessi-wiring.md` in `spacemit-x60-gcc-tune`).
- **GCC 15.2**: sibling unified patch in [`../gcc-15.2/`](../gcc-15.2/).
- **`type=shadd`**: still deferred (`patches/deferred/0005b-…`).

Do **not** treat local RV2 proof as an EESSI PR.
