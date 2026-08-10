# GCC 15.2 — SpacemiT X60 EasyBuild patch + RV2 A/Bs

Stock **GCC 15.2.0** pipeline/tune patch for SpacemiT X60, plus Orange Pi RV2
measurements that isolate **`-mtune=spacemit-x60`** vs **`-mtune=generic-ooo`**.

Upstream work lives in
[`spacemit-x60-gcc-tune`](https://github.com/opensolvers/spacemit-x60-gcc-tune)
(local clone used for this snapshot). This directory is the **EasyBuild-facing
artifact** + key result tables for the benchmarks repo.

> **Change one variable.** Same `-march`, same OpenBLAS target / HPL config —
> only GCC **mtune** (and the patch that teaches GCC about X60) differs.

> **Bottom line (RV2, clean canaries):** `fma_chain` **−8.7%**, `div_mix`
> **−7.7%** ns/call vs `generic-ooo`. OpenBLAS DGEMM **+2.2–3.8%** GF/s;
> modest HPL N=3000 **+0.8%**. Local proof only — not an EESSI PR yet.

---

## Patch

[`GCC-15.2.0-spacemit-x60.patch`](GCC-15.2.0-spacemit-x60.patch) — one file for
stock GCC 15.2.0 (`patch -p1`).

Composition (Layer A then Layer B; **`type=shadd` deferred**):

| Layer | Contents |
|-------|----------|
| A | tune/DFA (`spacemit-x60`) + table-form `xsmtvdot` / `xsmtvdotii` |
| B | 0001 → 0001b → 0002 → 0003 → 0005(**clmul-only**) → 0004 → 0006 |

Finished RV2 semantics: **atomic@12**, **memory_cost=4**, **vector_cost**
wired, **clmul@2**, **no** `type=shadd`.

Apply from the extracted `gcc-15.2.0` source root (EasyBuild via
`patches = [...]`):

```bash
patch -p1 < GCC-15.2.0-spacemit-x60.patch
```

Pristine apply proof (`--fuzz=0`):
[`results/easybuild-unified-verify.log`](results/easybuild-unified-verify.log).

EasyBuild sketch:

```python
# In a GCCcore-15.2.0 / GCC-15.2.0 easyconfig (illustrative — not submitted):
patches = [
    'GCC-15.2.0-spacemit-x60.patch',
]
```

See also [`EASYBUILD-NOTE.md`](EASYBUILD-NOTE.md) (still need binutils IME encode
separately; do not set `EASYBUILD_OPTARCH=-mtune=spacemit-x60` until hosts use
this patched GCCcore).

---

## Results (Orange Pi RV2)

### Scheduler canaries

Source set: `rv2-gcc152-x60-ab-clean` (revalidation after pristine Layer B).

| Kernel | Δ% x60 vs `generic-ooo` (mean ns/call ↓) |
|--------|------------------------------------------|
| `load_add_chain` | **−1.4%** |
| `fma_chain` | **−8.7%** |
| `div_mix` | **−7.7%** |
| `sh1add` | **−1.0%** |

Full tables + asm/logs: [`results/canaries/`](results/canaries/).

### OpenBLAS DGEMM + HPL

Source set: `rv2-gcc152-openblas-hpl-ab` (static OpenBLAS `RISCV64_ZVL256B`,
same march; only mtune differs).

| Axis | Δ% (x60 vs ooo) |
|------|-----------------|
| DGEMM N=512 / 1024 / 2048 | **+2.2% / +2.3% / +3.8%** GF/s |
| HPL N=3000, NB=192, 2×4 | **+0.8%** Gflops (both PASSED) |

Summaries, raw outs, harness: [`results/openblas-hpl/`](results/openblas-hpl/).

**Omitted from git (too large / rebuildable):** static `libs/*.a`, ELF
`bin/bench-*` / `xhpl-*`, multi-MB OpenBLAS build/`nohup` logs.

---

## Layout

```
gcc-15.2/
  README.md
  GCC-15.2.0-spacemit-x60.patch
  EASYBUILD-NOTE.md
  results/
    easybuild-unified-verify.log
    canaries/          # clean scheduler A/B
    openblas-hpl/      # OpenBLAS + HPL mtune A/B
```

Related ISA notes in this repo: [`cores/x60/`](../cores/x60/).
