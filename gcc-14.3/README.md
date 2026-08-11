# GCC 14.3 — SpacemiT X60 EasyBuild patch + RV2 canaries

Stock **GCC 14.3.0** pipeline/tune patch for SpacemiT X60, plus Orange Pi RV2
measurements that isolate **`-mtune=spacemit-x60`** vs **`-mtune=generic-ooo`**.

Upstream work lives in
[`spacemit-x60-gcc-tune`](https://github.com/opensolvers/spacemit-x60-gcc-tune)
(local clone used for this snapshot). This directory is the **EasyBuild-facing
artifact** + key canary tables for the benchmarks repo (EESSI’s current GCCcore
line).

> **Change one variable.** Same `-march`, same canary binary — only GCC
> **mtune** (and the patch that teaches GCC about X60) differs.

> **Bottom line (RV2):** canaries `load_add_chain` **−1.45%**, `fma_chain`
> **−5.04%**, `div_mix` **−6.73%**, `sh1add` **−0.20%** ns/call vs
> `generic-ooo`. OpenBLAS DGEMM **−6.9/−6.7/−3.0%** at N=512/1024/2048 (sign
> flip vs 15.2); modest HPL N=3000 **+6.8%** (both PASSED). Local proof only.

---

## Patch

[`GCC-14.3.0-spacemit-x60.patch`](GCC-14.3.0-spacemit-x60.patch) — one file for
stock GCC 14.3.0 (`patch -p1`).

Composition (Layer A then Layer B; **`type=shadd` deferred**):

| Layer | Contents |
|-------|----------|
| A | tune/DFA (`spacemit-x60`) + table-form `xsmtvdot` / `xsmtvdotii` |
| B | 0001 → 0001b → 0002 → 0003 → 0005(**clmul-only**) → 0004 → 0006 |

Finished RV2 semantics: **atomic@12**, **memory_cost=4**, **vector_cost**
wired, **clmul@2**, **no** `type=shadd`.

Apply from the extracted `gcc-14.3.0` source root (EasyBuild via
`patches = [...]`):

```bash
patch -p1 < GCC-14.3.0-spacemit-x60.patch
```

Pristine apply proof (`--fuzz=0`):
[`results/easybuild-unified-14.3-verify.log`](results/easybuild-unified-14.3-verify.log)
(layered walkthrough:
[`results/pristine-apply-14.3.log`](results/pristine-apply-14.3.log)).

EasyBuild sketch:

```python
# In a GCCcore-14.3.0 / GCC-14.3.0 easyconfig (illustrative — not submitted):
patches = [
    'GCC-14.3.0-spacemit-x60.patch',
]
```

See also [`EASYBUILD-NOTE.md`](EASYBUILD-NOTE.md) (still need binutils IME encode
separately; do not set `EASYBUILD_OPTARCH=-mtune=spacemit-x60` until hosts use
this patched GCCcore).

---

## Results (Orange Pi RV2)

### Scheduler canaries

Source set: `rv2-gcc143-x60-ab` (install
`/home/orangepi/gcc-tune/install/gcc-14.3.0-x60`, 2026-08-11).

| Kernel | Δ% x60 vs `generic-ooo` (mean ns/call ↓) |
|--------|------------------------------------------|
| `load_add_chain` | **−1.45%** |
| `fma_chain` | **−5.04%** |
| `div_mix` | **−6.73%** |
| `sh1add` | **−0.20%** |

Full tables + asm/logs: [`results/canaries/`](results/canaries/).

**Omitted from git (rebuildable):** ELF `bin/bench-*` / `x60_latency_probe`,
static OpenBLAS `.a`.

### OpenBLAS DGEMM + modest HPL

[`results/openblas-hpl/`](results/openblas-hpl/) — same harness as 15.2
(`RISCV64_ZVL256B`, static, interleaved DGEMM, HPL N=3000 NB=192 2×4).

| Metric | 14.3 Δ% (x60 vs ooo) | 15.2 Δ% |
|--------|----------------------|---------|
| DGEMM N=512 | **−6.88%** | +2.19% |
| DGEMM N=1024 | **−6.69%** | +2.25% |
| DGEMM N=2048 | **−3.02%** | +3.79% |
| HPL N=3000 | **+6.75%** | +0.79% |

---

## Layout

```
gcc-14.3/
  README.md
  GCC-14.3.0-spacemit-x60.patch
  EASYBUILD-NOTE.md
  results/
    easybuild-unified-14.3-verify.log
    pristine-apply-14.3.log
    canaries/          # scheduler A/B (no ELF bins)
    openblas-hpl/      # DGEMM + modest HPL mtune A/B
```

Related: [`gcc-15.2/`](../gcc-15.2/), ISA notes in [`cores/x60/`](../cores/x60/).
