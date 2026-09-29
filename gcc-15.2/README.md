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

## Patch (K3 X100 / A100)

[`GCC-15.2.0-spacemit-x100-a100.patch`](GCC-15.2.0-spacemit-x100-a100.patch) —
adds `-mcpu`/`-mtune=spacemit-x100` and `spacemit-a100` for SpacemiT K3.
**Requires** the X60 patch first (A100 reuses the X60 DFA). Uses the GCC 15.2
`riscv_tune_param` layout (extra fields vs 14.3).

| `-mcpu` / `-mtune` | Pipeline | Costs |
|--------------------|----------|-------|
| `spacemit-x100` | `generic_ooo` | LLVM/trunk (`issue_rate=4`) |
| `spacemit-a100` | `spacemit_x60` | K3-measured (`issue_rate=2`) |

```bash
patch -p1 < GCC-15.2.0-spacemit-x60.patch
patch -p1 < GCC-15.2.0-spacemit-x100-a100.patch
```

EESSI RISC-V currently has GCCcore 14.x only — this pair is for EasyBuild /
future 15.x.

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
patch -p1 < GCC-15.2.0-spacemit-x100-a100.patch
# optional native backport (below):
patch -p1 < GCC-15.2.0-riscv-march-native.patch
patch -p1 < GCC-15.2.0-riscv-mcpu-mtune-native.patch
```

Pristine apply proof (`--fuzz=0`):
[`results/easybuild-unified-verify.log`](results/easybuild-unified-verify.log)
(native+SpacemiT:
[`results/native-after-spacemit-verify.log`](results/native-after-spacemit-verify.log)).

EasyBuild sketch:

```python
# In a GCCcore-15.2.0 / GCC-15.2.0 easyconfig (illustrative — not submitted):
patches = [
    'GCC-15.2.0-spacemit-x60.patch',
    'GCC-15.2.0-spacemit-x100-a100.patch',
    'GCC-15.2.0-riscv-march-native.patch',
    'GCC-15.2.0-riscv-mcpu-mtune-native.patch',
]
```

See also [`EASYBUILD-NOTE.md`](EASYBUILD-NOTE.md) (binutils IME1+IME2 asm under
`../patches/binutils/`; do not set `EASYBUILD_OPTARCH=-mtune=spacemit-x60`
until hosts use this patched GCCcore).

---

## Patch (RISC-V `-march`/`-mcpu`/`-mtune=native`)

Same approach as [`../gcc-14.3/`](../gcc-14.3/): stock 15.2 rejects `native` on
RISC-V; trunk’s hwprobe + 6-arg `RISCV_CORE` stack is too invasive here.
Self-contained backport (does **not** change `riscv-cores.def` shape):

| Patch | Adds |
|-------|------|
| [`GCC-15.2.0-riscv-march-native.patch`](GCC-15.2.0-riscv-march-native.patch) | `driver-riscv.cc`, minimal `riscv-hwprobe.h`, `riscv_ext_is_known_p`, `-march=native` via `/proc/cpuinfo` (+ `vlenb` → `zvl*b`) |
| [`GCC-15.2.0-riscv-mcpu-mtune-native.patch`](GCC-15.2.0-riscv-mcpu-mtune-native.patch) | `-mcpu=native` / `-mtune=native`; hardcoded ID table (SiFive u74/p550; SpacemiT x60/x100/a100) |

Either apply order (SpacemiT ↔ native) is `--fuzz=0` clean; recommended
SpacemiT first so native can resolve SpacemiT names. Do **not** reuse the
14.3 native patches on 15.2 (and vice versa) — source context differs.

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
  GCC-15.2.0-spacemit-x100-a100.patch
  GCC-15.2.0-riscv-march-native.patch
  GCC-15.2.0-riscv-mcpu-mtune-native.patch
  EASYBUILD-NOTE.md
  results/
    easybuild-unified-verify.log
    native-after-spacemit-verify.log
    native-only-verify.log
    canaries/          # clean scheduler A/B
    openblas-hpl/      # OpenBLAS + HPL mtune A/B
```

Related ISA notes in this repo: [`cores/x60/`](../cores/x60/).
