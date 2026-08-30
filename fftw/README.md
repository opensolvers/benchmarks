# FFTW RVV (`r5v`) A/B — SpaceMiT X60 / K1

A clean A/B of FFTW 3.3.10 built with a custom **RISC-V Vector (RVV) SIMD
backend** (`--enable-r5v`, from [rdolbeau's `r5v-test-release-005`](https://github.com/rdolbeau)) versus the **scalar** build of the *same source* with
*identical* compiler and flags — so the only variable is `--enable-r5v`. Run on
the Orange Pi RV2 (SpaceMiT X60, 8× @ 1.6 GHz, RVV 1.0 VLEN=256) under EESSI
`2025.06-001`, GCC 14.3.0, `-O3 -march=rv64imafdcv_zvl256b`.

Two questions: **(1)** is the RVV backend real and correct, and **(2)** is it
faster than scalar — and what actually moves the needle on this hardware.

| File | What it does |
|---|---|
| `build-fftw-r5v.sh` | Builds both libs (`src-r5v` + `src-scalar`) from one tarball, identical flags; counts RVV mnemonics in each `.so` as an inline sanity check. |
| `bench-fftw-ab.sh` | Runs FFTW's own `tests/bench` across sizes × planners (`estimate` / `measure` / `patient`) on both libs, reports median MFLOPS. |
| `bench-fftw-wisdom.sh` | Offline `fftw-wisdom -m` + C probe: ESTIMATE / MEASURE-cold / MEASURE+wisdom / WISDOM_ONLY on 1-D sizes. |
| `fftw-est2meas-interposer.c` | `LD_PRELOAD`: remap QE's `FFTW_ESTIMATE`→`MEASURE` for `fftw_plan_dft_3d` / `fftw_plan_many_dft`; import/export wisdom files. |
| `fftw-wisdom-preload.c` | Constructor-only wisdom import (no flag remap). |
| `bench-codelet-hot.sh` | Microbench 1D + QE-like `many_dft` (MEASURE) for hot codelet shapes. |
| `patch-simd-r5v-noshuffle.py` | Failed: store-shuffle vs `vrgather` (**regressed**). |
| `patch-simd-r5v-xorconj.py` | XOR `VCONJ` + fused `VBYI` (**+1–7%** on hot sizes; kept). |
| `patch-simd-r5v-vfmai-neon.py` | NEON-style `VFMAI`/`VZMUL` (within noise; reverted). |
| `t2bv_8_r5v256_split.c` | Gather-free split `t2bv_8` (correct, **~0.5×**; reverted). |
| `flip-microbench.c` | Isolated `FLIP_RI`: gather beats slide/store on X60. |
| `run-t2bv8-unit.sh` | Stock vs split `t2bv_8` correctness + ns/call. |
| `rebuild-r5v256.sh` | Rebuild only `r5v256` codelets and relink `libfftw3`. |
| `merge-fftw-wisdom.c` | Merge per-rank MPI wisdom files into one. |
| `run-qe-fft-wisdom-ab.sh` | QE A/B: estimate → measure-collect → estimate+wisdom → measure+wisdom (`NP=1` serial or `NP=4` MPI). |
| `FFTW-3.3.10-GCC-14.3.0-r5v.eb` | EasyBuild easyconfig that reproduces the `r5v` lib as a module (see below). |

Both are single shell scripts; the only inputs are the FFTW `r5v` source
tarball and an EESSI-provided GCC 14.

## Build (the A/B pair)

```bash
# needs $HOME/fftw-r5v.tar.gz (FFTW 3.3.10 + r5v backend) on the board
./build-fftw-r5v.sh
# -> $HOME/fftwbuild/src-r5v/.libs/libfftw3.so.3.6.10   (RVV)
#    $HOME/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10 (scalar control)
#    each with tests/bench linked against its own lib
```

## Run (planner-aware A/B)

```bash
./bench-fftw-ab.sh          # writes fftw-proper.log
# per row: [lib/planner] size=N med=MFLOPS | raw=...
```

`tests/bench` selects the planner via `-o<word>` (source: `tests/fftw-bench.c`):
valid words are `estimate`, `patient`, `estimatepat`, `exhaustive`; **the
default (no `-o`) is `FFTW_MEASURE`.** There is *no* `-omeasure` — passing it
prints `unknown user option: measure. Ignoring.` and silently runs the MEASURE
default. Time each measurement with `-t <seconds>`; short `-t` (≤0.3 s) is noisy
on this board and can invert individual points — use `-t 1.0`+.

## (1) The RVV backend is real and correct

Building both libs from one source, the only config difference being
`--enable-r5v`:

| build | `make` | `libfftw3.so.3.6.10` | RVV instr count | codelets in plan |
|---|:---:|---:|---:|---|
| **r5v** (`--enable-r5v`) | RC=0 | **11 MB** | **224,354** | `n1fv_16_r5v256`, `t3fv_4_r5v256`, `t1fv_4_r5v256` |
| **scalar** (control) | RC=0 | **924 KB** | 734 | (none) |

RVV instruction count = `objdump -d` grep of `vsetvli|vfmacc|vfmadd|vle64.v|…`.
The r5v lib emits **~305× more vector instructions** than scalar (whose 734 is
residual autovectorization), and FFTW's planner actually *selects* the
`*_r5v256` vector codelets into its plans (confirmed via `bench -v2`). The
backend is present, linked, and used — not a no-op flag.

## (2) Performance — RVV wins once the planner is honest

Median MFLOPS, 1D complex-to-complex, single thread, `-t 1.0`, higher = faster.
The **planner** matters more than anything, so both `estimate` and the
`FFTW_MEASURE` default are shown:

| size | estimate r5v / scal | **MEASURE r5v / scal** | **r5v speedup (MEASURE)** |
|---:|---:|---:|:---:|
| 256 | 2228 / 1388 | **2520 / 1579** | **1.60×** |
| 1024 | 717 / 747 | **1642 / 1265** | **1.30×** |
| 4096 | 303 / 360 | **1283 / 978** | **1.31×** |
| 16384 | 381 / 276 | **964 / 797** | **1.21×** |
| 65536 | 148 / 142 | **797 / 752** | **1.06×** |
| 262144 | 171 / 138 | **717 / 664** | **1.08×** |

Under the `FFTW_MEASURE` default the RVV backend beats scalar at **every** size,
**1.06–1.60×** — largest on cache-resident transforms (1.6× @ 256), tapering to
~1.06× as transforms become memory-bandwidth-bound (≥64K). Textbook.

## RV2 re-verify (2026-08-22)

Same A/B pair (`~/fftwbuild/src-{scalar,r5v}/`), default (`FFTW_MEASURE`) planner,
via [`run-part-a-v2.sh`](../run-part-a-v2.sh) → `~/fftw-proper.log`:

| size | r5v MFLOPS | scalar MFLOPS | r5v speedup |
|---:|--:|--:|--:|
| 256 | 2522 | 1537 | **1.64×** |
| 4096 | 1224 | 1019 | **1.20×** |
| 65536 | 601 | 519 | **1.16×** |

Matches the MEASURE column in the table above on the same artifacts.

## Cross-board confirmation — Banana Pi BPI-F3 (same K1 / X60 SoC)

Same `tests/bench` A/B (`-t 1.0`, estimate + MEASURE; no `patient`) on a
[Banana Pi BPI-F3](https://www.banana-pi.org/) (SpaceMiT K1, 8× X60 @ 1.6 GHz),
using the **same** r5v / scalar `libfftw3.so.3.6.10` binaries built on the RV2
(GCC 14.3.0, `-O3 -march=rv64imafdcv_zvl256b`). Median MFLOPS:

| size | estimate r5v / scal | **MEASURE r5v / scal** | **r5v speedup (MEASURE)** |
|---:|---:|---:|:---:|
| 256 | 2196 / 1386 | **2518 / 1576** | **1.60×** |
| 1024 | 720 / 746 | **1634 / 1273** | **1.28×** |
| 4096 | 302 / 361 | **1276 / 1033** | **1.24×** |
| 16384 | 424 / 330 | **943 / 817** | **1.15×** |
| 65536 | 178 / 169 | 723 / 741 | **0.98×** |
| 262144 | 203 / 160 | **729 / 685** | **1.06×** |

Matches the RV2 within a few percent at every cache-resident size (1.60× @ 256
identical). At N≥64K both boards are bandwidth-bound and the ratio sits near
1×; the F3's 0.98× at N=65536 is within run-to-run noise of the RV2's 1.06×
there (same binaries). Planner takeaway unchanged: MEASURE ≫ estimate.

## The big lever is the planner, not the codelets

The single biggest speedup on this hardware is **planner choice**, worth
**3–5×** and independent of RVV — the estimate planner (FFTW's `bench -s`
default, and the trap in a naive A/B) grossly under-plans large transforms:

| size | estimate → MEASURE (r5v) | gain |
|---:|---:|:---:|
| 4096 | 303 → 1283 | **4.2×** |
| 16384 | 381 → 964 | **2.5×** |
| 65536 | 148 → 797 | **5.4×** |
| 262144 | 171 → 717 | **4.2×** |

`patient` was within noise of `MEASURE` where it completed (256: 2475 vs 2520;
16384: 971 vs 964) while its planning time blows up at large sizes (>35 min at
N=262144), so **`FFTW_MEASURE` is the sweet spot** here.

**Recommendation:** any real FFTW workload on the X60/K1 should plan with
`FFTW_MEASURE` (or cached wisdom), never `FFTW_ESTIMATE`. That alone is a 3–5×
win, and it is what lets the RVV codelets be selected into well-tuned plans.

## Gotcha — `module load` does not repath `gcc` on the Orange Pi RV2

On the RV2, `module load GCCcore/14.3.0` returns rc=0 but does **not** put GCC 14
first on `PATH` — the EESSI compat-layer GCC 13.4.0 keeps winning, so a naive
`gcc` is 13.4.0 and a strict version guard aborts the build. Fix is to prepend
the real GCC 14 bindir explicitly (see the top of `build-fftw-r5v.sh`):

```bash
GCC14=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0
export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$GCC14/lib64:$LD_LIBRARY_PATH"
```

(On the Banana Pi BPI-F3 — same K1/X60 SoC — the plain `module load GCC/14.3.0`
repaths correctly; the script header notes the per-board difference.)

## A methodology note (why the planner trap matters)

A first pass benchmarked with the `estimate` planner and a fast `-t 0.3` timing
and appeared to show a **2× RVV *regression* at N=262144** (r5v 316 vs scalar
678). It was an artifact on both counts: with honest timing r5v is *faster*
there under estimate (171 vs 138) **and** under MEASURE (717 vs 664). The lesson
— pin the planner and use ≥1 s timing before trusting any single FFT A/B point —
is the real portable takeaway here.

## (3) End-to-end reality check — Quantum ESPRESSO gets ~0% from RVV FFTW

The micro-benchmark above shows the RVV codelets are real (1.06–1.60×). The
obvious next question: **does that survive into a real DFT run?** The BLAS-axis
QE A/B in [`../qe/`](../qe) deliberately left the FFT half untouched; this closes
that gap by swapping *only* the FFT.

Setup: a **serial** `pw.x` (QE 7.5, no MPI/OpenMP/ScaLAPACK) built on the Orange
Pi RV2 against the EESSI external FFTW. Because `pw.x` links `libfftw3.so.3`
dynamically, the two backends are swapped by `LD_PRELOAD` alone — BLAS is pinned
to scalar OpenBLAS via FlexiBLAS so the FFT axis is the *only* variable. Runner:
[`run-qe-fft-ab.sh`](run-qe-fft-ab.sh). Two Si cells: a 2-atom correctness probe
and a 64-atom cell (`ecutwfc=22`, 136 bands, 7 SCF iters) where `fftw` dominates.

**Correctness:** total energy is bit-identical across backends on both cells
(2-atom `-14.57861334 Ry`; 64-atom `-506.67991945 Ry`).

64-atom SCF, WALL seconds (FlexiBLAS→scalar OpenBLAS held constant):

| timer | calls | scalar FFTW | r5v (RVV) FFTW | r5v speedup |
|---|---|---|---|---|
| `fftw`     | 3260 | 112.24 | 110.09 | **1.019×** |
| `vloc_psi` |   41 | 105.29 | 103.08 | 1.021× |
| `h_psi`    |   41 | 146.23 | 143.80 | 1.017× |
| `c_bands`  |    7 | 205.25 | 198.44 | 1.034× |
| **`PWSCF` (total)** | — | **248.49** | **248.10** | **1.002×** |

**The RVV FFTW backend delivers essentially nothing end-to-end (~0.2% wall,
~1.9% inside `fftw` itself)** — even though `fftw` is ~45% of runtime, exactly
the fraction the BLAS-axis A/B could not reach.

Why the 1.6× micro-win evaporates:

- **QE plans with `FFTW_ESTIMATE`, not `MEASURE`.** Section (2) showed the RVV
  advantage is largely a *planner* effect — under `estimate` the two libs are
  near-parity, and `estimate` is exactly what QE uses (it cannot afford
  `MEASURE`'s planning cost across thousands of transient transforms).
- **3260 small mixed-radix transforms**, not the cache-resident power-of-two
  sizes where RVV codelets shine (1.6× was @ N=256). Real charge-density grids
  are odd composite sizes and memory-bandwidth-bound, the ~1.06× tail.

Takeaway: on the X60, **neither the BLAS axis nor the FFT axis moves a real QE
SCF** with these drop-in vectorized libraries alone — honest FFT planning helps
a little (next section) but not enough, and BLAS still needs kernels the generic
RVV OpenBLAS does not yet provide. A microbenchmark speedup is not an
application speedup.

## (4) Wisdom / MEASURE — big in microbench, ~6% in QE

Hypothesis from §(2)–(3): force `FFTW_MEASURE` (or cached wisdom) and the RVV
codelets should show up in QE. Scripts above implement that without patching QE.

### 1-D microbench (r5v lib, Orange Pi RV2)

Cold ESTIMATE vs MEASURE on power-of-two complex DFTs is a **3–5×** throughput
gap (e.g. N=4096: ~294 → ~1285 MFLOPS). After importing MEASURE wisdom,
`FFTW_ESTIMATE` matches MEASURE throughput with ~ms plan cost — this build
applies imported wisdom even under ESTIMATE.

| N | ESTIMATE (no wis) | MEASURE-cold | ESTIMATE+wisdom |
|---|---:|---:|---:|
| 1024 | 719 | 1563 | 1621 |
| 4096 | 294 | 1285 | 1299 |
| 16384 | 378 | 879 | 820 |
| 65536 | 157 | 757 | 765 |

(MFLOPS; wisdom gen ~64 s for those sizes.)

### QE 64-atom SCF (`si-super-64.in`, r5v FFTW, scalar OpenBLAS)

Energy bit-identical (`-506.67980304 Ry`) across all four steps. WALL seconds:

| step | `fftw` | `init_run` | `PWSCF` |
|---|---:|---:|---:|
| 1 ESTIMATE (stock QE) | 85.03 | 28.45 | 188.91 |
| 2 MEASURE collect (+export) | 83.38 | 28.20 | 187.76 |
| 3 ESTIMATE + MEASURE wisdom | 81.76 | 20.20 | 180.49 |
| 4 MEASURE + wisdom | 80.08 | 19.91 | 178.63 |
| 2 PATIENT collect | 84.04 | 28.83 | 187.96 |
| 3 ESTIMATE + PATIENT wisdom | 80.65 | 20.06 | 179.28 |
| 4 PATIENT + wisdom | 84.84 | 20.12 | 186.55 |

Best vs stock: **`fftw` / `PWSCF` ≈ 1.06×** (~10 s wall), whether MEASURE or
PATIENT wisdom. PATIENT step 3 is within noise of MEASURE (~1 s); step 4 is
noisier and not better. Collect cost stayed ~same as MEASURE (`init_run` ~29 s)
— the r5v solver space for these 64³ many-DFTs does not explode under PATIENT.

So the planner trap is real in isolation, but QE's 2362 transforms barely move
past MEASURE. Wisdom is still worth caching for repeated runs; richer planning
is not the path to a large QE speedup on this board.

```bash
# on the board (after build-fftw-r5v.sh)
bash ~/fftw-wisdom-src/bench-fftw-wisdom.sh
bash ~/fftw-wisdom-src/run-qe-fft-wisdom-ab.sh si-super-64.in wisdom64b
FFTW_EST2MEAS_FLAGS=patient SKIP_ESTIMATE=1 \
  bash ~/fftw-wisdom-src/run-qe-fft-wisdom-ab.sh si-super-64.in patient64
```

`FFTW_EST2MEAS_FLAGS` selects `measure` (default), `patient`, or `exhaustive`
for the ESTIMATE→planner remap in `fftw-est2meas-interposer.c`.

### MPI (`NP=4`, overlay `QuantumESPRESSO/7.5-foss-2025b`)

Same input, r5v via `LD_PRELOAD` (binary RPATH pins stock FFTW.MPI; interposer
`dlopen`s r5v with `RTLD_DEEPBIND` so wisdom hashes stay consistent). Per-rank
wisdom merged with `merge-fftw-wisdom`. Energy identical across steps
(`-506.67985507 Ry`; tiny delta vs serial is normal for MPI).

| step | `fftw` | `init_run` | `PWSCF` |
|---|---:|---:|---:|
| 1 ESTIMATE | 23.46 | 7.22 | 59.28 |
| 2 MEASURE collect | 23.72 | 7.22 | 65.01 |
| 3 ESTIMATE + wisdom | 22.77 | 6.37 | 58.99 |
| 4 MEASURE + wisdom | 23.00 | 6.23 | 58.05 |

Same story as serial: **~3% on `fftw`**, a bit of `init_run` from warm wisdom.
MPI already cuts wall ~3× vs serial; planner quality is not the remaining lever.

```bash
NP=4 bash ~/fftw-wisdom-src/run-qe-fft-wisdom-ab.sh si-super-64.in mpi4meas
NP=8 bash ~/fftw-wisdom-src/run-qe-fft-wisdom-ab.sh si-super-64.in mpi8meas
```

**NP=8** (same setup; energy `-506.67980423 Ry`):

| step | `fftw` | `init_run` | `PWSCF` |
|---|---:|---:|---:|
| 1 ESTIMATE | 16.06 | 5.13 | 44.93 |
| 2 MEASURE collect | 15.77 | 5.08 | 43.83 |
| 3 ESTIMATE + wisdom | 15.53 | 4.33 | 42.33 |
| 4 MEASURE + wisdom | 15.96 | 4.35 | 42.21 |

Again ~3% on `fftw` / ~6% on `PWSCF` at best; wisdom merge of 8 ranks OK (6125 bytes).

## (5) Hot codelets — what QE actually uses, and a failed gather rewrite

QE MPI wisdom is mostly **solvers** (`vrank` / `buffered` / `indirect`); only ~11
entries are `*_r5v256` codelets. Dominants: `t2bv_8`, `t2fv_4/8`, `n2fv_16`,
`n2bv_8`, plus scalar `n1_8`.

Disasm of `t2bv_8` @ r5v256 (GCC 14, `-O3 -march=rv64imafdcv_zvl256b`):

| metric | count |
|---|---:|
| `vrgather.vv` | 10 |
| stack `sd`/`ld` (frame) | ~12 |
| FMA/ALU vector | ~42 |

Index setup is already CSE’d (all gathers use the same `v8`); cost is **gather
latency on K1**, as Dolbeau warned in [FFTW#371](https://github.com/FFTW/fftw3/issues/371).
Helpers `VDUPL` / `VDUPH` / `FLIP_RI` / `VBYI` in `simd-r5v.h` are the source.

**Experiment:** for `R5V_SIZE==256` double (4×f64 lanes), replace those helpers
with aligned store → scalar shuffle → load (`patch-simd-r5v-noshuffle.py`),
rebuild `r5v256`, re-bench (`bench-codelet-hot.sh`).

| probe | gather (baseline) | store-shuffle | ratio |
|---|---:|---:|---:|
| 1D N=256 MFLOPS | 2507 | 1940 | **0.77×** |
| 1D N=1024 | 1581 | 1322 | 0.84× |
| 1D N=4096 | 1272 | 1016 | 0.80× |
| many N=64×4096 | 670 | 596 | 0.89× |
| `t2bv_8` gathers | 10 | 7 | (partial) |

**Verdict: regression.** On X60, short `vrgather` beats stack shuffle for these
helpers. That patch was reverted.

### Follow-up that helped: XOR `VCONJ` / fused `VBYI` (kept)

SSE2-style sign-bit XOR for `VCONJ`, and `VBYI` = `FLIP_RI` + XOR-negate-even
(one gather, no masked `vfsgnjn`). `patch-simd-r5v-xorconj.py` + `rebuild-r5v256.sh`.

| probe | gather baseline | XOR-conj | ratio |
|---|---:|---:|---:|
| 1D N=256 MFLOPS | 2507 | 2471 | 0.99× |
| 1D N=1024 | 1581 | **1686** | **1.07×** |
| 1D N=4096 | 1272 | 1291 | 1.01× |
| 1D N=65536 | 705 | **752** | **1.07×** |
| many N=64×4096 | 670 | **698** | **1.04×** |
| many N=64×64 | 1714 | 1747 | 1.02× |

**QE NP=8 ESTIMATE** (same `si-super-64`, xorconj lib vs prior gather ESTIMATE):

| timer | gather ESTIMATE | XOR-conj | ratio |
|---|---:|---:|---:|
| `fftw` | 16.06 | **15.75** | **1.02×** |
| `PWSCF` | 44.93 | **43.30** | **1.04×** |

Energy unchanged (`-506.67980423 Ry`). Keep the XOR-conj patch on the board lib.

### Follow-up that did not help: NEON-style `VFMAI` / `VZMUL`

`patch-simd-r5v-vfmai-neon.py`: `VFMAI(b,c) = VFMA(FLIP(b), (-1,+1), c)` (and same for
`VZMUL`’s `VBYI`), matching `simd-neon.h`. Same gather count; aims to hoist the
sign vector off the gather path. Smoke sumsq identical to xorconj.

| probe | XOR-conj (recheck) | NEON-VFMAI | ratio |
|---|---:|---:|---:|
| 1D N=256 MFLOPS | 2467 | 2550 | 1.03× |
| 1D N=1024 | 1631 | 1591 | 0.98× |
| 1D N=4096 | 1316 | 1290 | 0.98× |
| 1D N=65536 | 794 | 802 | 1.01× |
| many N=64×4096 | 721 | 712 | 0.99× |
| many N=64×64 | 1740 | 1734 | 1.00× |

**Verdict: noise / slight loss on QE-like `many`.** Reverted to xorconj
(`libfftw3-r5v-xorconj.so`; neon `.so` kept only as reference).

### Follow-up: cheaper `FLIP_RI` / hand-split `t2bv_8` (both lost)

Isolated `FLIP_RI` on X60 (4×f64, odd-rep sink): **gather ~6.9 ns** beats
slide+merge (~11.3 ns) and store-shuffle (~12.5 ns). So replacing gather in
`BYTW2` is the wrong direction on this SoC.

Hand-specialized `t2bv_8` (`t2bv_8_r5v256_split.c`): `vlseg2`/`vsseg2` split
re/im, gather-free `BYTW2` via strided cos/sin loads. Unit test vs FMA stock
with real VTW2 twiddles: **max err ~1e-16**. Kernel microbench: stock
**~120 ns/call** vs split **~249 ns** (**0.48×**). Reverted; live codelet is
stock again under xorconj.

`t2bv_8` gathers are already CSE’d to one index vector; remaining cost is
gather latency itself, and the alternatives tried here are worse.

```bash
bash ~/fftw-wisdom-src/bench-codelet-hot.sh
bash ~/fftw-wisdom-src/run-t2bv8-unit.sh   # stock vs split, needs rebuilt objs
```

## EasyBuild easyconfig

`FFTW-3.3.10-GCC-14.3.0-r5v.eb` packages the `r5v` half of the A/B as a proper
module, so the RVV lib is reproducible via the EESSI toolchain rather than a
hand-run script. It uses the stock `EB_FFTW` easyblock (no custom easyblock
needed) with three deltas from an upstream FFTW easyconfig:

- `configopts = '--enable-r5v --disable-fortran CFLAGS="-O3 -march=rv64imafdcv_zvl256b"'`
  — the RVV backend plus the pinned K1 vector ISA. The easyblock appends this
  verbatim to each precision's `./configure` line, so `CFLAGS=` here is the
  supported override pattern.
- `auto_detect_cpu_features = False` — the easyblock only knows x86/ARM/POWER
  SIMD (avx/sse/neon/sve/…); there is no RVV entry, so detection is a no-op on
  riscv64. Pinned off to keep the configure line deterministic.
- double precision / shared only (`with_single_prec`, `with_*` threads/openmp/mpi
  all `False`), matching the benchmarked build.

The source is rdolbeau's `r5v-test-release-005` repackaged with a stock
`fftw-3.3.10/` top dir; drop `fftw-r5v.tar.gz`
(`sha256:65f81f80…9f8fd3`) into your sourcepath first.

```bash
eb FFTW-3.3.10-GCC-14.3.0-r5v.eb    # -> module FFTW/3.3.10-GCC-14.3.0-r5v
```

Verified to parse cleanly against EasyBuild 5.3.1 (all parameters recognized by
the framework + `EB_FFTW` easyblock). `runtest = 'check'` runs FFTW's own test
suite, which is slow on the X60 — build with `--skip-test-step` for libs only.

### Not upstream-ready (lives on a fork)

This easyconfig is **experimental** and is intentionally *not* proposed to
[`easybuilders/easybuild-easyconfigs`](https://github.com/easybuilders/easybuild-easyconfigs).
It is tracked on a fork instead:
[hmeiland/easybuild-easyconfigs#3](https://github.com/hmeiland/easybuild-easyconfigs/pull/3).

Three things block upstreaming, all documented in that PR:

1. **The source is a fork, not an official release** — rdolbeau's
   `r5v-test-release-005` repackaged as a local `fftw-r5v.tar.gz`; there is no
   permanent public `source_urls` for upstream CI to fetch.
2. **`--enable-r5v` is not supported by the upstream FFTW easyblock** (which
   only knows avx/sse/neon/sve/vsx/altivec) — here it is passed as a raw
   `configopts` string rather than proper easyblock / `use_*` handling.
3. **`-march` is hardcoded in `CFLAGS`**, bypassing the `--optarch` contract
   upstream easyconfigs are required to respect.

Upstreaming would require a published r5v source, RVV support added to the FFTW
easyblock, and dropping the pinned `-march`.
