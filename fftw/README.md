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
SCF** with these drop-in vectorized libraries — the win requires honest FFT
planning (unavailable to QE) and BLAS kernels the generic RVV OpenBLAS does not
yet provide. A microbenchmark speedup is not an application speedup.

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
