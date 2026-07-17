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
