# ESPResSo — P3M FFT backend A/B (soft-matter MD)

Whole-app FFT probe through [ESPResSo](https://espressomd.org/) 4.2.2 P3M
(Particle–Particle Particle–Mesh) electrostatics. Companion to
[`../gromacs`](../gromacs) (PME/`libfftw3f`) and [`../fftw`](../fftw): here the
swapped axis is **double-precision** `libfftw3.so.3` under an unchanged
`pypresso` binary via `LD_PRELOAD`.

ESPResSo itself is not vendored — load `ESPResSo/4.2.2-foss-2025b` from
`dev.eessi.io/riscv`.

## Contents

| File | Purpose |
|---|---|
| `p3m_lj.py` | charged WCA/LJ fluid on a cubic lattice + P3M |
| `run-espresso-fft-ab.sh` | scalar vs r5v (RVV) `libfftw3` A/B |
| `hotpath_models.py` | multi-model Coulomb-share survey (sample-inspired) |
| `run-hotpath-survey.sh` | one `pypresso` per model (ESPResSo: one `System`) |
| `mesh_fft_sweep.py` / `run-mesh-sweep.sh` | fixed cao/r_cut/alpha, sweep mesh |
| `run-espresso-mpi-scale.sh` | MPI strong scaling (`ESP_CASE=lattice512\|dense_large`) |
| `mpi_dense_large.py` | denser P3M case for MPI scaling |
| `run-espresso-perf.sh` | `perf cpu-clock` hot-function profile |
| `build-espresso-opt.sh` | slim myconfig + patches → `~/espresso-opt` |
| `run-espresso-opt-ab.sh` | EESSI vs local opt A/B (`ESP_CASE`, `ESP_REPS`) |
| `run-espresso-opt-v2.sh` | skin sweep + multi-rep A/B |
| `run-espresso-opt-v3.sh` | cao retune + FFTW RVV + k-space patch bench |
| `p3m_cao_tune.py` | auto-tune one fixed cao (one `System` per run) |
| `patches/` | ForceKernelRef, P3M monomorph, SR table, k-space loop |
| `myconfig-slim.hpp` | minimal feature set (WCA + P3M only) |

Env knobs:

| Var | Default | Role |
|---|---|---|
| `ESP_N` | 512 | particles |
| `ESP_STEPS` | 200 | timed MD steps |
| `ESP_BOX` | 20 | box length |
| `ESP_ACCURACY` | `1e-3` | P3M accuracy goal (used when tuning) |
| `ESP_MESH` / `ESP_CAO` / `ESP_RCUT` / `ESP_ALPHA` | unset | pin P3M params (fair A/B) |
| `ESP_TUNE` | `1` | set `0` with full pins to skip auto-tune |

When comparing FFT backends, **pin the same mesh/cao/r_cut/alpha on both
sides**. Auto-tune can pick different meshes if one FFTW is faster during
timing probes, which invalidates the A/B.

## Run (Orange Pi RV2)

```bash
# need double-prec FFTW A/B pair (same builds used by fftw/):
#   ~/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10
#   ~/fftwbuild/libfftw3-r5v-xorconj.so   # preferred; falls back to src-r5v
module load ESPResSo/4.2.2-foss-2025b

# auto-tuned (quick smoke):
ESP_N=512 ESP_STEPS=200 ESP_BOX=20 bash run-espresso-fft-ab.sh

# fair pinned A/B (mesh 18 from a prior tune):
ESP_N=512 ESP_STEPS=1000 ESP_BOX=20 ESP_ACCURACY=1e-3 \
  ESP_MESH=18 ESP_CAO=5 ESP_RCUT=4.4896 ESP_ALPHA=0.53745 ESP_TUNE=0 \
  bash run-espresso-fft-ab.sh
```

Correctness gate: `E_coul` (and `E_tot`) must agree across backends.

## Results — Orange Pi RV2 (2026-08-30)

`ESPResSo/4.2.2-foss-2025b`, 1 thread, `OPENBLAS_CORETYPE=RISCV64_GENERIC`,
r5v = `libfftw3-r5v-xorconj.so`. Energies bit-match on every fair pair.

### Fair pinned (same P3M params both sides)

| case | mesh | steps | scalar wall | r5v wall | speedup |
|---|--:|--:|--:|--:|--:|
| N=512, box=20, acc=1e−3 | 18³ | 1000 | 13.226 s | 13.259 s | **0.998×** |
| N=512, box=20, acc=1e−5 | 24³ | 1000 | 34.719 s | 35.562 s | **0.976×** |
| N=4096, box=40, acc=1e−4 | 32³ | 300 | 57.282 s | 57.410 s | **0.998×** |
| N=4096, box=40, acc=1e−4 | 44³ | 200 | 37.154 s | 36.842 s | **1.008×** |

Logs: `~/logs/espresso-fft-ab-pin{18,24}-s1000.log`,
`~/logs/espresso-fft-ab-pin{32,44}-n4096-*.log`.

### Auto-tuned short runs (same day)

| case | mesh (both) | steps | scalar | r5v | speedup |
|---|--:|--:|--:|--:|--:|
| N=512, acc=1e−3 | 18³ | 200 | 2.568 s | 2.528 s | **1.016×** |
| N=512, acc=1e−5 | 24³ | 200 | 6.596 s | 6.354 s | **1.038×** |

### Takeaway

On this P3M workload, end-to-end RVV FFTW is **noise / ~0–4%**. Mesh sizes stay
tiny (18³–44³), so wall time is dominated by real-space LJ/Coulomb, thermostat,
and P3M spread/gather — not the FFT. That matches the diluted QE story more than
GROMACS PME (~1.14–1.23×), where the FFT share is larger.

An earlier short run (2026-08-21, N=512 / 200 steps / mesh 18³) reported
**~1.12×**; that did not reproduce under longer fair pins with the current
xorconj build. Prefer the pinned table above.

## Hot-path survey — models (2026-08-30)

Inspired by upstream `samples/` (`p3m.py`, `minimal-charged-particles.py`,
`grand_canonical.py`, `electrophoresis.py`) plus our lattice harness.
Method: time MD **before** adding P3M, then again with tuned P3M
(`ESP_HOT_STEPS=150`). Log: `~/logs/espresso-hotpath-survey3.log`.

| model | N | box | mesh | t_full | Coulomb share |
|---|--:|--:|--:|--:|--:|
| `lattice512` (our A/B) | 512 | 20 | 18³ | 2.23 s | **87.5%** |
| `sample_p3m` | 300 | 10 | 16³ | 1.63 s | **88.3%** |
| `dense_wca` | 868 | 10.7 | 18³ | 3.37 s | **72.1%** |
| `dense_large` | ~5.5k | 24 | 32³ | 36.5 s | **86.4%** |
| `salt_box50` (dilute) | 250 | 50 | 12³ | 0.68 s | ~0% (P3M cheap) |
| `electrophoresis` (dilute) | 140 | 100 | 16³ | 1.16 s | ~0% (P3M cheap) |

Dense charged fluids: **Coulomb/P3M is the hot path (~70–90%)**. Dilute / large-box
samples are not FFT probes — P3M is a small add-on.

### Mesh sweep (FFT sensitivity) — lattice N=512

Pin cao=5, r_cut=4.4896, alpha=0.53745; vary mesh only (`ESP_HOT_STEPS=400`).
Log: `~/logs/espresso-mesh-sweep.log`.

| mesh | nocoul | full | coulomb extra | coulomb share |
|--:|--:|--:|--:|--:|
| 12³ | 0.84 s | 3.71 s | 2.87 s | 77% |
| 18³ | 0.85 s | 5.30 s | 4.45 s | 84% |
| 24³ | 0.83 s | 9.03 s | 8.20 s | 91% |
| 32³ | 0.85 s | 19.7 s | 18.8 s | 96% |
| 48³ | 0.84 s | 50.0 s | 49.2 s | 98% |
| 64³ | 0.84 s | 155 s | 155 s | 99.5% |

Wall grows much faster than particle work as mesh increases (k-space / mesh loops /
FFT). At the **auto-tuned** mesh (~18³) that extra is still modest versus
spread + real-space.

### Artificial FFT-heavy A/B (mesh 48³)

Same pins as sweep, 400 steps — energies match; still no RVV win:

| backend | wall | speedup |
|---|--:|--:|
| scalar | 49.515 s | — |
| r5v xorconj | 50.324 s | **0.98×** |

So even when mesh-dependent work dominates the clock, swapping `libfftw3` does
not help: ESPResSo’s own mesh/spread/k-space work (and/or tiny 3D FFT sizes)
dwarfs any RVV FFTW gain. **Further FFTW tuning will not move Espresso P3M.**

## MPI strong scaling (2026-08-30) — priority “use the cores”

Pinned P3M, `OMP_NUM_THREADS=1`, scalar `libfftw3`, OpenMPI 5 on 8× X60.
Runner: `run-espresso-mpi-scale.sh` (`ESP_CASE=lattice512|dense_large`).

| np | lattice512 wall (400 steps) | speedup | dense_large wall (200 steps) | speedup |
|--:|--:|--:|--:|--:|
| 1 | 5.119 s | 1.00× | 46.001 s | 1.00× |
| 2 | 3.674 s | **1.39×** | 26.237 s | **1.75×** |
| 4 | 2.878 s | **1.78×** | 23.574 s | **1.95×** |
| 8 | 2.600 s | **1.97×** | 16.289 s | **2.82×** |

Logs: `~/logs/espresso-mpi-scale-lattice512.log`,
`~/logs/espresso-mpi-scale-dense_large.log`.

Energies match across `np`. Small N=512 saturates early (~2× at 8 ranks);
**dense_large (~5.5k particles) reaches ~2.8× at 8 ranks** — real Coulomb win,
still far from ideal 8× (P3M FFT/comms + small mesh).

## `perf` hot functions (2026-08-30)

Kernel has no matching `linux-tools`; used
`/usr/lib/linux-riscv-6.17-tools-6.17.0-38/perf record -e cpu-clock -g`
(`run-espresso-perf.sh`). Samples cover whole run (setup + MD).

### dense_large (N≈5528, mesh 32³, 250 steps) — primary Coulomb case

| DSO | overhead |
|---|--:|
| `Espresso_core.so` | **75%** |
| `Espresso_script_interface.so` | 8% |
| `libm` | 6% |
| **`libfftw3`** | **4.4%** |
| `libc` | 4% |

Top symbols (flat):

| ~% | symbol | role |
|--:|---|---|
| 22 | `add_non_bonded_pair_force` | pair loop (WCA + Coulomb short-range callbacks) |
| 14 | `force_calc` | force orchestration |
| 10 | `CoulombP3M::long_range_kernel` | k-space / mesh P3M (Espresso) |
| 9 | `ShortRangeForceKernel` → P3M | real-space Ewald (`erfc`/`exp`) |
| 8 | `BoxGeometry::get_mi_vector` | minimum-image |
| 4.5 | `libm __exp_finite` | short-range Coulomb math |
| 3.7 | `ghost_communicator` | ghosts |
| 2.9 | `CoulombP3M::charge_assign` | charge spreading |
| ~4 | `pack_block_permute*` | Espresso FFT transpose/pack |
| ~3 | FFTW `n1_8` / `t2_4` / … | actual `libfftw3` kernels |

### lattice512 (mesh 18³) — our A/B case

| DSO | overhead |
|---|--:|
| `Espresso_core.so` | **56%** |
| **`libfftw3`** | **12%** |
| `libc` / Python | ~20% |

`long_range_kernel` ~11%, `charge_assign` ~5%, short-range P3M kernel ~5%.
FFTW share is higher than dense_large only because the pair loop is smaller — still
a minority of wall time, matching ~0% RVV FFTW A/B.

**Where to improve Coulomb:** Espresso short-range pair + P3M real-space
(`ShortRangeForceKernel` / `__exp_finite`), then `long_range_kernel` /
`charge_assign` / pack-permute — **not** FFTW. Logs:
`~/logs/espresso-perf-{dense_large,lattice512}.{log,data}`.

## Optimized local build vs EESSI (2026-08-30)

Built ESPResSo 4.2.2 from source on RV2 with slim `myconfig` (7 features vs
EESSI’s 34) and opensolvers patches under `patches/`. Install:
`~/espresso-opt`; runner: `build-espresso-opt.sh`.

### Patches (cumulative)

| step | change | target |
|---|---|---|
| v1 | `ForceKernelRef` replaces `std::function` in Coulomb SR | pair-loop indirection |
| v1 | slim myconfig — WCA + P3M only | drop NPT/extra potentials from hot loop |
| v2 | `add_non_bonded_pair_force_p3m` — direct `CoulombP3M::pair_force` | monomorph pair loop |
| v2 | 4096-bin SR Ewald force table | kill per-pair `exp` |
| v2 | `ESP_SKIN` sweep — **0.4** best for dense fluid | neighbor list |
| v3 | k-space differentiation loop — pointer locals | `long_range_kernel` |

### A/B vs EESSI (`dense_large`, N≈5528, mesh 32³, 200 steps, 3 reps)

| build | mean wall | vs EESSI |
|---|---:|---:|
| EESSI 4.2.2 (full myconfig) | 46.8 s | 1.0× |
| opt v1 (ForceKernelRef + slim) | 31.8 s | **1.44×** |
| opt v2 (+ monomorph + SR table) | 28.1 s | **1.66×** |
| opt v3 (+ k-space patch, cao=6) | **26.8 s** | **1.74×** |

Energies match (`E_tot` / `E_coul` to ~1e−4). `lattice512`: EESSI ~8.0 s → opt v2
~**6.0 s** (**1.35×**).

Logs: `~/logs/espresso-opt-ab-multirep.log`, `espresso-opt-v2-ab.log`,
`espresso-opt-v3.log`.

### Follow-ups tried (#4–6) — no further FFTW lever

| lever | result |
|---|---|
| **#4** lower cao (4/5) + retune | **slower** — tuner raises `r_cut` to hold accuracy; pair work dominates |
| **#5** k-space loop patch | ~**3%** on top of v2 |
| **#6** RVV `libfftw3` on opt build | **~0%** (scalar **27.5 s** vs r5v **28.3 s** mean) |

`perf` on opt build: `add_non_bonded_pair_force` / `std::function` gone; hot path
is `force_calc` + P3M SR + `long_range_kernel`; FFTW still ~4%.

### MPI + production path

Same pinned P3M: **~2.8× at 8 ranks** on `dense_large` (see MPI table above).
Best wall-clock win for Coulomb-heavy runs remains **MPI + opt binary**, not more
FFTW RVV.

## Prior note (2026-08-21)

| backend | wall (integrator) | E_coul | E_tot |
|---|--:|--:|--:|
| scalar `libfftw3` | 2.706 s | −98.404639 | 632.452568 |
| r5v (RVV) `libfftw3` | 2.425 s | −98.404669 | 632.452480 |
| **speedup** | **1.12×** | Δ ~3e−5 | Δ ~9e−5 |

Log: `~/logs/espresso-fft-ab-20260821-140009.log`.
