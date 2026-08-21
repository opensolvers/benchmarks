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
| `p3m_lj.py` | charged WCA/LJ fluid on a cubic lattice + auto-tuned P3M |
| `run-espresso-fft-ab.sh` | scalar vs r5v (RVV) `libfftw3` A/B |

Env knobs: `ESP_N` (particles, default 512), `ESP_STEPS` (timed steps, default
200), `ESP_BOX` (box length, default 20).

## Run (Orange Pi RV2)

```bash
# need double-prec FFTW A/B pair (same builds used by fftw/):
#   ~/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10
#   ~/fftwbuild/src-r5v/.libs/libfftw3.so.3.6.10
module load ESPResSo/4.2.2-foss-2025b
ESP_N=512 ESP_STEPS=200 ESP_BOX=20 bash run-espresso-fft-ab.sh
```

Correctness gate: `E_coul` (and `E_tot`) must agree across backends.

## Results — Orange Pi RV2 (2026-08-21)

`ESPResSo/4.2.2-foss-2025b`, N=512, box=20, 200 timed steps, 1 thread,
`OPENBLAS_CORETYPE=RISCV64_GENERIC`, log
`~/logs/espresso-fft-ab-20260821-140009.log`:

| backend | wall (integrator) | E_coul | E_tot |
|---|--:|--:|--:|
| scalar `libfftw3` | 2.706 s | −98.404639 | 632.452568 |
| r5v (RVV) `libfftw3` | 2.425 s | −98.404669 | 632.452480 |
| **speedup** | **1.12×** | Δ ~3e−5 | Δ ~9e−5 |

Energies match; RVV FFTW wins a modest **~1.12×** on the timed MD segment —
same ballpark as GROMACS PME 3D-FFT (~1.14–1.23×), diluted by non-FFT work
(real-space LJ/Coulomb, thermostat, P3M spread/gather).
