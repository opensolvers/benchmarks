# MetalWalls — FFT / FlexiBLAS A/B (SpaceMiT X60 / Orange Pi RV2)

Constant-potential electrochemical MD
[`MetalWalls/21.06.1-foss-2023b`](https://gitlab.com/ampere2/metalwalls) on
EESSI `riscv.eessi.io` **20240402**. Binary `mw` links FlexiBLAS + FFTW.

## Workload

Upstream **`example/tip4p-water`** (215 TIP4P molecules, no electrodes),
200 MD steps, `mpirun -np 1`. Profile on this system (blas_stock run):

| Bucket | ≈ % wall |
|--------|---------:|
| Coulomb short-range | ~50% |
| van der Waals | ~29% |
| Coulomb **long-range** (Ewald/FFT) | ~11% |
| RATTLE | ~7% |

Stock electrode examples (`thomas-fermi-gold-capacitor` ~3240 electrode atoms
with `matrix_inversion` or `cg`) need many minutes per few steps and were
deferred for a short A/B.

## Results (2026-08-21)

| Tag | Axis | Wall s | `Total elapsed` s |
|-----|------|-------:|------------------:|
| fft_scalar | LD_PRELOAD scalar `libfftw3` | 41.27 | 34.96 |
| fft_r5v | LD_PRELOAD r5v `libfftw3` | 40.59 | 35.13 |
| blas_scalar | `OPENBLAS_CORETYPE=RISCV64_GENERIC` | 40.83 | 35.13 |
| blas_stock | default OpenBLAS via FlexiBLAS | 40.53 | 35.06 |
| blas_patched | `FLEXIBLAS=~/libopenblas_x60_eb_fixed.so` | 40.99 | 35.23 |

**FFT r5v vs scalar ≈ 1.02×** (noise; LR is only ~11% of the run).  
**BLAS backends ≈ 1.00×** (tip4p is not LAPACK-bound).  
Temperatures match bit-for-bit: `TEMP_LAST=200  2.664016202017195E+002`.

Log: `~/logs/metalwalls-ab-20260821-192034.log` on the board.

## Reproduce

```bash
# examples (once):
git clone --depth 1 --filter=blob:none --sparse \
  https://gitlab.com/ampere2/metalwalls.git ~/mw-bench/metalwalls
cd ~/mw-bench/metalwalls && git sparse-checkout set example/tip4p-water

MW_STEPS=200 bash ~/mw-bench/run-metalwalls-ab.sh
```

Needs custom FFTW A/B pair under `~/fftwbuild/src-{scalar,r5v}/` (same as
ESPResSo/ScaFaCoS). Optional `RVV_LIB` for FlexiBLAS backend swap.
