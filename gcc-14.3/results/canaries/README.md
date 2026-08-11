# RV2 GCC 14.3-x60 schedule A/B

Host: Orange Pi RV2, performance @ 1.60 GHz, `taskset -c 0`. Date: 2026-08-11.

| | |
|--|--|
| Compiler | `/home/orangepi/gcc-tune/install/gcc-14.3.0-x60/bin/gcc` (`gcc (GCC) 14.3.0`) |
| Flags | `-O3 -march=rv64gcv_zba_zbb_zbc_zvl256b -mabi=lp64d` |
| Primary A/B | same binary: `-mtune=generic-ooo` vs `-mtune=spacemit-x60` |
| Also | `default-tune` (omit `-mtune`); stock EESSI `GCC/14.3.0` march-only |

## x60 vs `generic-ooo` (mean ns/call; lower better)

| kernel | generic-ooo | x60 | Δ% | 15.2 clean Δ% |
|--------|-------------|-----|----|---------------|
| `load_add_chain` | 7909.1 | 7794.6 | **−1.45%** | −1.36% |
| `fma_chain` | 21927.9 | 20822.1 | **−5.04%** | −8.67% |
| `div_mix` | 37952.1 | 35398.1 | **−6.73%** | −7.67% |
| `sh1add` | 10606.7 | 10585.3 | **−0.20%** | −0.95% |

n=22 per cell (2 passes × 11 reps). Full table: `summary.md` / `summary.csv`.

Directionally matches the 15.2 clean series; `fma_chain` is a bit weaker /
noisier on 14.3. `sh1add` near flat — expected with `type=shadd` deferred.

**ELF `bin/` omitted** from this archive (rebuildable). Schedule canaries only;
OpenBLAS/HPL not re-run on 14.3.

## Related

- Patch apply proof: [`../easybuild-unified-14.3-verify.log`](../easybuild-unified-14.3-verify.log)
- Full campaign tree: upstream `spacemit-x60-gcc-tune` (`measurements/results/`)
