# RV2 GCC 15.2 x60 A/B — clean series revalidation

Re-run after regenerating Layer B as a pristine `patch(1)` series
(clmul-only 0005; `type=shadd` deferred). Same install binary as the
finish pass (`gcc-15.2.0-x60`).

Stamp: 2026-08-10T13:05Z. Governor `performance` @ 1.60 GHz, `taskset -c 0`.

## x60 vs `generic-ooo` (mean ns/call; lower better)

| Kernel | Δ% (clean) | Δ% (finish earlier) |
|--------|------------|---------------------|
| `load_add_chain` | **−1.4%** | +2.9% (was noisy) |
| `fma_chain` | **−8.7%** | −5.9% |
| `div_mix` | **−7.7%** | −6.5% |
| `sh1add` | **−1.0%** | +0.8% |

All four kernels favor x60 in this run. See `summary.md`.

## Related

- Patch apply proof: [`../easybuild-unified-verify.log`](../easybuild-unified-verify.log)
- Full campaign tree: upstream `spacemit-x60-gcc-tune` (`measurements/results/`)
