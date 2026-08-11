# RV2 GCC x60 A/B summary

Baseline label: `generic-ooo` (ns/call; lower is better).

| kernel | label | mean ± stdev (ns) | median | Δ% vs baseline | n |
|--------|-------|-------------------|--------|----------------|---|
| `load_add_chain` | `generic-ooo` | 7909.1 ± 517.7 | 7804.6 | +0.00% | 22 |
| `load_add_chain` | `x60` | 7794.6 ± 56.4 | 7756.8 | -1.45% | 22 |
| `load_add_chain` | `default-tune` | 7818.9 ± 120.1 | 7779.1 | -1.14% | 22 |
| `load_add_chain` | `eessi-14.3` | 7809.7 ± 76.0 | 7781.2 | -1.26% | 22 |
| `fma_chain` | `generic-ooo` | 21927.9 ± 523.2 | 21818.0 | +0.00% | 22 |
| `fma_chain` | `x60` | 20822.1 ± 1552.5 | 20510.6 | -5.04% | 22 |
| `fma_chain` | `default-tune` | 21895.1 ± 95.8 | 21908.1 | -0.15% | 22 |
| `fma_chain` | `eessi-14.3` | 21757.7 ± 95.1 | 21732.5 | -0.78% | 22 |
| `div_mix` | `generic-ooo` | 37952.1 ± 48.4 | 37945.6 | +0.00% | 22 |
| `div_mix` | `x60` | 35398.1 ± 54.4 | 35383.2 | -6.73% | 22 |
| `div_mix` | `default-tune` | 37949.1 ± 61.1 | 37935.6 | -0.01% | 22 |
| `div_mix` | `eessi-14.3` | 38704.2 ± 3482.4 | 37964.7 | +1.98% | 22 |
| `sh1add` | `generic-ooo` | 10606.7 ± 35.2 | 10610.3 | +0.00% | 22 |
| `sh1add` | `x60` | 10585.3 ± 176.2 | 10559.6 | -0.20% | 22 |
| `sh1add` | `default-tune` | 10630.7 ± 186.3 | 10605.0 | +0.23% | 22 |
| `sh1add` | `eessi-14.3` | 10619.5 ± 58.9 | 10626.8 | +0.12% | 22 |

## Tuned vs baseline

| kernel | generic-ooo mean | x60 mean | Δ% |
|--------|---------------|----------|----|
| `load_add_chain` | 7909.1 | 7794.6 | -1.45% |
| `fma_chain` | 21927.9 | 20822.1 | -5.04% |
| `div_mix` | 37952.1 | 35398.1 | -6.73% |
| `sh1add` | 10606.7 | 10585.3 | -0.20% |
