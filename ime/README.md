# ime-bench — reusable int8 (s8s8s32) GEMM core for the SpaceMiT X60 IME

A standalone `int8 × int8 → int32` matrix-multiply microkernel built on the X60
IME instruction `smt.vmadot`, plus a harness that checks it **bit-exactly**
against a scalar reference and times it against a plain **RVV** int8 baseline.

This is the shared core to lift into a framework backend (MLAS / XNNPACK / ruy):
the packing here produces the same `B`-is-`N×K` (pre-transposed) tile layout that
`mmt4d` / ggml-repack feed to the matrix unit.

## What it computes

`C = A · Bᵀ`, i.e. `C[i][j] = Σ_k A[i][k]·B[j][k]`, with `A` = `M×K`, `B` = `N×K`
(both row-major int8) and `C` = `M×N` int32.

## The IME tile

One `smt.vmadot vd, vs1, vs2` does, at VLEN=256 with `vl=32, e8`:

```
C(4×4 int32) += A(4×8 int8) · B(4×8 int8)ᵀ      # M0=4, N0=4, K0=8
```

`vs1`/`vs2` hold row-major 4×8 int8 tiles; `vd` is an even `vd:vd+1` pair holding
the 4×4 int32 accumulator (EMUL=2). We pack A/B into contiguous 32-byte tiles,
accumulate across K, then scatter the result to `C`.

### Register blocking (the main kernel)

One `vmadot` per pair of loads is latency-bound — the store of each 4×4 tile
waits on its accumulate. The main path `ime_block_8x16` instead holds an **8×16
output** in 8 accumulator pairs (`v16..v31`): each K-step loads 2 A-tiles + 4
B-tiles and issues **8 `vmadot`s**, so every load feeds several MACs and the 8
independent accumulators hide the unit's latency. The 8 result tiles are written
**straight to `C`** (vectorized, no scalar scratch copy). The plain 4×4
`ime_tile` remains for the M/N block edges. An L2 panel loop keeps the reused
B-panel (~128 KB) resident in the 512 KB cluster L2 across the M sweep.
**Amortized packing** matches that loop: `pack_b_panel` runs once per N-panel
(before the M sweep); `pack_a_panel` runs once per M-tile pair on the first
N-panel only (`n0==0`). Same total pack work as a monolithic `pack_a`/`pack_b`,
but B bytes are gathered just before the panel that reuses them (modest gain,
~parity on RV2 at 768³).

| File | Role | Portable? |
|---|---|---|
| `gemm_ref.c` | scalar reference + **shared packing** + packed scalar path | yes |
| `gemm_ime.c` | `smt.vmadot` kernels: register-blocked 8×16 + 4×4 edge | X60 only |
| `gemm_rvv.c` | RVV int8 baseline (`vwmul`+widening reduce) | X60 only |
| `bench.c` | fill / cross-check / GOP-s timing (+ peak mode) | yes |
| `bench_gap.c` | insn peak → kloop → pack → full GEMM waterfall | X60 only |

Because the packing and tile-index math live in the **portable** `gemm_ref.c`
(`gemm_packed_ref` uses the exact same layout as `gemm_ime`), everything except
the two asm inner loops is verifiable off-target.

## Build & run

The IME kernel emits `smt.vmadot` as a **raw instruction word** (`.insn 4,
0xe200302b|…`, letting the assembler compute the register fields), so it needs no
`xsmtvdot`-aware toolchain — any RVV-capable binutils builds it. That is what
lets `make board` work on the RV2's stock **binutils 2.42**.

```bash
# Portable self-test (x86, or any host): verifies packing + harness, no RISC-V.
make host && ./ime-bench-host 256 256 128

# X60 build: scalar + RVV baseline + IME kernels. Pin to an IME-capable core.
make board && taskset -c 0 ./ime-bench 512 512 512

# X60 peak mode: 4th arg = rep count. The scalar oracle runs once for the
# bit-exact check, then N back-to-back IME reps report min/median/max GOP/s.
taskset -c 0 ./ime-bench 768 768 512 50

# Isolated vmadot peak (cpufp-style, L1-resident micro-tile):
make board-peak && taskset -c 0 ./ime-bench-peak 64 500000 0

# GEMM with anti-alias padding (ldc=N+16, buffer offset GEMM_BUF_PAD):
taskset -c 0 ./ime-bench 768 768 512 25 1

# Full peak + aliasing A/B sweep:
bash run-ime-peak-alias.sh

# Theoretical vs measured GOP/s waterfall (pack / compute / full GEMM):
make board-gap && taskset -c 0 ./ime-bench-gap 768 768 512 25 1
bash run-ime-gap.sh

# Multi-core synthetic IME (OpenMP M-split):
make board-mc && bash run-ime-mc.sh

# X60, RVV baseline only (skips the IME kernel entirely):
make board-rvv && taskset -c 0 ./ime-bench
```

Multi-core harness: [`bench_mc.c`](bench_mc.c) + [`run-ime-mc.sh`](run-ime-mc.sh)
(`make board-mc` → `ime-bench-mc`). Splits `M` across OpenMP threads on cluster 0;
each thread needs its own pack buffers (`Ap`/`Bp`).

Dims must satisfy `M%4=0, N%4=0, K%8=0`. On the K1/M1 the IME lives on one 4-core
cluster — pin to a core in it (`taskset -c 0`; cluster 0 = cores 0-3).

## Results (measured — Orange Pi RV2 / X60, single core 0, 1.6 GHz, performance gov)

Built with `make board` (stock binutils 2.42, raw `.insn`). Every path is `ok`,
bit-exact vs the scalar reference at every size tested.

Throughput is **single-core GOP/s** (`2·M·N·K / t`). The board is multi-tenant
and cores 0-3 share one 512 KB L2, so a neighbouring core polluting L2 mid-run
adds noise. The figures below are the **clean-layout peak** — the max over ≥5
process launches, each internally 15-50 reps (contention only ever lowers a run,
so the max is the true single-core capability). See the aliasing note for the
spread.

| M×N×K | scalar | RVV int8 | **IME (8×16 blocked)** | IME/RVV | prev 4×4-only |
|---|--:|--:|--:|--:|--:|
| 256×256×256   | 0.39 | 5.05 | **33** | 6.5× | 22.0 |
| 512×512×512   | 0.39 | 5.22 | **39** | 7.5× | 25.8 |
| 768×768×512   | 0.39 | ~5.2 | **42** | 8.1× | —    |
| 1024×1024×512 | 0.39 | 5.19 | **32** | 6.2× | 26.7 |
| 2048×2048×512 | 0.39 | ~5.2 | **34** | 6.5× | —    |

GOP/s. Register blocking (8 accumulators, straight-to-`C` stores) lifts the IME
kernel **+20–50 %** over the earlier single-4×4-tile path and pushes the
IME-vs-RVV ratio from ~5× to **~7–8×** — peak **42 GOP/s at 768³**, ~108× the
scalar reference. The clean peak crests at 768³ (~42) then settles to ~32–34 at
1024³–2048³ as the working set outgrows the 512 KB L2 (at 1024³ the `C` matrix
alone is 4 MB): the large sizes are turning memory-bound, the main tuning lead
from here.

## RV2 re-verify (2026-08-22)

Quick sanity run via [`run-part-a-v2.sh`](../run-part-a-v2.sh) (`taskset -c 0`,
512×512×512, single rep):

| path | GOP/s | check |
|---|--:|---|
| scalar ref | 0.39 | — |
| RVV int8 | 5.23 | ok |
| IME (8×16 blocked) | **24.5** | ok (bit-exact) |

IME/RVV **4.7×** at this size (below the clean-layout peak table above because
this was a single launch, not max-over-reps). Log:
`~/logs/part-a-v2-20260822-091805.log`.

## i8i8 kernel selftest + A/B (2026-08-22, RV2)

[`bench_i8i8.cpp`](bench_i8i8.cpp) links `gemm_kernel_i8i8` from the
`-x60-ime-q8_0` llama.cpp build. Runner: [`run-i8i8-selftest-ab.sh`](run-i8i8-selftest-ab.sh);
log: `~/logs/i8i8-selftest-20260822-205847.log`.

**Correctness gate** (`chk=1`, varied finite inputs): `finite=1` at 512³.

**Kernel throughput** (varied fill, cluster 0, median of 25 reps):

| M×N×K | i8i8 GOP/s | ime s8s8s32 | i8i4 (ref) |
|---|--:|--:|--:|
| 512×512×512 | **23.4** | 33.3 | ~22 |
| 768×768×512 | **28.1** | 24.7 | — |
| 1024×1024×512 | **22.0** | 30.3 | ~32 |

End-to-end Q8_0 `llama-bench` (repacked B): see [`../llamacpp/README.md`](../llamacpp/README.md).

## Multi-core synthetic IME (2026-08-23, RV2)

Raw `s8s8s32` throughput on cluster 0 (cores 0–3, shared 512 KB L2 + IME unit).
Runner: [`run-ime-mc.sh`](run-ime-mc.sh); log: `~/logs/ime-mc-20260823-035745.log`.
All paths `check=ok` (bit-exact vs scalar ref).

**OpenMP M-split** (one GEMM, `B` shared, per-thread `Ap`/`Bp`):

| M×N×K | 1c wall GOP/s | 4c wall GOP/s | 4c/1c | 4c sum-panel |
|---|--:|--:|--:|--:|
| 512×512×512 | 24.6 | 67.6 | **2.7×** | 76.0 |
| 768×768×512 | 25.3 | 79.9 | **3.2×** | 85.2 |
| 1024×1024×512 | 30.2 | 74.6 | **2.5×** | 78.4 |

Per-panel rates on 4c sit at ~19–22 GOP/s (vs ~25–30 on 1c full-`M`), so scaling
is **~2.5–3.2×**, not 4× — the four cores share one IME matrix unit and one L2.
OpenMP M-split still beats launching four independent full GEMMs: **4× independent
768³** (same size, 25 reps each, wall aggregate) ≈ **27 GOP/s** total with each
core ~25 GOP/s median — heavy contention when every process packs its own full `B`.

**Takeaway:** for raw int8 GEMM on this part, one 4-thread OpenMP job on cluster 0
is the right shape (~80 GOP/s at 768³); four separate 1c jobs fight over the same
IME/L2 and sum to ~27 GOP/s aggregate.

## Isolated vmadot peak + aliasing fix (2026-08-23, RV2)

Runner: [`run-ime-peak-alias.sh`](run-ime-peak-alias.sh); log:
`~/logs/ime-peak-alias-20260823-042346.log`.

### (a) cpufp-style silicon ceiling

[`bench_peak.c`](bench_peak.c) times only the `ime_kloop_8x16` inner loop (loads +
8× `vmadot` per K-tile) on one L1-resident 8×16×K micro-tile — no full-GEMM
pack, no panel loops:

| K | variant | insn-equiv GOP/s | cycles/vmadot | vs 409.6 @1.6GHz |
|---:|---|--:|--:|--:|
| 256 | seq (old) | 148 | 2.77 | 36.1 % |
| 256 | **piped (prod)** | **215** | **1.91** | **52.5 %** |
| 256 | block+store | 167 | 2.46 | 40.7 % |

Software-pipelined dual-buffer K-loop (`v8–v13` compute while prefetching
`v0–v5`) is the production path. Naive load/vmadot interleave without prefetch
**regresses** — only the cross-K pipeline wins.

Full GEMM with anti-alias after piped kernel: **~38 GOP/s** @768³ (was ~36).

### Port into llama.cpp i8i8 (2026-08-23)

Tried carrying the synthetic wins into `gemm_kernel_i8i8` (`-x60-ime-q8_0`):

| Change | Status | Result |
|---|---|---|
| Load/`smt.vmadot` INNER interleave | Ported ([`run-llama-pipe-build.sh`](run-llama-pipe-build.sh)) | Kernel **+4–5 %**; e2e pp512 **~0 %** (noise) |
| Dual-bank pipe (ime-bench style) | **Blocked** — FP acc uses `v24..v31`; only 6 free regs | — |
| Amortized packing | N/A — B is offline `repack` | — |
| Anti-alias `ldc` pad | Not ported — ggml owns C layout; short-K tax already low | — |

Why the synthetic **1.45×** kloop win vanishes: llama’s INNER is `BlkLen/16 = 2`
(Q8_0/Q4_0), not dozens of K-tiles; most time is scale epilogue + framework.
Bit-exact vs stock (`bench_i8i8` sum/sumsq/max). Quiet interleaved `llama-bench`
pp512 @ t4 on `qwen2.5-0.5b-q8_0`: stock 75.7 / 76.1 / 80.2 vs pipe 77.1 / 80.8 / 77.4.

### Hybrid prefill (IME) + decode (RVV) in one binary (2026-08-23)

[`apply-hybrid.py`](apply-hybrid.py) + [`run-llama-hybrid.sh`](run-llama-hybrid.sh):
spacemit buffers hold **native GGUF weights + IME tiles**. At runtime
(`SPACEMIT_HYBRID=1`, `SPACEMIT_IME_MIN_M=4`):

- `gemm_m >= 4` → IME M4 (prefill)
- `gemm_m < 4` → stock `ggml_compute_forward_mul_mat` on native weights (RVV decode)

Use **`LD_PRELOAD`** on rebuilt `libggml-cpu.so` (q8_0 RPATH ignores `LD_LIBRARY_PATH`).

| build | Q8_0 pp512 @ t4 | Q8_0 tg32 @ t4 |
|---|--:|--:|
| IME-only | 83.7 | 0.83 |
| **Hybrid** | **90.1** | **6.68** |
| RVV-only | 27.0 | 5.11 |

Q4_0 on `~/x60-ime` already has decent tg (~7.3); hybrid is mainly for **Q8_0 decode**.
Cost: **~2× weight RAM** in spacemit buffers.

### Wider micro-tiles (SpacemiT docs + RV2 trial, 2026-08-23)

Per [SpacemiT IME docs](https://github.com/spacemit-com/docs-ai/blob/main/en/architecture/ime_extension.md):

| Platform | int8 sub-extension tile (`M×K×N`) | Notes |
|---|---|---|
| **A60 / X60** | **4×8×4** | atomic insn remains 4×4×8 |
| **A100** | **8×16×8** | wider int8 data tile |
| **A100** | 8×32×8 | **int4 only**, not int8 |

Our **8×16** micro-block is already 2× the A60 sub-extension width in N. A true
**8×32 int8** block needs **16 acc register pairs** (v16–v47) — impossible on the
32-vreg file while also holding operand tiles (`vd` must not overlap `vs1`/`vs2`).

Trials on RV2 @ K=256 (single core, L1):

| Micro-block | insn-equiv / GEMM | c/vmadot | vs 8×16 piped |
|---|---:|---:|---|
| **8×16 piped** (production) | **214 GOPS** | **1.91** | — |
| 8×32 fused (2× 8×16) | 165 GOPS | 2.47 | −23 % |
| 4×32 kloop (1 A, 8 B) | 64 GOPS | 6.39 | −70 % |

**Conclusion:** on X60, **8×16 piped is the correct ceiling shape**; wider N tiles
add loads without fitting more accumulators. The path to ~1 c/vmadot remains
hardware-limited, not tile-width limited.

### (b) L2 set-aliasing fix (anti-alias allocation)

`bench.c` 6th arg `anti_alias=1`: `ldc = N + GEMM_C_PAD` (16 int32 cols) and
`GEMM_BUF_PAD` (2048 B) offset on `A`/`B`/`Ap`/`Bp`/`C` bases. Same kernel,
bit-exact check `ok`:

| M×N×K | unpadded median | anti-alias median | gain |
|---|--:|--:|--:|
| 512×512×512 | 16.9 | **34.2** | **2.0×** |
| 768×768×512 | 14.4 | **35.7** | **2.5×** |

This run landed on the **bad** side of the malloc lottery without padding; padding
recovers the **good** peak tier (~35 GOP/s) reliably. Production backends should
pad `ldc` and operand buffer bases the same way.

## Theoretical vs measured GOP/s gap (2026-09-01, RV2)

Open investigation — not a correctness blocker, but explains why **~42 GOP/s full
GEMM** sits far below SpacemiT’s **~511 GOPS/core** `vmadot` cpufp figure and our
own **~228 GOPS** isolated kloop microbench.

Runner: [`run-ime-gap.sh`](run-ime-gap.sh) → `ime-bench-gap`; log:
`~/logs/ime-gap-20260901-212728.log`.

### Ceilings

| Baseline | Value | Source |
|---|---:|---|
| **409.6 GOP/s** | 1.6 GHz × 256 ops/`vmadot` × 1 issue/cycle | Our RV2 runs (`performance` gov, pinned 1.6 GHz) |
| **511.5 GOP/s** | cpufp `vmadot(s32,s8,s8)` on K1 single core | [SpacemiT docs](https://github.com/spacemit-com/docs-ai/blob/main/en/architecture/concept.md), [pigirons/cpufp K1](https://github.com/pigirons/cpufp/blob/master/benchmark_result/riscv64/SpacemiT_K1.md) |
| **2.046 TOPS** | 4 cores cluster 0 | Same sources — **~511/core** when all four issue; not 4× independent units |

The **511** number is an **instruction-issue peak** (cpufp-style, likely near **2 GHz**
turbo: 2.0 × 256 ≈ 512). Our board tables use **1.6 GHz** → **409.6** is the fair
silicon comparison for measured runs here.

### Waterfall @ 768³×512, anti-alias, core 0 (good malloc tier)

| Layer | GOP/s | % of 409.6 | % of 511 cpufp | Notes |
|---|---:|---:|---:|---|
| kloop-only (L1 8×16 tile) | **228** | 55.7 % | 44.5 % | 1.80 c/`vmadot`; piped inner loop |
| 8×16 block + store | **193** | 47.0 % | 37.6 % | +vectorized C scatter |
| pack (amortized) | — | — | — | ~0.33 GB/s gather (not MAC work) |
| compute (pre-packed) | **42** | 10.3 % | 8.2 % | 9.7 c/`vmadot` effective |
| **full GEMM** | **38** | 9.2 % | 7.4 % | pack + compute each rep |

**Unpadded malloc** on the same size hits **~24 GOP/s** full GEMM (L2 set aliasing;
compute drops to ~33 GOP/s, 12.5 c/`vmadot`) — layout can swing GEMM **~1.5×**
without changing the insn peak.

### Where the gap lives

1. **Silicon/scheduling (228 → 409.6):** even L1-resident kloop reaches only **~56 %**
   of the 1.6 GHz insn ceiling (~**2×** below cpufp’s 511 @ ~2 GHz). Production
   piped kloop is already the right shape (wider 8×32 tiles regress).
2. **Memory + panel structure (228 → 42):** full-problem compute is **~5.4× slower**
   than one L1 tile — panel loops, 512 KB shared L2, streaming `A`/`B`/`C`, and
   block dispatch outside the inner kloop. Implied **~9.7 c/`vmadot`** vs **1.8** in
   isolation.
3. **Pack (42 → 38):** amortized gather adds **~12–15 %** on good layouts (**~34 %**
   on bad malloc) — smaller than compute/memory but not free.
4. **Framework taxes (38 → llama ~22–28):** block-scale FP in `i8i4`/`i8i8`, short-K
   panels, and cluster-0 IME sharing cap end-to-end (documented above).

**Bottom line:** **~511 GOPS** is cpufp **insn peak** per core; **~42 GOP/s** is
**full-GEMM application throughput** at 1.6 GHz with a tuned kernel — roughly
**9 % of 409.6** and **7 % of 511**. Closing the kloop→compute gap needs memory/L2
blocking (already panelized), anti-alias layout, and possibly TCM — not wider tiles.
Closing kloop→409.6 is largely **hardware issue width / clock** on this part.

## Panel / memory tuning (step 2, 2026-09-02, RV2)

Runner: [`run-ime-panel.sh`](run-ime-panel.sh) → `ime-bench-panel`; log:
`~/logs/ime-panel-20260902-050915.log`.

| Lever | Result @768³×512 anti-alias | Action |
|---|---|---|
| **nc sweep** | nc=8 (16 KB B-panel) **36.2** GOP/s vs nc=64 **33.2** vs old auto **31.4** | Default **`nc = 4096/K`** TN-tiles |
| **Megakernel cache-touch** | compute **−5 %** (45.5 vs 48.0 GOP/s) | Off by default (`gemm_ime_set_megakernel`) |
| **Offline B** (static weights) | **+9.8 %** (43.1 vs 39.2 GOP/s) | `pack_b` once; repack A per GEMM |
| **TCM** `/dev/tcm` | present but **root-only** on RV2 | [`tcm.c`](tcm.c) + `ime-tcm-probe` (128 KiB blocks) |

Pack path now does **monolithic `pack_a` once** + `pack_b_panel` per N-slab (same bytes,
better cache behaviour than scattered `pack_a_panel`).

### TCM userspace allocator (no libspine_tcm)

[`tcm.c`](tcm.c) / [`tcm.h`](tcm.h) talk directly to `drivers/misc/tcm.c`:

- Block-aligned allocations only (**128 KiB** on K1/X60; `TCM_BLOCK_KB`)
- `TCM_REQUEST_MEM` + `poll` with timeout (`TCM_POLL_MS`, default 5 s) — never hangs forever
- Pin to cluster 0: `sudo taskset -c 0 ./ime-tcm-probe`

```bash
make board-tcm-probe
sudo taskset -c 0 ./ime-tcm-probe          # 1 block self-test
sudo taskset -c 0 ./ime-tcm-probe 2        # 256 KiB (2 blocks)
TCM_PROBE_FULL=1 sudo taskset -c 0 ./ime-tcm-probe  # all 512 KiB (opt-in)
```

Remote: [`run-tcm-probe.sh`](run-tcm-probe.sh).

**Non-root access:** `/dev/tcm` defaults to `root:root 0600`. One-time on the board:

```bash
sudo ./setup-tcm-perms.sh orangepi   # creates group tcm + udev rule
newgrp tcm                           # or log out/in
./ime-tcm-probe 1                    # no sudo
```

Or install vendor rules: `sudo apt install spacemit-tcm` (Bianbu/K3 images).

**B-panel in TCM** — three paths benchmarked @768³ nc=8:

| Path | GOP/s | vs full GEMM |
|------|------:|-------------:|
| full pack+compute | 44.0 | — |
| offline-B DRAM | 48.2 | +9.4% |
| **TCM offline-B** (pack B once → TCM) | **47.9** | **+8.8%** |
| **TCM offline-B + fused pack_a** | **68.0** | **+55%** |
| TCM repack each panel | 41.1 | −6.7% |
| TCM staged memcpy | 41.6 | −5.5% |

**Recommended production path** (static weights, ~68 GOP/s @768³):

```c
tcm_init(0);
int8_t *Bp_tcm = tcm_malloc((size_t)N * K);
pack_b(B, Bp_tcm, N, K);   /* once at load */
for (...) {
    gemm_ime_compute_tcm_offline_b(A, Ap, Bp_tcm, C, M, N, K, ldc);
}
```

Uses **RVV pack** (automatic on board) + **M-outer fused pack_a** + **TCM-resident B**.
Verified bit-exact vs `gemm_ref`. Do **not** use fused pack_a with DRAM `Bp` (−13%).

## Cross-board confirmation — Banana Pi BPI-F3 (same K1 / X60 SoC)

Same `make board` binary path on a [Banana Pi BPI-F3](https://www.banana-pi.org/)
(SpaceMiT K1, 8× X60 @ 1.6 GHz, `performance` gov, `taskset -c 0`), EESSI
`GCC/14.3.0`. Every path is `ok` (bit-exact vs scalar) at every size. Clean-layout
peaks (max over ≥5 launches; same malloc-aliasing bimodality as the RV2):

| M×N×K | scalar | RVV int8 | **IME (8×16 blocked)** | IME/RVV |
|---|--:|--:|--:|--:|
| 256×256×256   | 1.93 | 6.13 | **34** | 5.5× |
| 512×512×512   | 2.00 | 6.62 | **40** | 6.1× |
| 768×768×512   | 1.99 | 6.49 | **45** | 6.9× |
| 1024×1024×512 | 1.97 | 6.44 | **36** | 5.5× |

Same conclusion as the RV2: IME is **~6–7×** RVV int8 and crests near **45 GOP/s**
at 768³ before L2 pressure pulls large sizes down. (Scalar GOP/s here is higher
than the older RV2 table above — GCC 14.3 vs the earlier host toolchain — but
IME absolute peaks and IME/RVV ratios match.)

### Cache-set aliasing (buffer placement)

Every size here runs **bimodal**: some launches hit the peak above, others sit
~35 % lower (512³: 25 vs 39; 768³: 27 vs 42; 1024³: 22 vs 32; 2048³: 24 vs 34).
The mode is *fixed within a process* (all reps in one launch agree to ±3 %) but
*varies between launches* — a `malloc`-placement effect, not thermal/DVFS (the
governor is `performance`, pinned at 1.6 GHz). The L2 is **16-way, 512 sets, 64 B
lines → a 32 KB way-span** (two addresses share a set when congruent mod 32 KB).
At K=512 every buffer (`A`,`B` = M·512 / N·512 B; `C` = M·N·4 B) is an exact
multiple of 32 KB whenever M,N are multiples of 64 — true for **all** sizes here,
not just the powers of two — so when `malloc` lays the packed-A / packed-B / C
streams 32 KB-congruent they map to the same 16 ways and thrash. 768³ aliases
just like 512³; the effect is placement- not power-of-two-specific. The share of
"bad" launches drifts with heap and board state (~⅓–½ observed). A production
backend pads leading dimensions so the streams can't stay congruent; this
prototype leaves them unpadded.

## Comparison with llama.cpp's shipping IME kernel

llama.cpp already ships a SpaceMiT IME backend (`ggml-cpu/spacemit`,
`GGML_CPU_RISCV64_SPACEMIT`). Its kernel is a *block-scaled* Q4_0 GEMM
(`spacemit_kernels::ime1::gemm_kernel_i8i4`: int8 activations × 4-bit weights,
per-32-block fp16 scale, fp32 output) — not the pure `s8s8s32` this benchmark
measures. [`bench_i8i4.cpp`](bench_i8i4.cpp) links that kernel straight out of a
prebuilt `libggml-cpu.so` and times it the same way, on the same X60 core.
(Throughput is data-independent; the rate cross-validates against `llama-bench`
pp512 ≈ 20 GOP/s/thread. The harness paths are board-specific.)

**(A) The block-scale tax.** The `i8i4` kernel tops out at **~28 GOP/s ≈
0.6–0.7× our raw-int8 ceiling** — the per-block int32→fp convert + scale is real
work `s8s8s32` never does:

| M×N×K | llama.cpp `i8i4` | this `s8s8s32` | ratio |
|---|--:|--:|--:|
| 512×512×512   | 22.8 | 39 | 0.58 |
| 768×768×512   | 27.9 | 42 | 0.66 |
| 1024×1024×512 | 20.6 | 32 | 0.64 |
| 2048×2048×512 | 20.5 | 34 | 0.60 |

**(B) Aliasing — and a partial fix.** Unlike ours, the `i8i4` kernel is *not*
randomly bimodal (its 16×16 streaming tiles dodge the `malloc`-placement
lottery), but it pays a **deterministic power-of-two penalty on the C
write-stream**. Padding the output leading dimension by one tile (`ldc=N+16`)
recovers it — **+21 % at 512³** even after the scratch→dst copy — but the gain
**collapses as K grows** (+8 % at K=1024, +2 % at K=2048), because the aliasing
is a per-output C-write cost that large-K GEMMs amortise away. Real transformer
weight-matmuls have large K, so the practical payoff is low single digits (and
zero on non-power-of-two dims). Real and free, but not a broad IME speed-up.

**End-to-end** (`llama-bench`, Qwen2.5-0.5B Q4_0, this X60):

| build / threads | pp512 (t/s) | tg128 (t/s) |
|---|--:|--:|
| IME1 `-t4` (cluster 0) | 79.0 | 7.54 |
| IME1 `-t8`             | 56.7 | 3.71 |
| RVV  `-t8`             | 90.7 | 11.16 |

IME is **1.51× RVV at `-t4`** on prompt processing but is confined to the 4-core
cluster 0 — it *regresses* at `-t8`, and well-threaded RVV wins overall; IME
never helps decode. **Takeaway:** the real IME headroom is the ~40 % block-scale
gap in the kernel's fp path (needs a kernel rewrite, not a driver tweak), and
even closing it is capped by the cluster-0 constraint on this part.

## Cutting the block-scale tax (kernel optimization)

Following up on that "needs a kernel rewrite": the tax is **measurable and
reducible**. Rebuilding `libggml-cpu.so` from a patched `ime1_kernels.cpp` and
A/B-benchmarking on the X60 — correctness gated bit-for-bit by
[`bench_i8i4.cpp`](bench_i8i4.cpp)'s `chk` mode (a sum/sumsq/max signature vs the
stock kernel on identical varied inputs, since throughput can't catch a wrong
rewrite):

**Where the tax lives** (strip parts of the per-block FP reduction, M4 no-zp path):

| M×N×K | stock | −scale-build | −all FP (vmadot only) |
|---|--:|--:|--:|
| 512³ | 22.5 | 29.6 | 32.7 |
| 768³ | 27.5 | 36.7 | 41.3 |
| 512×512×2048 | 27.5 | 38.7 | 43.6 |

The ~31–37 % FP tax is **dominated by the per-block scale build**
(`LOAD_SCALE_4x16_FP16`: 16 masked `vfmul.vf` + widening + copies), not the
`vfcvt`+`vfmacc` convert/accumulate (~10 %). Removal scales ≈ linearly with
op-count, so the X60 does **not** overlap IME (`smt.vmadot`) with RVV FP — cutting
FP op-count cuts time.

**The fix** ([`llama-ime1-scalebuild-opt.patch`](llama-ime1-scalebuild-opt.patch)):
rebuild the combined `A·B` scale with **8 `vfmul.vv`** against two prebuilt
row-scale vectors instead of **16 split/masked `vfmul.vf` + 4 `vmv`** (also fewer
`vsetvli` toggles). Identical `v8..v15` layout → **bit-identical** output.

| kernel path | used by | 512³ | 768³ |
|---|---|--:|--:|
| M4 no-zero-point | Q4_0 prefill | 23.0→24.1 (+4.8 %) | 27.6→29.2 (+5.8 %) |
| M4 zero-point | Q4_1-style prefill | 20.6→21.7 (+5.3 %) | 24.7→26.0 (+5.3 %) |
| M1 / GEMV | decode (`tg`) | 12.1 — N/A | — |

**~5 % on both prefill GEMM paths, verified bit-exact.** M1/GEMV has no equivalent
tax (its single-row scale build is already 4 `vfmul.vf`, no masking) and is
memory-bound. The residual block-scale cost is fundamental here: the bigger lever
— software-pipelining the FP reduction under `vmadot` — is register-blocked (the
fp output, int accumulator, and scale vectors already fill the register file), and
IME stays cluster-0-capped end-to-end regardless.

**End-to-end (`llama-bench`, Qwen2.5-0.5B Q4_0, `-t4` on cluster 0, stock vs patched
`.so`).** Correctness holds — the model answers correctly (*"The capital of France
is Paris."*) and `tg128` is bit-stable at **7.25 t/s** on both (decode is the
untouched M1/GEMV path). But the prefill gain is **not resolvable above the board's
noise**: over 13 interleaved A/B rounds `pp512` came out statistically tied (stock
mean 79.8 / peak 91.3 vs patched 79.5 / peak 86.3 t/s), because `pp512` swings
±15–20 % run-to-run from shared-L2 contention and the `malloc`-placement aliasing
noted above — far larger than the ~3–4 % a +5 % GEMM kernel can add to a prefill
that is only partly matmul. So the kernel win is real and bit-exact but sits below
the application-level noise floor on this multi-tenant part; it would surface on a
quieter board or an IME part whose matrix unit spans more cores.

## Caveats

- **qemu-user cannot run this.** It does not emulate the SpaceMiT custom
  `vmadot` — the IME path only executes on real X60 silicon.
- The RVV baseline is correct and representative but **not** cache-tuned (it
  reloads B rows); it is the honest "plain RVV int8" floor the IME kernel beats,
  not a state-of-the-art RVV GEMM.
- int32 accumulator is exact for full int8 while `K ≲ 133000`.
- Dims must be multiples of the tile (`M%4=N%4=0`, `K%8=0`): the 8×16 kernel
  handles M/N *block* edges via the 4×4 `ime_tile`, but there is no sub-tile
  remainder path. Large-N throughput is L2-bound — the next tuning target.

SPDX-License-Identifier: MIT
