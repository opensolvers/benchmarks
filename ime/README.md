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
accumulate across K in `v16:v17`, then scatter the 4×4 result to `C`.

| File | Role | Portable? |
|---|---|---|
| `gemm_ref.c` | scalar reference + **shared packing** + packed scalar path | yes |
| `gemm_ime.c` | `smt.vmadot` microkernel (the core) | X60 only |
| `gemm_rvv.c` | RVV int8 baseline (`vwmul`+widening reduce) | X60 only |
| `bench.c` | fill / cross-check / GOP-s timing | yes |

Because the packing and tile-index math live in the **portable** `gemm_ref.c`
(`gemm_packed_ref` uses the exact same layout as `gemm_ime`), everything except
the two asm inner loops is verifiable off-target — see below.

## Build & run

```bash
# Portable self-test (x86, or any host) — verifies packing + harness, no RISC-V:
make host && ./ime-bench-host 256 256 128

# X60 with native binutils < 2.46 (no xsmtvdot) — IME via raw .insn. USE THIS on the RV2:
make board-insn && taskset -c 0-3 ./ime-bench 512 512 512

# X60 with an xsmtvdot-capable assembler (binutils >= 2.46 / SpaceMiT cross): readable mnemonic
make board && taskset -c 0-3 ./ime-bench 512 512 512

# X60, RVV baseline only:
make board-rvv && taskset -c 0-3 ./ime-bench
```

`make board` needs an assembler that knows `xsmtvdot`; `make board-insn` needs
none — it emits `smt.vmadot` as its raw instruction word (`.insn 0xe294382b`), so
any RVV-capable binutils builds it. Dims must satisfy `M%4=0, N%4=0, K%8=0`.

## Results (measured — Orange Pi RV2 / X60, single core, cluster 0)

Built with `make board-insn` (native binutils 2.42). Every path is `ok`
(bit-exact vs the scalar reference).

| M×N×K | scalar | RVV int8 | **IME `smt.vmadot`** | IME / RVV |
|---|--:|--:|--:|--:|
| 512×512×512 | 0.39 | 5.23 | **25.8** | 4.9× |
| 1024×1024×512 | 0.39 | 5.22 | **26.7** | 5.1× |
| 256×256×256 | 0.39 | 5.05 | **22.0** | 4.4× |

GOP/s. IME is ~5× the (deliberately naive) RVV int8 baseline and ~66× scalar,
consistent with the ~4× raw peak ratio (0.5 vs 0.128 TOPS/core) plus headroom
from the untuned RVV reference.

## Caveats

- **qemu-user cannot run this.** It does not emulate the SpaceMiT custom
  `vmadot` — the IME path only executes on real X60 silicon. On K1, IME lives on
  one 4-core cluster; pin with `taskset -c 0-3`.
- The RVV baseline is correct and representative but **not** cache-tuned (it
  reloads B rows); it is the honest "plain RVV int8" floor the IME kernel beats,
  not a state-of-the-art RVV GEMM.
- int32 accumulator is exact for full int8 while `K ≲ 133000`.
- Prototype scope: dims are multiples of the tile (no remainder/edge handling);
  single `4×4` accumulator per call (no register-blocking yet).

SPDX-License-Identifier: MIT
