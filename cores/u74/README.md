# SiFive U74 — ISA & codegen notes

Reference notes on the **SiFive U74** RISC-V core (SoC: StarFive **JH7110**;
board: **StarFive VisionFive 2**) — specifically **what ISA it implements and
how to make a compiler target it correctly**. Companion to
[`../x60/`](../x60); the two cores sit at opposite ends of the RISC-V spectrum
(U74 = scalar RVA20-class, X60 = RVV 1.0 vector), so the codegen story differs
sharply.

## TL;DR

- **4× SiFive U74 @ 1.5 GHz**, in-order, dual-issue.
- **`rv64gc` — scalar. There is NO vector extension** (no `v`, no `zve*`),
  and on the JH7110 **no `zba`/`zbb`** bit-manip and **no `Zicbop`** prefetch.
- `rv64gc` == `rv64imafdc` (the `g` shorthand = `imafd` + `zicsr` + `zifencei`).
- Same RISC-V GCC rule as every RISC-V core: **no `-march=native`** — pass an
  explicit `-march=rv64…` string.
- For a **tuned** OpenBLAS build the target flags are
  `-march=rv64imafdc_zba_zbb -mabi=lp64d -mtune=sifive-u74` — but note the
  `_zba_zbb` there is a *tuning convenience for the kernel that assembles
  cleanly*; the **hand-written asm microkernel is pure `rv64gc`** (no `zba`/`zbb`,
  no `.option arch`) precisely because the JH7110 U74 lacks those extensions.

## Microarchitecture

| | |
|---|---|
| SoC | StarFive JH7110 |
| Cores | 4× SiFive U74 @ 1.5 GHz |
| Vector | **none** (scalar `rv64gc`) |
| Pipeline | dual-issue **in-order**, **single FP pipe** |
| FP | `fmadd.d` fully pipelined, ~7-cycle latency, 1/cycle throughput |
| Memory | **one load port**; L1D 32 KB / 64 B line; **2 MB shared L2** |
| Board (this repo) | StarFive VisionFive 2, 8 GB, Ubuntu 24.04 |
| Toolchain | EESSI 2025.06, **GCC 14.3.0** |

Peak DP ≈ **3.0 GFLOP/s/core** (12 GF board). Empirical ceilings measured on
the VF2: a load co-issues *free* with an FMA (distinct operands); the FP-pipe
peak is ≈ **2.90 GFLOP/s/core**; and the streaming DGEMM plateau is
**memory-latency-bound**, not FP-bound (one load port + shared L2).

## The ISA string

```
rv64gc   ==   rv64imafdc  (+ implied zicsr, zifencei)
```

- **`i m a f d c`** — base integer, mul/div, atomics, single + double FP,
  compressed. `d` ⇒ `-mabi=lp64d`.
- **`c`** — compressed 16-bit encodings.
- **No `v`** — *no vector unit at all.* Any RVV kernel is irrelevant here (this
  is why the X60's RVV `gemv_n` NaN bug — see [`../x60/`](../x60) — **cannot**
  affect the U74; the scalar U74 has no RVV kernels to be buggy).
- **No `zba`/`zbb`** on the JH7110 U74, and **no `Zicbop`** prefetch. Code that
  emits address-generation (`zba`) or basic bit-manip (`zbb`) instructions will
  fault. The DGEMM asm microkernel is therefore deliberately written to
  assemble under **plain `-march=rv64gc`** (no `.option arch` bumping the ISA).

## RISC-V GCC has no `-march=native`

Same contract as every RISC-V core (see [`../x60/`](../x60) for the on-board
proof): the GCC 14.3.0 RISC-V back-end has **no native detection** — `-march=native`
errors (`ISA string must begin with rv32 or rv64`) and `-mcpu=native` errors
(`unknown CPU`). You must give an explicit string. There is, however, a valid
**`-mtune`**: `-mtune=sifive-u74` (scheduling only — it changes instruction
*scheduling*, never which instructions are legal, which is governed solely by
`-march`).

## Build flags

**Generic / stock** (what a default `rv64gc` build uses):

```
-march=rv64gc -mabi=lp64d
```

**U74-tuned** (the OpenBLAS `U74` target added upstream — see below):

```
-march=rv64imafdc_zba_zbb -mabi=lp64d -mtune=sifive-u74
```

`rv64imafdc` is exactly `rv64gc`. The `_zba_zbb` appears in the OpenBLAS
`Makefile.prebuild` `TARGET_FLAGS` / `Makefile.riscv64` `CCOMMON_OPT` for the
compiler-emitted C kernels; the **critical hand-asm microkernel stays pure
`rv64gc`** so it never emits a `zba`/`zbb` instruction the hardware lacks.

## The OpenBLAS `U74` target (measured wins)

Adding a dedicated `U74` target + a hand-written 4×4 DGEMM asm microkernel
(software-pipelined, one-load-port aware) beat the generic `rv64gc` build on
the VisionFive 2:

| Metric | Stock (`rv64gc`) | U74-tuned | Gain |
|---|---|---|---|
| HPL, N=10000, 4 cores | 3.08–3.13 GFLOP/s | **5.28 GFLOP/s** | **1.69×** |
| DGEMM, single core | ~1.4 GFLOP/s | **1.77 GFLOP/s** | ~1.26× |
| DGEMM, 4 cores | ~3 GFLOP/s | **6.31 GFLOP/s** | ~2× |

HPL trails raw DGEMM because of its non-GEMM work (panel factorization,
pivoting, comms). Build: `make TARGET=U74 DYNAMIC_ARCH=0 -j4`.

Upstreamed:

- OpenBLAS source PR **OpenMathLib/OpenBLAS#5903** —
  <https://github.com/OpenMathLib/OpenBLAS/pull/5903>
- EasyBuild easyconfig PR **easybuilders/easybuild-easyconfigs#26436** —
  <https://github.com/easybuilders/easybuild-easyconfigs/pull/26436>

Files touched by the target (all in
`OpenBLAS-0.3.30_add-sifive-u74-target.patch`): `TargetList.txt`, `getarch.c`
(`FORCE_U74` block after `FORCE_RISCV64_GENERIC`), `cpuid_riscv64.c`, `param.h`
(U74 params, `Q=256`), `Makefile.prebuild`, `Makefile.riscv64`,
`kernel/riscv64/KERNEL.U74`, and the kernels `kern_u74.S` /
`gemmkernel_4x4_u74.c`.

## Sources

- `riscv-u74.md` — full U74 / OpenBLAS / EESSI working notes (measured on the
  VisionFive 2, JH7110, 4× U74 @ 1.5 GHz).
- `u74-openblas-gemm-optimization.md` / `.pdf` — the full design paper.
- `blog-hpl-u74-eessi.md` / `.pdf` — fresh-install HPL reproduction walkthrough.
- RISC-V unprivileged ISA (`rv64gc` = `rv64imafdc` + `zicsr`/`zifencei`);
  SiFive U74 core complex manual.
