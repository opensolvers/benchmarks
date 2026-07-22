# SpaceMiT X60 (K1) — ISA & codegen notes

Reference notes on the **SpaceMiT X60** RISC-V core (SoC: SpaceMiT **K1** /
"M1"; boards: **Orange Pi RV2**, **Banana Pi BPI-F3**) — specifically **what
ISA it implements and how to make a compiler target it correctly**. These are
the facts you need before building any vectorized HPC kernel for this board.

## TL;DR

- **8× SpaceMiT X60 @ 1.6 GHz**, in-order.
- **RVV 1.0 — ratified, `v` in the ISA string, `VLEN=256` (`zvl256b`)**.
  *Not* the 0.7.1 draft. Full 32-register `v` with `zve*`/`zvl*` sub-extensions.
- Kernel-reported ISA is `rv64imafdcv` + ~26 sub-extensions (`Zicbom`, `Zba`,
  `Zbb`, `Zbc`, `Zbs`, `Zfh`, `Zvfh`, `Zvkt`, …).
- **RISC-V GCC has *no* native CPU detection** — `-march=native`,
  `-mcpu=native`, and `-mcpu=spacemit-x60` all **fail** on this toolchain. You
  **must** pass an explicit `-march=rv64…` string. EESSI already does, and its
  string is X60-correct.

## Microarchitecture

| | |
|---|---|
| SoC | SpaceMiT K1 (a.k.a. M1) |
| Cores | 8× X60 @ 1.6 GHz |
| Vector | **RVV 1.0**, `VLEN=256` bits, `ELEN=64` |
| Board (this repo) | Orange Pi RV2, 8–16 GB LPDDR4 |
| Toolchain | EESSI 2025.06 / `dev.eessi.io` riscv overlay, **GCC 14.3.0** |

## The ISA string

The kernel advertises the following in `/proc/cpuinfo` (`isa :` line). All 30
tokens, canonical order:

```
rv64imafdcv
  zicbom zicboz zicntr zicond zicsr zifencei zihintpause zihpm
  zfh zfhmin
  zca zcd
  zba zbb zbc zbs
  zkt
  zve32f zve32x zve64d zve64f zve64x
  zvfh zvfhmin zvkt
  sscofpmf sstc svinval svnapot svpbmt
```

What matters for **code generation**:

- **`v`** — full RVV 1.0 vector (implies the `zve*` embedded-vector subsets and
  `zvl*` up to the machine `VLEN=256`).
- **`f d`** — hardware single + double FP (so `-mabi=lp64d`).
- **`zba zbb zbc zbs`** — bit-manip (address-gen, basic, carry-less, single-bit).
- **`zfh zfhmin` / `zvfh zvfhmin`** — scalar + vector half-precision FP.
- **`zicbom zicboz`** — cache-block management / zero (only `zicbom` is the
  block-op set; **there is no `zicbop` prefetch** on this core).
- **`zkt`, `zvkt`** — data-independent (constant-time) execution latency
  guarantees (crypto), not perf-relevant for GEMM/FFT.
- **`ss*` / `sv*`** (`sscofpmf`, `sstc`, `svinval`, `svnapot`, `svpbmt`) are
  **supervisor / OS-facing** (counter-overflow, stimecmp, TLB-invalidate,
  NAPOT page, page-based memory types). The kernel uses them; they are **not
  code-generation-relevant** for a user-space compute kernel.

## RISC-V GCC has no `-march=native` — verified on the board

Unlike x86 (which resolves `-march=native` via `CPUID`) and AArch64 (via
`HWCAP` / MIDR), **the RISC-V GCC 14.3.0 back-end has no runtime native
detection at all.** All three "just detect it" spellings fail on the X60:

```
$ gcc -mcpu=native ...
  → error: unknown CPU 'native'
$ gcc -march=native ...
  → error: ISA string must begin with rv32 or rv64
$ gcc -mcpu=spacemit-x60 ...
  → error: unknown CPU 'spacemit-x60'      # not in this GCC build's core DB
```

**Consequence:** the RISC-V GCC contract is an **explicit `-march=` string** —
there is no autodetection fallback. Anything relying on `native` (e.g. a build
system, or **Kokkos's `ARCH_NATIVE`**) will either error out or silently emit
scalar `rv64gc`.

## EESSI's `-march` is already X60-correct

EESSI compiles with an explicit `-march` that **matches the kernel's X60 ISA
exactly** (same 30 extensions, only reordered), plus the explicit `zvl` tokens:

```
-march=rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_
       zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_
       zvfh_zvfhmin_zvkt_zvl128b_zvl32b_zvl64b_sscofpmf_sstc_svinval_svnapot_svpbmt
  -mabi=lp64d -misa-spec=20191213
```

So for HPC builds on this board you do **not** need to change `-march`: the
EESSI default already targets the X60 correctly. The only place "native" bites
is a build system that *asks* for it — see the Kokkos note below.

### Practical build lever (Kokkos / LAMMPS, seen in this repo's stack)

Kokkos defaults to `Kokkos_ARCH_NATIVE=ON`, which makes it append
`-mcpu=native -mtune=native` → **fails on RISC-V GCC** (per above). The fix is
to route Kokkos to its no-op generic arch so it adds **no** `-m` flags and
inherits the EESSI `-march`:

```python
# in the LAMMPS/Kokkos easyconfig
kokkos_arch = 'EASYBUILD_GENERIC'    # -> -DKokkos_ARCH_EASYBUILD_GENERIC=yes, a no-op target
```

(Setting `-DKokkos_ARCH_NATIVE=OFF` via `configopts` alone does **not** work —
the LAMMPS easyblock appends its own `Kokkos_ARCH_*=yes` *after* your
`configopts`, so it wins. The `kokkos_arch` easyconfig parameter is the
sanctioned lever.)

## Where the real X60 wins are

The vector FP (`v`, VLEN=256) is genuine but, on the whole-application
benchmarks in this repo, drop-in vectorized BLAS/FFT libraries **do not move a
real DFT/MD run much**. The distinctive X60 win is its **IME** integer-matrix
extension (`smt.vmadot`, an `Xsmtvdot`-style vendor op) for int8/int4 GEMM —
see [`../../ime/`](../../ime) and [`../../onnx/`](../../onnx).

## Sources

- Verified on-board (`/proc/cpuinfo`, `gcc -mcpu=native` / `-march=native` /
  `-mcpu=spacemit-x60` probes) on the Orange Pi RV2, EESSI GCC 14.3.0.
- `riscv-x60.md` — full X60 / OpenBLAS / EESSI working notes.
- SpaceMiT K1/X60 documentation; RISC-V RVV 1.0 spec; RISC-V `Zvl*` / `Zve*` /
  bit-manip (`Zb*`) extension specs.
