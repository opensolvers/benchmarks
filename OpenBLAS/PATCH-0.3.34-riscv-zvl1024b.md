# OpenBLAS 0.3.34 — `RISCV64_ZVL1024B` (SpacemiT A100 / VLEN=1024)

Adds a new OpenBLAS target for RISC-V Vector with **minimum VLEN 1024**, aimed at
SpacemiT **K3 A100** AI cores (Banana Pi BPI-SM10). Companion to the ZVL256B
path used on X100 / X60.

## Files

| File | Role |
|------|------|
| `OpenBLAS-0.3.34_add-riscv64-zvl1024b.patch` | Target wiring + generated GEMM/TRMM kernels (`dgemm` **16×8**, `sgemm` 16×8, `cgemm` 8×8, `zgemm` 8×4) via upstream `kernel/riscv64/generate_kernel.py` |
| `OpenBLAS-0.3.34_zvl256b-vlen-portable-gemm-vget.patch` | Makes stock ZVL256B GEMM safe if ever run at `vlenb > 32` (split wide `vle`+`vget`) |
| `build-openblas-zvl1024b-k3.sh` | Example static `TARGET=RISCV64_ZVL1024B` build + A100 DGEMM smoke (k3) |

`DYNAMIC_ARCH` dispatch (in the ZVL1024B patch): `vlenb ≥ 128` → ZVL1024B,
`≥ 32` → ZVL256B, else ZVL128B.

## Apply / build

```bash
tar xzf OpenBLAS-0.3.34.tar.gz && cd OpenBLAS-0.3.34
patch -p1 < OpenBLAS-0.3.34_add-riscv64-zvl1024b.patch
# optional: also apply zvl256b-vlen-portable-gemm-vget.patch for dual DYNAMIC_ARCH

# A100-only static (do not run the BLAS test suite on X100 — VLEN too short)
make -j$(nproc) libs netlib shared \
  TARGET=RISCV64_ZVL1024B USE_OPENMP=1 NUM_THREADS=8
```

Or rebuild `DYNAMIC_ARCH=1` so one `.so` serves X100 (ZVL256B) and A100 (ZVL1024B).

## HPL on BPI-SM10 (k3), 2026-09-27

EESSI `HPL/2.3-foss-2025b`, NB=192, FlexiBLAS backend swap. A100 ranks use
`/proc/set_ai_thread` then pin to cores 8–15.

| Setup | GFLOP/s @ N=12000 |
|-------|------------------:|
| X100 ×8, ZVL256B | 52–53 |
| A100 ×8, ZVL256B (portable) | 15.0 |
| A100 ×8, **ZVL1024B** | **35.9** |
| Hetero 8 X100 + 8 A100 (ZVL256B + **ZVL1024B**) | **57.5** |

Board write-up: [opensolvers SM10](https://www.opensolvers.com/boards/SM10.html).
