# MODFLOW 6 — FlexiBLAS A/B (SpaceMiT X60 / Orange Pi RV2)

Groundwater flow (`MODFLOW/6.4.4-foss-2023b`) on EESSI `riscv.eessi.io`
**20240402**. Binary `mf6` links PETSc → FlexiBLAS (and MUMPS/Hypre/etc.).

## Workload

USGS example **`ex-gwf-lgrv-lgr`** (parent 25×90×78 + child 9×61×49), from
[modflow6-examples](https://github.com/MODFLOW-ORG/modflow6-examples)
`mf6examples.zip`.

Run with **`mpirun -np 1 mf6 -p`** so the PETSc linear path is active
(serial IMS without `-p` finishes the same model in ~54 s). A `.petscrc`
requesting MUMPS LU was present but left **unused** by MF6 6.4.4 (IMS
shell PC); wall time still exercises the PETSc/FlexiBLAS-linked binary.

## Results (2026-08-21)

| Tag | Axis | Wall s | MF6 elapsed |
|-----|------|-------:|-------------|
| scalar | `OPENBLAS_CORETYPE=RISCV64_GENERIC` | 189.61 | 3:03.7 |
| stock | default OpenBLAS via FlexiBLAS | 189.00 | 3:03.4 |
| patched | `FLEXIBLAS=~/libopenblas_x60_eb_fixed.so` | 189.98 | 3:04.3 |

Patched vs scalar ≈ **1.00×** (flat). `parent.hds` / `child.hds` MD5s match
across all three tags.

## Harness

```bash
# once
mkdir -p ~/mf6-bench && cd ~/mf6-bench
wget -O mf6examples.zip \
  https://github.com/MODFLOW-ORG/modflow6-examples/releases/download/current/mf6examples.zip
# A/B
bash run-modflow-blas-ab.sh
```

Log: `~/logs/modflow-blas-ab-20260821-221931.log`
