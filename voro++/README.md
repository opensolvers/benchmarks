# Voro++ — RVV auto-vec A/B (SpaceMiT X60 / Orange Pi RV2)

Cell-based 3D Voronoi tessellation ([Voro++ 0.4.6](http://math.lbl.gov/voro++/)).
Stock EESSI `Voro++/0.4.6-GCCcore-14.3.0` is built `-march=rv64gc` only; this
harness rebuilds the library from upstream for a compiler-flag A/B.

## Result (2026-08-21)

| Variant | Flags | `cell.o` RVV-ish insn | best ms (N=20k) |
|---------|-------|----------------------:|----------------:|
| **novec** | `-O3 -march=rv64gc -fno-tree-vectorize` | 0 | **1244.8** |
| **gcv** | `-O3 -march=rv64gcv -ftree-vectorize` | 309 | 1261.6 |

**Speedup gcv vs novec: 0.99×** (RVV auto-vec ≈ flat / slightly slower).

Checksums match: `VVOL=1`, `FACES=297872`, `|Δvol| ~ 5e-15`.

Interpretation: Voro++ is irregular (per-cell plane cuts, short variable-length
loops, pointer-heavy). GCC does emit RVV in `cell.o`, but it does not pay off
on this workload — useful negative control next to waLBerla SoA auto-vec wins.

Log: `~/logs/voro-autovec-ab-20260821-170022.log` on the board.

## Reproduce

```bash
# on RV2
bash ~/voro-harness/run-voro-autovec-ab.sh
# or from this dir after scp:
VORO_N=20000 VORO_REPS=5 bash run-voro-autovec-ab.sh
```

Requires `GCCcore/14.3.0` via EESSI `2025.06-001`; downloads upstream
`voro++-0.4.6.tar.gz` into `~/voro-ab/` on first run.
