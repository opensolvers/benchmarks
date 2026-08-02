# `01_BasicLBM_CollideStreamSplit`

Fork of waLBerla 7.2 `apps/tutorials/lbm/01_BasicLBM` with `Parameters.sweepMode`:

- `stock` — fused `lbm::makeCellwiseSweep`
- `split` — stream-pull (`CellwiseSweep::stream`) then contiguous SoA collide (`SoaCollideKernels.cpp`)

Build/run via [`scripts/collide-stream-split-ab.sh`](../../scripts/collide-stream-split-ab.sh).
See [`patches/collide-stream-split/README.md`](../../patches/collide-stream-split/README.md) and the main [`walberla/README.md`](../../README.md) section.
