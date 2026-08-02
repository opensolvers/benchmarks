# Collide-SoA then stream (vs fused CellwiseSweep)

Fork of waLBerla tutorial `01_BasicLBM` with `sweepMode`:

| Mode | Path |
|---|---|
| `stock` | Fused `lbm::makeCellwiseSweep` (stream-pull + collide per cell) |
| `split` | `CellwiseSweep::stream` (gather) then contiguous **SoA collide** (`SoaCollideKernels.cpp`) |

Physics: D2Q9 incompressible SRT matching CellwiseSweep’s D2Q9 specialization (plain `double`, no `walberla::simd`, no pystencils).

## Layout

| Piece | Path |
|---|---|
| App sources | [`apps/collide_stream_split/`](../../apps/collide_stream_split/) |
| CMake snippet | [`CMakeLists.snippet.txt`](CMakeLists.snippet.txt) |
| Script | [`scripts/collide-stream-split-ab.sh`](../../scripts/collide-stream-split-ab.sh) |
| PRMs | [`prm/01_BasicLBM_collide_stream_split*.prm`](../../prm/) |
| Results | [`results/collide-stream-split.txt`](../../results/collide-stream-split.txt) |

On the board, the script copies sources into
`~/walberla-bench/src/walberla-7.2/apps/tutorials/lbm/` and builds target
`01_BasicLBM_CollideStreamSplit` inside `build-gcv`.

## Caveats

- Collide **does** auto-vectorize (GCC `-fopt-info-vec-optimized`).
- Stream remains a **gather** (neighbor `get` / row copies) and is not expected to auto-vec well.
- Split does **two** full-field passes per step vs one fused pass → often **slower** overall on bandwidth-bound LBM despite a vectorized collide.
