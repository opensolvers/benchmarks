# HeatEquation bench fork

Prm-driven fork of waLBerla 7.2 `apps/tutorials/pde/02_HeatEquation.cpp`:

- Keeps **stock Jacobi** (`WALBERLA_FOR_ALL_CELLS_XYZ` + D2Q5 neighbors)
- **VTK off**
- Reads `DomainSetup` / `Parameters` from prm
- Requires `np == blocks.x*y*z`
- Prints `checksum_u_sum` for A/B correctness

Used by [`scripts/heat-equation-rvv-ab.sh`](../../scripts/heat-equation-rvv-ab.sh).
