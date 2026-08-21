# OSU Micro-Benchmarks — on-node MPI interconnect baseline

Calibrates OpenMPI / UCX shared-memory performance on the SpaceMiT X60 so
MPI-heavy app results (HPL, ScaLAPACK, QE, PETSc, …) can be read against a
known latency/bandwidth floor.

OSU itself is not vendored — load `OSU-Micro-Benchmarks/7.5.1-gompi-2025b` from
`dev.eessi.io/riscv`.

## Run

```bash
# on the board (forces on-node shared-memory BTLs: self+vader)
bash run-osu-baseline.sh
# log: ~/logs/osu-baseline-*.log
```

Suite: `osu_latency` / `osu_bw` / `osu_bibw` (np=2), then `osu_allreduce` /
`osu_bcast` / `osu_alltoall` (np=8, full X60).

## Results — Orange Pi RV2 (2026-08-21)

`OSU-Micro-Benchmarks/7.5.1-gompi-2025b`, OpenMPI 5.0.8, on-node
`pml=ob1 btl=self,vader`, `--bind-to core`. Log
`~/logs/osu-baseline-20260821-143550.log`.

### Point-to-point (np=2)

| metric | value |
|---|--:|
| Latency @ 1 B | **1.12 μs** |
| Latency @ 1 KiB | 2.73 μs |
| Latency @ 1 MiB | 640.56 μs |
| Uni-directional BW peak | **2069 MB/s** (@ 512 KiB) |
| Bi-directional BW peak | **2456 MB/s** (@ 256 KiB) |

### Collectives (np=8)

| test | latency @ 4 B / 1 KiB / 1 MiB |
|---|---|
| `osu_allreduce` (`MPI_INT`) | 6.96 / 18.51 / 13410 μs |
| `osu_bcast` | 3.04 / 8.86 / 3758 μs |
| `osu_alltoall` | 10.97 / 41.90 / 24411 μs |

## How to read this against app benches

- Sub-2 μs shared-memory latency and ~2 GB/s uni-BW mean pure MPI overhead on
  this board is small next to BLAS/FFT work in QE / HPL / GROMACS.
- When an 8-rank ScaLAPACK/PETSc run looks communication-bound, compare its
  message sizes to the collective table above — `alltoall` at ≥64 KiB is already
  multi-millisecond per call.
- This is **on-node only**. Cross-node Ethernet numbers need a second board.
