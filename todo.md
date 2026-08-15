# RV2 verification TODO — apps to build/test on the SpaceMiT X60

Orange Pi RV2 (SpaceMiT X60, `rv64gcv`, RVV 1.0, VLEN=256, 8× @ 1.6 GHz).

**Active:** PETSc hand-RVV SpMV probe **done** — CSR RVV ≈ no win; structured
5-pt stencil RVV **~3.6×** vs PETSc `MatMult` on RV2. See [`petsc/`](petsc).

Two RISC-V software sources are mounted on the board:

- **`riscv.eessi.io` `20240402`** — production RISC-V stack, **`foss-2023b`**
  (GCC-13.2.0), `riscv64/generic`, **403 modules**.
- **`dev.eessi.io/riscv` `2025.06-001`** — newer dev port, **`foss-2025b`**
  (matches the toolchain the current repo benchmarks are built against);
  app modules: GROMACS 2026.2, ESPResSo 4.2.2, HPL 2.3, PLUMED 2.9.4,
  OSU 7.5.1, plus ScaFaCoS, waLBerla, MUMPS, ELPA, ScaLAPACK, SCOTCH, Voro++.

(`software.eessi.io` itself has **no** riscv64 — only aarch64/x86_64 — so all
RISC-V apps come from the two repos above.)

---

## Part A — benchmarks already in this repo (verify end-to-end)

These already have a directory + runner; "ready" = lib/artifact confirmed on
board. Unchanged from prior tracking.

- [ ] **onnx** — int4 `MatMulNBits` on ONNX Runtime, X60 IME `smt.vmadot` core.
  Reproduce ~9–10× `accuracy_level=4` CompInt8-vs-CompFp32 + correctness gate.
  *Lib:* custom `ONNX-Runtime/1.29.0-foss-2025b-xsmtvdot`. **Highest priority —
  extends the shipped IME work.**
- [ ] **ime** — s8s8s32 int8 GEMM microkernel, bit-exact vs RVV. *(shipped:
  `ime/` + `llamacpp/`; remaining = the `i8i8-selftest` A/B row.)*
- [ ] **BLIS** — dgemm/trsm BLAS-3 vs OpenBLAS (link A/B). *Lib:* `BLIS`.
- [ ] **OpenBLAS / dgemm** — dgemm GFLOP/s + `gemv_n` NaN / TRSM correctness.
  *Lib:* FlexiBLAS + custom `libopenblas_x60_eb_fixed.so`.
- [ ] **numpy** — `A@B` (dgemm) + `eigvalsh` (dsyevd) finite-gate. *Lib:* SciPy-bundle.
- [ ] **hpl** — Linpack end-to-end, FlexiBLAS backend swap. *Lib:* `HPL/2.3`.
- [ ] **elpa** — dense real-symmetric eigensolver finite-gate. *Lib:* `ELPA`.
- [ ] **scalapack** — `pdsyev`, pure-MPI 8-rank grid. *Lib:* `ScaLAPACK`.
- [ ] **fftw** — r5v (RVV) vs scalar FFT MFLOPS. *Artifact:* A/B pair built.
- [ ] **gromacs** — MD FFT + RVV-Force backend A/B. *Lib:* `GROMACS/2026.2-foss-2025b`.
- [ ] **qe** — Quantum ESPRESSO `pw.x` SCF, FlexiBLAS swap. **Blocker:** no
  `QuantumESPRESSO` module in either RISC-V repo yet → must build from an
  easyconfig (or use the `dev.eessi.io/espresso` tree, which currently ships
  only aarch64 ESPResSo, not QE). Confirm/produce a `riscv64` QE build first.

---

## Part B — available on the board but NOT yet benchmarked (the gap)

Apps/solvers present as loadable modules in `riscv.eessi.io 20240402`
(foss-2023b) and/or `dev.eessi.io/riscv 2025.06-001` (foss-2025b) with **no
directory in this repo**. Each is a candidate new benchmark. Ordered by
expected signal.

### B1 — real MD/materials applications (highest value: whole-app A/B)

- [x] **LAMMPS** (`22Jul2025_update4-foss-2025b-kokkos` on RV2) — built & compute
  verified (`lmp -h`, `in.melt`); installed under `eessi-overlay` (EB failed at
  `ctest`, manual install). Bench set in `~/lammps-bench`. Learnings:
  [`kokkos/`](kokkos); RVV Pair prototype + `lj/cut/rvv` plugin:
  [`lammps/rvv-lj/`](lammps/rvv-lj) (microbench ~1.61× vs scalar; in-LAMMPS
  ~1.02× vs stock via indexed gather).
- [ ] **ESPResSo** (`4.2.2` in **both** 2023b and 2025b) — soft-matter MD;
  P3M long-range solver → FFT-backend A/B (FFTW r5v vs scalar), the MD
  companion to `fftw/` and `gromacs/`. *No repo dir.*
- [ ] **MetalWalls** (`21.06.1-foss-2023b`) — constant-potential electrochem MD;
  heavy on Ewald/FFT + dense linear algebra → BLAS/FFT backend A/B. *2023b only.*
- [x] **waLBerla** (`7.2-foss-2025b`, RV2) — campaign in [`walberla/`](walberla):
  BasicLBM ISA ~1–4%; HeatEquation gcv **1.64×** np1; UniformGrid `--not-fused`
  WALL **1.30×** / collide **1.54×**; hand RVV `simd` loses to FORCE_SCALAR;
  plain SoA auto-vec ~**2.4×** vs novec (~**9×** vs hand simd); collide/stream
  split slower than fused stock. Prefer contiguous auto-vec over hand simd.
- [ ] **PLUMED** (`2.9.4-foss-2025b` / `2.9.2-2023b`) — enhanced-sampling library;
  bench its internal matrix/CV kernels, or use as a GROMACS/LAMMPS plugin to
  measure RVV overhead on collective-variable evaluation. *No repo dir.*
- [ ] **MODFLOW** (`6.4.4-foss-2023b`) — groundwater flow FE solver; sparse-solve
  dominated → pairs with the PETSc/MUMPS column below. *2023b only, no repo dir.*

### B2 — sparse / eigen solvers (one FlexiBLAS-or-link swap validates several)

- [x] **PETSc** (`3.24.0-foss-2025b` overlay) — Jacobi-CG ~1.06×; **dense MatMult
  ~1.70×** (stock RVV NaN); SuperLU/UMFPACK need patched OpenBLAS; MUMPS finite
  but no speedup at tested sizes ([`petsc/`](petsc)). SLEPc not built.
- [ ] **MUMPS** standalone microbench — PETSc-wrapped MUMPS done in `petsc/`; optional
  larger 3D frontal A/B still open.
- [x] **SuperLU_DIST** — exercised via PETSc direct bench (stock RVV NaN; patched OK).
- [x] **SuiteSparse / UMFPACK** — exercised via PETSc direct bench (stock RVV NaN; patched OK).
- [ ] **ScaFaCoS** (`foss-2025b`, dev repo) — scalable long-range Coulomb solver
  (FMM/P3M); FFT + MPI probe, new in the 2025b port. *No repo dir.*

### B3 — networking / baseline microbenchmarks (cheap, high-confidence)

- [ ] **OSU-Micro-Benchmarks** (`7.5.1-gompi-2025b` / `7.2-2023b`) — MPI
  latency/bandwidth baseline for the board's OpenMPI/UCX stack; needed context
  for interpreting every MPI app above (scalapack/qe/petsc/superlu). **Do this
  early — it's fast and calibrates the interconnect.** *No repo dir.*
- [ ] **Voro++** (`0.4.6`) — Voronoi tessellation; scalar compute-bound kernel,
  a simple RVV-autovec vs `-fno-tree-vectorize` A/B. *No repo dir.*

### B4 — data-science / numerical stacks (Python/R, FlexiBLAS-backed)

- [ ] **scikit-learn** (`1.4.0-gfbf-2023b`) — its BLAS-backed estimators (PCA,
  kmeans, linear models) ride FlexiBLAS → free RVV-OpenBLAS A/B on a real ML
  workload, extending `numpy/`. *No repo dir.*
- [ ] **R** (`4.3.3` / `4.4.1-gfbf`) — R's `%*%` / `La_*` LAPACK path through
  FlexiBLAS; a second high-level consumer of the RVV BLAS column. *No repo dir.*
- [ ] **Armadillo** (`12.8.0-foss-2023b`) — C++ linear-algebra template lib over
  BLAS/LAPACK; thin-wrapper dgemm/eig A/B. *No repo dir.*
- [ ] **VTK / ParaView** (`foss-2023b`) — visualization/data pipelines; lower
  priority, mostly I/O + scalar. *No repo dir — likely defer.*

---

## Part C — high-impact apps with an easyconfig but NOT yet built for RISC-V EESSI

EasyBuild 5.0.0 (+ 4.9.1/4.9.4) is on the board, and the full easyconfig
archive ships on cvmfs. These apps have an upstream easyconfig but **no
riscv64 module** in either RISC-V repo — i.e. buildable via EasyBuild, just
never built for the X60. Split by how far the newest *foss* easyconfig is from
the board's `foss-2023b` / `foss-2025b` stack.

### C1 — ready foss-2023b easyconfig (build as-is, no bump)

- [ ] **OpenMM** (`OpenMM-8.1.2-foss-2023b.eb`) — GPU/CPU MD engine; on CPU it's
  a dense-kernel + FFT workload → RVV force/PME probe. Build straight from the
  2023b EC. *No riscv module.*
- [ ] **MUMPS** (`5.7.2-foss-2023b-parmetis`) — newer multifrontal solver than the
  5.6.1 already present; build to compare + feed the B2 sparse-solver column.

### C2 — one toolchain-bump away (newest foss EC is 2023a/2024a → rebump to 2023b/2025b)

These are the flagship HPC apps. Each has a mature foss easyconfig one step
below the board stack; the work is a toolchain rebump + riscv64 fixes, **not** a
from-scratch port. High scientific impact, high effort.

- [ ] **QuantumESPRESSO** (`7.4-foss-2024a`) — **unblocks the existing `qe/`
  benchmark**, which currently has no runnable QE module. Highest-value C2 item:
  bump 7.4 to foss-2025b, build, then the FlexiBLAS/FFT A/B in `qe/` runs.
- [ ] **CP2K** (`2023.1-foss-2023a`) — DFT/AIMD; heavy DBCSR sparse-GEMM +
  FFT → premier RVV BLAS/FFT whole-app probe. *No riscv module.*
- [ ] **WRF** (`4.6.1-foss-2024a-dmpar`) — numerical weather prediction; large
  stencil + MPI, a bandwidth/interconnect showcase on the X60. *No riscv module.*
- [ ] **OpenFOAM** (`v2406-foss-2023a`) — finite-volume CFD; sparse linear solve
  dominated, the CFD counterpart to waLBerla's LBM. *No riscv module.*
- [ ] **Siesta** (`5.2.2-foss-2023a`) — linear-scaling DFT; ScaLAPACK/ELPA-backed
  → rides the dense-eigen column. *No riscv module.*
- [ ] **NAMD** (`3.0-foss-2024a-mpi`) — large-scale MD; Charm++ + FFT. *No riscv module.*
- [ ] **DFTB+** (`24.1-foss-2023a`) — approximate DFT; dense LA + eigensolve.
- [ ] **BerkeleyGW** (`4.0-foss-2022a`) — GW/BSE many-body; FFT + ScaLAPACK,
  older EC (2022a) so a larger bump. *No riscv module.*
- [ ] **ABINIT** (`9.6.2-foss-2022a`) — plane-wave DFT; another 2022a-era bump.

### C3 — no foss easyconfig yet (hardest; intel-only or bespoke)

- [ ] **NWChem**, **Yambo**, **Octopus** — easyconfigs exist but **none on a
  `foss` toolchain** (intel/iompi-only). Would need a new foss easyconfig
  authored before a RISC-V build is even attemptable. Defer unless specifically
  requested.

---

## Part D — VisionFive 2 / SiFive U74 (scalar `rv64gc`)

U74 OpenBLAS + HPL are done (~1.69×). No RVV/IME on this chip, so the next useful
work is software that **does not vectorize anyway** — prove the EESSI/EasyBuild
path and correctness/perf on a real scientific workload that is irregular /
branchy, not BLAS-bound.

- [ ] **BWA** (`BWA-0.7.19-GCCcore-14.3.0.eb` preferred; already on RV2 inventory) —
  short-read alignment. Classic non-SIMD scientific code (string matching,
  branching, irregular memory). Build via EESSI-extend on VF2
  (`ubuntu@192.168.1.219`), run a small reference alignment, record wall time +
  correctness vs a known-good host/x86 or prior RV2 run. *Primary U74 next item
  when the board is back online.*
- [ ] **BCFtools** / **BamTools** (companions) — same bio column; cheaper smoke
  tests once BWA is up.
- [ ] Optional later: **MODFLOW** / **PETSc** KSP — sparse-solve dominated HPC
  apps (also weakly vectorizing); only after the bio path is proven.

Access notes: see `riscv-learnings` `docs/riscv-u74.md` (`ubuntu@192.168.1.219`).

---

## Recommended order

1. **onnx int4** (Part A) — dramatic, reproducible, extends shipped IME work.
2. **U74 — BWA** (Part D) — when VisionFive 2 is online; non-vector EasyBuild
   app with real scientific value (post–OpenBLAS/HPL).
3. **OSU-Micro-Benchmarks** (B3) — fast interconnect baseline that every MPI app
   result depends on; establish it before scalapack/qe/petsc.
4. **FlexiBLAS RVV OpenBLAS A/B column** — one backend swap validates
   numpy → hpl → elpa → scalapack → scikit-learn → R → Armadillo at once
   (the `_fixed` lib carries the `gemv_n` NaN + TRSM VLEN corrections).
5. **ESPResSo** (B1) — soft-matter MD / P3M FFT A/B (LAMMPS build+compute on RV2
   done; remaining = repo benchmark dir / Force RVV A/B if pursued).
6. **PETSc/SLEPc + MUMPS + SuperLU_DIST** (B2) — the sparse-solver column,
   complementary to the dense eigen probes.
7. **ScaFaCoS** (B2) — long-range Coulomb / FMM companion; waLBerla RVV/auto-vec
   campaign is done ([`walberla/`](walberla): HeatEq/UG wins; hand simd lesson).
8. **QuantumESPRESSO build** (C2) — bump `7.4-foss-2024a` → 2025b and build;
   this is the single item that unblocks the already-present `qe/` benchmark.
9. **CP2K + OpenFOAM + WRF** (C2) — flagship DFT / CFD / NWP whole-app probes;
   each a toolchain rebump, high impact, tackle after the QE build proves the
   easyconfig-bump workflow on the X60.

## Part D — GPU compute: hardware landscape (RVV boards vs Imagination BXM)

Context for a future GPU-compute benchmark column (Vulkan/OpenCL). The RV2's
GPU is **entry-tier**, so any GPU-offload probe wants a higher-tier board.

- **Orange Pi RV2 (SpaceMiT K1/X60)** — GPU **Imagination PowerVR BXE-2-32**
  (BXE = *entry* tier). Fine for UI, weak for compute offload.
- Imagination B/C-series tier order: **BXE (entry) < BXM (mid) < BXT/AXT/CXT
  (premium)**. Only **BXM-4-64** variants ship in SBCs today — no board with a
  larger BXM (BXM-8-256) yet.

RISC-V SBCs that ship a **BXM** GPU (both purchasable now):

| Board | SoC | GPU | Cores | RVV | Note |
|---|---|---|---|---|---|
| **Milk-V Jupiter2** | SpacemiT K3 | BXM-4-64 MC1 | 8× X100 @2.4GHz | **RVV 1.0 / RVA23** | best RISC-V ecosystem fit; both BXM GPU *and* standard vector |
| **Lichee Pi 4A** / BeagleV Ahead | T-Head TH1520 | BXM-4-64 MC1 | 4× C910 @1.5GHz | **RVV 0.7.1 (draft)** | GPU upgrade but non-standard vector — see caveat |

- **C910 vector caveat:** TH1520's C910 implements **RVV 0.7.1 (draft)**, NOT
  ratified RVV 1.0 — binary-incompatible, needs T-Head's `v0p7` toolchain fork.
  A step back from the RV2's RVV 1.0 for our vectorized HPC apps. → **For a
  GPU-compute bench that must ALSO keep RVV 1.0 parity with the RV2 column,
  Jupiter2 (K3) is the only good option.**
- ARM aside (not RISC-V): Allwinner A733 (Radxa Cubie A7S/A7A) also has BXM-4-64.
- Higher tiers: Sophgo SG2380 (RISC-V) uses IMG **AXT-16-512** (premium, not BXM);
  no shipping RISC-V board uses BXT/CXT.
- **Not BXM (ruled out):** StarFive JH7110 = BXE-4-32; ESWIN EIC7700 / Ky X1 /
  Sophgo SG2044 = no confirmed BXM.
- Sources: [Phoronix TH1520 BXM-4-64 fw](https://www.phoronix.com/news/IMG-PowerVR-BXM-4-64-FW),
  [Milk-V Jupiter2](https://milkv.io/jupiter2),
  [SpacemiT K3 datasheet](https://github.com/spacemit-com/docs-chip/blob/main/en/key_stone/k3/k3_docs/k3_ds.md),
  [Imagination B-Series](https://www.imaginationtech.com/products/gpu/img-b-series-gpu/).
  *(ESWIN/Ky X1/SG2044 = "not confirmed", search hit an API limit — not "confirmed absent".)*

---

## Notes on toolchain choice

- Prefer the **`dev.eessi.io/riscv 2025.06-001` (foss-2025b)** modules where an
  app exists there (GROMACS, ESPResSo, HPL, PLUMED, OSU, MUMPS, ScaLAPACK,
  ScaFaCoS, waLBerla, ELPA) — matches the repo's existing build environment.
- Apps that exist **only** in `riscv.eessi.io 20240402 (foss-2023b)** —
  MetalWalls, MODFLOW, PETSc, SLEPc, SuperLU_DIST, SuiteSparse, scikit-learn, R,
  Armadillo, VTK/ParaView — run against the 2023b stack; note the toolchain
  difference in each benchmark's README so results aren't cross-compared naively.
  (**LAMMPS** was moved off this list — it now has a working **foss-2025b** build
  path via a custom easyconfig against `dev.eessi.io/riscv 2025.06-001`; see
  [`lammps/`](lammps/).)
- **QuantumESPRESSO has no RISC-V module in either repo** — the `qe/` benchmark
  needs a from-easyconfig build before it can run end-to-end.

---

## LAMMPS page — DONE → see [`lammps/`](lammps/) (PR #31)

The `lammps/` benchmark page now exists: `lammps/README.md` +
`run-lammps-bench.sh`, opened as PR #31 to `opensolvers/benchmarks`. It carries
the full 5-workload x 3-mode RVV-Kokkos matrix (32000 atoms / 100 steps; lj/eam
best on Kokkos at 6.2x/7.2x, chain/chute/rhodo best on MPI), the RVV-verification
evidence (`Tag_RISCV_arch` v1p0/zve64d, ~53k `vsetvli`), and all five
RISC-V-specific easyconfig fixes (`kokkos_arch=EASYBUILD_GENERIC`; CVMFS
readlinkat-storm allowlist; pinned binutils; `-DUSE_SPGLIB=OFF`; `MDI` removed)
plus the ctest-gate caveat — so the build recipe is documented in the page rather
than here. The custom foss-2025b build (LAMMPS 22Jul2025-u4, Kokkos 4.6.2, GCC
14.3.0, OpenMPI 5.0.8) can be swapped for a plain `module load` once the
`dev.eessi.io/riscv` easyconfig lands upstream. Full build resume state remains in
`LAMMPS_RV2_STATUS.md` in the riscv workspace.
