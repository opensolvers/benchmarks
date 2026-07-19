# RV2 verification TODO — apps ready to A/B on the SpaceMiT X60

Tracking list of applications in this repo that can be verified end-to-end on the
Orange Pi RV2 (SpaceMiT X60, `rv64gcv`, RVV 1.0, VLEN=256, 8× @ 1.6 GHz) under
EESSI `2025.06-001`. Each benchmark is a single-variable A/B probing one
underlying library; "ready" means that library's module / custom lib / built
artifact is already present on the board.

## Ready to verify now (lib + artifacts confirmed on board)

- [ ] **onnx** — int4 `MatMulNBits` on ONNX Runtime. Probes the X60 IME
  `smt.vmadot` core (same core as `ime/`). Reproduce the ~9–10× `accuracy_level=4`
  CompInt8-vs-CompFp32 result + correctness gate. *Lib:* custom
  `ONNX-Runtime/1.29.0-foss-2025b-xsmtvdot` (installed). **Highest priority —
  directly extends the IME scale-build PR.**
- [ ] **ime-bench** — s8s8s32 int8 GEMM microkernel, bit-exact vs RVV baseline.
  *Artifact:* `~/ime-bench/` built.
- [ ] **BLIS** — dgemm / trsm, BLAS-3 vs OpenBLAS (link A/B, no FlexiBLAS).
  *Lib:* `BLIS/2.0-GCC-14.3.0`; `~/blis-bench/` built.
- [ ] **OpenBLAS / dgemm** — dgemm GFLOP/s + `gemv_n` NaN / TRSM correctness.
  *Lib:* FlexiBLAS live (OPENBLAS/BLIS/NETLIB) + custom RVV libs
  `libopenblas_x60_eb_fixed.so`, `libopenblas_tuned.so`.
- [ ] **numpy** — `A@B` (dgemm) + `eigvalsh` (dsyevd) finite-gate.
  *Lib:* `SciPy-bundle/2025.07-gfbf-2025b` (FlexiBLAS→OpenBLAS).
- [ ] **hpl** — Linpack end-to-end, FlexiBLAS backend swap.
  *Lib:* `HPL/2.3-foss-2025b`; `~/hpl-blis/` configs.
- [ ] **elpa** — dense real-symmetric eigensolver, finite-gate.
  *Lib:* `ELPA/2025.06.002-foss-2025b`.
- [ ] **scalapack** — `pdsyev`, pure-MPI 8-rank grid.
  *Lib:* `ScaLAPACK/2.2.2-gompi-2025b-fb`.
- [ ] **fftw** — r5v (RVV) vs scalar, FFT MFLOPS. *Artifact:* A/B pair already
  built (`fftwbuild/src-r5v` + `src-scalar` `.so`s).
- [ ] **gromacs** — MD FFT A/B, RVV-float backend. *Lib:* `GROMACS/2026.2-foss-2025b`;
  `~/gmx-bench/`.

## Ready pending a module check

- [ ] **qe** — Quantum ESPRESSO `pw.x` DFT SCF, FlexiBLAS backend swap.
  `~/qe-bench/` inputs + prior FFT-correctness logs present, but confirm a
  `QuantumESPRESSO/7.5-foss-2025b` module is loadable on the board first.

## Recommended order

1. **onnx int4** — dramatic (~9–10×), reproducible, and extends the shipped
   `ime/` IME work through a real ORT LLM-FFN stack.
2. **FlexiBLAS RVV OpenBLAS A/B** — one backend swap validates a whole column
   (numpy / hpl / elpa / scalapack / qe) at once; the `_fixed` lib carries the
   `gemv_n` NaN + TRSM VLEN corrections.
