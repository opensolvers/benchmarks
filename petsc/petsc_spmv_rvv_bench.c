/*
 * petsc_spmv_rvv_bench.c - AIJ SpMV: PETSc MatMult vs CSR scalar/RVV vs stencil RVV
 *
 * Assembles a 2D 5-point Laplacian with PETSc, extracts SeqAIJ CSR, then times:
 *   1) PETSc MatMult
 *   2) scalar CSR SpMV
 *   3) hand RVV CSR SpMV (gather via vluxei64)
 *   4) structured 5-point stencil RVV (contiguous loads; same operator)
 *
 * Short CSR rows (~5 nnz) often limit (3); (4) shows what a structure-aware
 * PETSc kernel could do for this common PDE pattern.
 *
 * Build (needs -march with V):
 *   make petsc_spmv_rvv_bench
 *
 * Run:
 *   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./petsc_spmv_rvv_bench [n] [reps]
 *   defaults: n=800 (640k dofs), reps=20
 *
 * SPDX-License-Identifier: MIT
 */
#include <petscksp.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#define HAVE_RVV 1
#else
#define HAVE_RVV 0
#endif

static double wall_sec(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static PetscErrorCode AssembleLaplacian2D(Mat A, PetscInt n)
{
  PetscFunctionBeginUser;
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n * n, n * n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 5, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 5, NULL, 5, NULL));
  const PetscReal h2 = 1.0 / ((PetscReal)(n + 1) * (PetscReal)(n + 1));
  for (PetscInt i = 0; i < n; ++i) {
    for (PetscInt j = 0; j < n; ++j) {
      const PetscInt row = i * n + j;
      PetscInt       cols[5];
      PetscScalar    vals[5];
      PetscInt       nnz = 0;
      cols[nnz] = row;
      vals[nnz] = 4.0 / h2;
      ++nnz;
      if (j > 0) {
        cols[nnz] = row - 1;
        vals[nnz] = -1.0 / h2;
        ++nnz;
      }
      if (j < n - 1) {
        cols[nnz] = row + 1;
        vals[nnz] = -1.0 / h2;
        ++nnz;
      }
      if (i > 0) {
        cols[nnz] = row - n;
        vals[nnz] = -1.0 / h2;
        ++nnz;
      }
      if (i < n - 1) {
        cols[nnz] = row + n;
        vals[nnz] = -1.0 / h2;
        ++nnz;
      }
      PetscCall(MatSetValues(A, 1, &row, nnz, cols, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void spmv_csr_scalar(const PetscInt *ai, const PetscInt *aj, const PetscScalar *aa,
                            const PetscScalar *x, PetscScalar *y, PetscInt nrows)
{
  for (PetscInt i = 0; i < nrows; ++i) {
    PetscScalar sum = 0.0;
    for (PetscInt k = ai[i]; k < ai[i + 1]; ++k) sum += aa[k] * x[aj[k]];
    y[i] = sum;
  }
}

#if HAVE_RVV
/* Row-wise CSR SpMV with RVV gather. Best when rows are long; 5-pt stencil is short. */
static void spmv_csr_rvv(const PetscInt *ai, const PetscInt *aj, const PetscScalar *aa,
                         const PetscScalar *x, PetscScalar *y, PetscInt nrows)
{
  for (PetscInt i = 0; i < nrows; ++i) {
    PetscInt           k   = ai[i];
    const PetscInt     end = ai[i + 1];
    vfloat64m1_t       vsum = __riscv_vfmv_v_f_f64m1(0.0, 1);
    while (k < end) {
      const size_t     vl = __riscv_vsetvl_e64m2((size_t)(end - k));
      vfloat64m2_t     va = __riscv_vle64_v_f64m2((const double *)&aa[k], vl);
      /* Build byte offsets for gather: aj[k] * sizeof(double). */
      uint64_t         offs[16]; /* VL max for m2 e64 @ VLEN=1024 would be more; 16 covers VLEN=512 */
      const size_t     ncap = sizeof(offs) / sizeof(offs[0]);
      size_t           use  = vl > ncap ? ncap : vl;
      /* If VL somehow > 16, fall back chunk-wise via repeated setvl (loop handles). */
      for (size_t t = 0; t < use; ++t) offs[t] = (uint64_t)aj[k + (PetscInt)t] * sizeof(double);
      vuint64m2_t      vidx = __riscv_vle64_v_u64m2(offs, use);
      vfloat64m2_t     vx   = __riscv_vluxei64_v_f64m2((const double *)x, vidx, use);
      vfloat64m2_t     vp   = __riscv_vfmul_vv_f64m2(va, vx, use);
      vfloat64m1_t     vred = __riscv_vfmv_v_f_f64m1(0.0, 1);
      vred                  = __riscv_vfredusum_vs_f64m2_f64m1(vp, vred, use);
      double           partial = __riscv_vfmv_f_s_f64m1_f64(vred);
      vsum                  = __riscv_vfadd_vf_f64m1(vsum, partial, 1);
      k += (PetscInt)use;
    }
    y[i] = __riscv_vfmv_f_s_f64m1_f64(vsum);
  }
}

/* Structured 5-point stencil matching AssembleLaplacian2D (Dirichlet grid).
 * ih2 must be (n+1)^2 (== 1/h2 used at assembly). */
static void spmv_stencil5_rvv(PetscInt n, PetscReal ih2, const PetscScalar *x, PetscScalar *y)
{
  const PetscScalar c0 = 4.0 * ih2;
  const PetscScalar co = -1.0 * ih2;
  for (PetscInt i = 0; i < n; ++i) {
    PetscInt j = 0;
    while (j < n) {
      const size_t vl = __riscv_vsetvl_e64m2((size_t)(n - j));
      const PetscInt row0 = i * n + j;
      vfloat64m2_t vc = __riscv_vfmul_vf_f64m2(__riscv_vle64_v_f64m2((const double *)&x[row0], vl), (double)c0, vl);

      /* left neighbor: x[row0-1] for j>0; boundary contributes 0 (Dirichlet) */
      {
        double left_buf[16];
        size_t use = vl > 16 ? 16 : vl;
        for (size_t t = 0; t < use; ++t) {
          const PetscInt jj = j + (PetscInt)t;
          left_buf[t] = (jj > 0) ? (double)x[row0 + (PetscInt)t - 1] : 0.0;
        }
        vfloat64m2_t vlft = __riscv_vle64_v_f64m2(left_buf, use);
        vc = __riscv_vfmacc_vf_f64m2(vc, (double)co, vlft, use);
      }
      /* right */
      {
        double right_buf[16];
        size_t use = vl > 16 ? 16 : vl;
        for (size_t t = 0; t < use; ++t) {
          const PetscInt jj = j + (PetscInt)t;
          right_buf[t] = (jj + 1 < n) ? (double)x[row0 + (PetscInt)t + 1] : 0.0;
        }
        vfloat64m2_t vrt = __riscv_vle64_v_f64m2(right_buf, use);
        vc = __riscv_vfmacc_vf_f64m2(vc, (double)co, vrt, use);
      }
      /* up / down: contiguous when present */
      if (i > 0) {
        vfloat64m2_t vup = __riscv_vle64_v_f64m2((const double *)&x[row0 - n], vl);
        vc = __riscv_vfmacc_vf_f64m2(vc, (double)co, vup, vl);
      }
      if (i + 1 < n) {
        vfloat64m2_t vdn = __riscv_vle64_v_f64m2((const double *)&x[row0 + n], vl);
        vc = __riscv_vfmacc_vf_f64m2(vc, (double)co, vdn, vl);
      }
      __riscv_vse64_v_f64m2((double *)&y[row0], vc, vl);
      j += (PetscInt)vl;
    }
  }
}

static void spmv_stencil5_scalar(PetscInt n, PetscReal ih2, const PetscScalar *x, PetscScalar *y)
{
  const PetscScalar c0 = 4.0 * ih2;
  const PetscScalar co = -1.0 * ih2;
  for (PetscInt i = 0; i < n; ++i) {
    for (PetscInt j = 0; j < n; ++j) {
      const PetscInt row = i * n + j;
      PetscScalar    s   = c0 * x[row];
      if (j > 0) s += co * x[row - 1];
      if (j + 1 < n) s += co * x[row + 1];
      if (i > 0) s += co * x[row - n];
      if (i + 1 < n) s += co * x[row + n];
      y[row] = s;
    }
  }
}
#endif /* HAVE_RVV */

static int check_close(const PetscScalar *a, const PetscScalar *b, PetscInt n, double tol, double *max_abs)
{
  double m = 0.0;
  int    ok = 1;
  for (PetscInt i = 0; i < n; ++i) {
    if (!isfinite((double)a[i]) || !isfinite((double)b[i])) {
      ok = 0;
      m = INFINITY;
      break;
    }
    const double d = fabs((double)a[i] - (double)b[i]);
    if (d > m) m = d;
  }
  *max_abs = m;
  if (m > tol) ok = 0;
  return ok;
}

int main(int argc, char **argv)
{
  PetscInt n = 800, reps = 20;
  Mat      A;
  Vec      x, y;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  if (argc > 1 && argv[1][0] != '-') n = (PetscInt)atoi(argv[1]);
  if (argc > 2 && argv[2][0] != '-') reps = (PetscInt)atoi(argv[2]);
  if (n < 2 || reps < 1) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "usage: %s [n] [reps]\n", argv[0]));
    PetscCall(PetscFinalize());
    return 2;
  }

  const PetscInt  dofs = n * n;
  /* AssembleLaplacian2D uses h2=1/(n+1)^2 and coeffs ±1/h2, 4/h2. */
  const PetscReal ih2  = (PetscReal)(n + 1) * (PetscReal)(n + 1);

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(AssembleLaplacian2D(A, n));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecSet(x, 1.0));

  /* Extract SeqAIJ CSR (single-rank). */
  const PetscInt    *ai = NULL, *aj = NULL;
  PetscInt           ncols = 0;
  PetscBool          done = PETSC_FALSE;
  PetscScalar       *aa = NULL;
  PetscCall(MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &ncols, &ai, &aj, &done));
  if (!done || ncols != dofs) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatGetRowIJ failed (done=%d ncols=%" PetscInt_FMT ")\n", (int)done, ncols));
    PetscCall(PetscFinalize());
    return 1;
  }
  PetscCall(MatSeqAIJGetArray(A, &aa));

  PetscScalar *xarr = NULL, *y_petsc = NULL, *y_ref = NULL, *y_tmp = NULL;
  PetscCall(VecGetArray(x, &xarr));
  PetscCall(PetscMalloc1(dofs, &y_petsc));
  PetscCall(PetscMalloc1(dofs, &y_ref));
  PetscCall(PetscMalloc1(dofs, &y_tmp));

  /* Reference: PETSc MatMult into y, copy out. */
  PetscCall(MatMult(A, x, y));
  {
    const PetscScalar *ya;
    PetscCall(VecGetArrayRead(y, &ya));
    PetscCall(PetscArraycpy(y_petsc, ya, dofs));
    PetscCall(VecRestoreArrayRead(y, &ya));
  }

  spmv_csr_scalar(ai, aj, aa, xarr, y_ref, dofs);
  double max_abs = 0.0;
  int    ok_csr  = check_close(y_petsc, y_ref, dofs, 1e-9 * (double)n, &max_abs);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "check CSR scalar vs PETSc: ok=%d max_abs=%.3e  HAVE_RVV=%d\n",
                        ok_csr, max_abs, HAVE_RVV));

  /* Timing helpers via macros */
#define TIME_LOOP(label, code)                                                                 \
  do {                                                                                         \
    code; /* warmup */                                                                         \
    double best = 1e300;                                                                       \
    for (PetscInt r = 0; r < reps; ++r) {                                                      \
      const double t0 = wall_sec();                                                            \
      code;                                                                                    \
      const double dt = wall_sec() - t0;                                                       \
      if (dt < best) best = dt;                                                                \
    }                                                                                          \
    const double gflops = (2.0 * (double)ai[dofs] /* nnz */) / best / 1e9;                     \
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,                                                    \
                          "BEST %-18s n=%" PetscInt_FMT " dofs=%" PetscInt_FMT ": %.6f s  ~%.3f GF/s (2*nnz/t)\n", \
                          label, n, dofs, best, gflops));                                      \
  } while (0)

  TIME_LOOP("PETSc MatMult", {
    PetscCall(MatMult(A, x, y));
  });

  TIME_LOOP("CSR scalar", { spmv_csr_scalar(ai, aj, aa, xarr, y_tmp, dofs); });

#if HAVE_RVV
  spmv_csr_rvv(ai, aj, aa, xarr, y_tmp, dofs);
  ok_csr = check_close(y_petsc, y_tmp, dofs, 1e-8 * (double)n, &max_abs);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "check CSR RVV vs PETSc: ok=%d max_abs=%.3e\n", ok_csr, max_abs));

  TIME_LOOP("CSR RVV", { spmv_csr_rvv(ai, aj, aa, xarr, y_tmp, dofs); });

  spmv_stencil5_scalar(n, ih2, xarr, y_tmp);
  ok_csr = check_close(y_petsc, y_tmp, dofs, 1e-8 * (double)n, &max_abs);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "check stencil5 scalar vs PETSc: ok=%d max_abs=%.3e\n", ok_csr, max_abs));
  TIME_LOOP("stencil5 scalar", { spmv_stencil5_scalar(n, ih2, xarr, y_tmp); });

  spmv_stencil5_rvv(n, ih2, xarr, y_tmp);
  ok_csr = check_close(y_petsc, y_tmp, dofs, 1e-8 * (double)n, &max_abs);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "check stencil5 RVV vs PETSc: ok=%d max_abs=%.3e\n", ok_csr, max_abs));
  TIME_LOOP("stencil5 RVV", { spmv_stencil5_rvv(n, ih2, xarr, y_tmp); });
#else
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "RVV not enabled at compile time — rebuild with -march=rv64gcv\n"));
#endif

  PetscCall(MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &ncols, &ai, &aj, &done));
  PetscCall(MatSeqAIJRestoreArray(A, &aa));
  PetscCall(VecRestoreArray(x, &xarr));
  PetscCall(PetscFree(y_petsc));
  PetscCall(PetscFree(y_ref));
  PetscCall(PetscFree(y_tmp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
