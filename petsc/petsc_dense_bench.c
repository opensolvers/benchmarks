/*
 * petsc_dense_bench.c - dense MatMult / dense KSP FlexiBLAS A/B
 *
 * Times MATDENSE MatMult (and optional dense CG) so the BLAS backend is on
 * the hot path (unlike sparse Jacobi-CG).
 *
 *   ./petsc_dense_bench [n] [reps] [mode]
 *   mode: mult (default) | cg
 *
 * SPDX-License-Identifier: MIT
 */
#include <petscksp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_sec(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static PetscErrorCode FillDenseSPD(Mat A, PetscInt n)
{
  PetscScalar *a;
  PetscFunctionBeginUser;
  PetscCall(MatDenseGetArrayWrite(A, &a));
  /* Column-major dense storage. */
  for (PetscInt j = 0; j < n; ++j) {
    for (PetscInt i = 0; i < n; ++i) {
      PetscScalar v = 0.1 * ((PetscScalar)((i * 131 + j * 17) % 1000) / 1000.0);
      if (i == j) v += (PetscScalar)n;
      a[i + j * n] = v;
    }
  }
  PetscCall(MatDenseRestoreArrayWrite(A, &a));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt    n = 2000, reps = 5;
  const char *mode = "mult";
  Mat         A;
  Vec         x, y;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  if (argc > 1 && argv[1][0] != '-') n = (PetscInt)atoi(argv[1]);
  if (argc > 2 && argv[2][0] != '-') reps = (PetscInt)atoi(argv[2]);
  if (argc > 3 && argv[3][0] != '-') mode = argv[3];
  if (n < 2 || reps < 1) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "usage: %s [n] [reps] [mult|cg]\n", argv[0]));
    PetscCall(PetscFinalize());
    return 2;
  }

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(FillDenseSPD(A, n));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecSet(x, 1.0));

  if (!strcmp(mode, "mult")) {
    /* Warmup */
    PetscCall(MatMult(A, x, y));
    double best = 1e300;
    int    all_finite = 1;
    for (PetscInt r = 0; r < reps; ++r) {
      const double t0 = wall_sec();
      PetscCall(MatMult(A, x, y));
      const double dt = wall_sec() - t0;
      PetscReal nrm = 0.0;
      PetscCall(VecNorm(y, NORM_2, &nrm));
      const int finite = isfinite((double)nrm) ? 1 : 0;
      all_finite &= finite;
      if (dt < best) best = dt;
      const double gflops = (2.0 * (double)n * (double)n) / dt / 1e9;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "PETSc dense MatMult n=%" PetscInt_FMT ": %.6f s  %.3f GF/s  |y|=%.6e finite=%d\n",
                            n, dt, gflops, (double)nrm, finite));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "BEST dense MatMult n=%" PetscInt_FMT ": %.6f s  %.3f GF/s  finite=%d\n",
                          n, best, (2.0 * (double)n * (double)n) / best / 1e9, all_finite));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
    return all_finite ? 0 : 1;
  }

  /* mode == cg: dense CG, no PC (or none) */
  {
    Vec  b;
    KSP  ksp;
    PC   pc;
    PetscCall(VecDuplicate(x, &b));
    PetscCall(VecSet(b, 1.0));
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCNONE));
    PetscCall(KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, 200));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(VecSet(x, 0.0));
    PetscCall(KSPSolve(ksp, b, x)); /* warmup */

    double best = 1e300;
    int    best_its = -1;
    int    all_finite = 1;
    for (PetscInt r = 0; r < reps; ++r) {
      PetscCall(VecSet(x, 0.0));
      const double t0 = wall_sec();
      PetscCall(KSPSolve(ksp, b, x));
      const double dt = wall_sec() - t0;
      PetscInt its = 0;
      PetscReal rnorm = 0.0, nrm = 0.0;
      KSPConvergedReason reason;
      PetscCall(KSPGetIterationNumber(ksp, &its));
      PetscCall(KSPGetResidualNorm(ksp, &rnorm));
      PetscCall(KSPGetConvergedReason(ksp, &reason));
      PetscCall(VecNorm(x, NORM_2, &nrm));
      const int finite = (reason > 0) && isfinite((double)rnorm) && isfinite((double)nrm) ? 1 : 0;
      all_finite &= finite;
      if (dt < best) {
        best = dt;
        best_its = (int)its;
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "PETSc dense CG n=%" PetscInt_FMT ": %.6f s  its=%" PetscInt_FMT " rnorm=%.6e finite=%d reason=%d\n",
                            n, dt, its, (double)rnorm, finite, (int)reason));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "BEST dense CG n=%" PetscInt_FMT ": %.6f s  its=%d finite=%d\n",
                          n, best, best_its, all_finite));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
    return all_finite ? 0 : 1;
  }
}
