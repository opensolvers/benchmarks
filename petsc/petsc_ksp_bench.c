/*
 * petsc_ksp_bench.c - 2D Laplacian KSP micro-benchmark (FlexiBLAS A/B)
 *
 * Purpose
 *   Time a sparse CG solve of a 2D 5-point Poisson problem as a proxy for the
 *   performance AND numerical correctness of the BLAS/LAPACK backend under
 *   PETSc. Sparse matvecs dominate; dense BLAS still appears in orthogonal
 *   reductions / preconditioners. Swapping the FlexiBLAS backend (e.g. stock
 *   RVV OpenBLAS vs scalar vs patched RVV) changes runtime and - if the
 *   backend is buggy - whether the solution / residual stay finite.
 *
 * Design
 *   Single MPI rank by default (threaded BLAS via OMP/OPENBLAS_NUM_THREADS).
 *   Matrix: unit-square Dirichlet Laplacian on an n x n grid (n^2 unknowns),
 *   right-hand side = 1, Jacobi preconditioned CG, relative tolerance 1e-8.
 *
 * Build (PETSc + FlexiBLAS modules loaded):
 *   make
 *
 * Run:
 *   OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ./petsc_ksp_bench [n] [reps]
 *   defaults: n=400 (160k dofs), reps=3
 *
 * A/B via FlexiBLAS / OpenBLAS (no rebuild):
 *   scalar : OPENBLAS_CORETYPE=RISCV64_GENERIC ...
 *   stock  : (default dispatch)
 *   patched: FLEXIBLAS=/path/to/libopenblas.so ...
 *
 * Output (one line per timed solve; final summary line):
 *   PETSc KSP n=.. dofs=..: <sec> s  its=.. rnorm=.. finite=<0|1>
 *
 * SPDX-License-Identifier: MIT
 */
#include <petscksp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

int main(int argc, char **argv)
{
  PetscInt n = 400, reps = 3;
  Mat      A;
  Vec      x, b;
  KSP      ksp;
  PC       pc;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-reps", &reps, NULL));
  if (argc > 1 && argv[1][0] != '-') n = (PetscInt)atoi(argv[1]);
  if (argc > 2 && argv[2][0] != '-') reps = (PetscInt)atoi(argv[2]);
  if (n < 2 || reps < 1) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "usage: %s [n] [reps]\n", argv[0]));
    PetscCall(PetscFinalize());
    return 2;
  }

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(AssembleLaplacian2D(A, n));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(b, 1.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));
  PetscCall(KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));

  /* Warmup (not timed). */
  PetscCall(VecSet(x, 0.0));
  PetscCall(KSPSolve(ksp, b, x));

  double best = 1e300;
  int    best_its = -1;
  double best_rnorm = NAN;
  int    all_finite = 1;
  const PetscInt dofs = n * n;

  for (PetscInt r = 0; r < reps; ++r) {
    PetscCall(VecSet(x, 0.0));
    const double t0 = wall_sec();
    PetscCall(KSPSolve(ksp, b, x));
    const double dt = wall_sec() - t0;

    PetscInt       its = 0;
    PetscReal      rnorm = 0.0;
    PetscReal      nrm = 0.0;
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
      best_rnorm = (double)rnorm;
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "PETSc KSP n=%" PetscInt_FMT " dofs=%" PetscInt_FMT ": %.6f s  its=%" PetscInt_FMT " rnorm=%.6e finite=%d reason=%d\n",
                          n, dofs, dt, its, (double)rnorm, finite, (int)reason));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "BEST n=%" PetscInt_FMT " dofs=%" PetscInt_FMT ": %.6f s  its=%d rnorm=%.6e finite=%d\n",
                        n, dofs, best, best_its, best_rnorm, all_finite));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return all_finite ? 0 : 1;
}
