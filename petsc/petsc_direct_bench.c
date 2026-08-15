/*
 * petsc_direct_bench.c - sparse-direct LU via PETSc (MUMPS / SuperLU_DIST / PETSc)
 *
 * Assembles a 2D/3D Laplacian AIJ matrix and times KSPPREONLY + PCLU with a
 * chosen factor package. Frontal dense BLAS-3 is the FlexiBLAS-sensitive part.
 *
 *   ./petsc_direct_bench [n] [reps] [package] [dim]
 *   package: mumps | superlu_dist | petsc  (default mumps)
 *   dim: 2 | 3   (default 2; 3 => n^3 dofs — keep n small)
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

static PetscErrorCode AssembleLaplacian3D(Mat A, PetscInt n)
{
  PetscFunctionBeginUser;
  const PetscInt N = n * n * n;
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 7, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 7, NULL, 7, NULL));
  const PetscReal h2 = 1.0 / ((PetscReal)(n + 1) * (PetscReal)(n + 1));
  for (PetscInt i = 0; i < n; ++i) {
    for (PetscInt j = 0; j < n; ++j) {
      for (PetscInt k = 0; k < n; ++k) {
        const PetscInt row = (i * n + j) * n + k;
        PetscInt       cols[7];
        PetscScalar    vals[7];
        PetscInt       nnz = 0;
        cols[nnz] = row;
        vals[nnz] = 6.0 / h2;
        ++nnz;
        if (k > 0) {
          cols[nnz] = row - 1;
          vals[nnz] = -1.0 / h2;
          ++nnz;
        }
        if (k < n - 1) {
          cols[nnz] = row + 1;
          vals[nnz] = -1.0 / h2;
          ++nnz;
        }
        if (j > 0) {
          cols[nnz] = row - n;
          vals[nnz] = -1.0 / h2;
          ++nnz;
        }
        if (j < n - 1) {
          cols[nnz] = row + n;
          vals[nnz] = -1.0 / h2;
          ++nnz;
        }
        if (i > 0) {
          cols[nnz] = row - n * n;
          vals[nnz] = -1.0 / h2;
          ++nnz;
        }
        if (i < n - 1) {
          cols[nnz] = row + n * n;
          vals[nnz] = -1.0 / h2;
          ++nnz;
        }
        PetscCall(MatSetValues(A, 1, &row, nnz, cols, vals, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt    n = 120, reps = 3, dim = 2;
  const char *package = "mumps";
  Mat         A;
  Vec         x, b;
  KSP         ksp;
  PC          pc;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  if (argc > 1 && argv[1][0] != '-') n = (PetscInt)atoi(argv[1]);
  if (argc > 2 && argv[2][0] != '-') reps = (PetscInt)atoi(argv[2]);
  if (argc > 3 && argv[3][0] != '-') package = argv[3];
  if (argc > 4 && argv[4][0] != '-') dim = (PetscInt)atoi(argv[4]);
  if (n < 2 || reps < 1 || (dim != 2 && dim != 3)) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "usage: %s [n] [reps] [mumps|superlu_dist|umfpack|petsc] [2|3]\n", argv[0]));
    PetscCall(PetscFinalize());
    return 2;
  }

  const PetscInt dofs = (dim == 2) ? (n * n) : (n * n * n);
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  if (dim == 2) PetscCall(AssembleLaplacian2D(A, n));
  else PetscCall(AssembleLaplacian3D(A, n));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(b, 1.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  if (!strcmp(package, "mumps")) {
    PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
  } else if (!strcmp(package, "superlu_dist")) {
    PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST));
  } else if (!strcmp(package, "umfpack")) {
    PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERUMFPACK));
  } else if (!strcmp(package, "petsc")) {
    PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERPETSC));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "unknown package '%s'\n", package));
    PetscCall(PetscFinalize());
    return 2;
  }
  PetscCall(KSPSetFromOptions(ksp));

  /* Factor once (warmup includes first numeric factor). */
  PetscCall(VecSet(x, 0.0));
  PetscCall(KSPSolve(ksp, b, x));

  double best = 1e300;
  int    all_finite = 1;
  for (PetscInt r = 0; r < reps; ++r) {
    PetscCall(VecSet(x, 0.0));
    /* Destroy factor to force refactor each rep (BLAS-heavy path). */
    PetscCall(KSPSetOperators(ksp, A, A));
    const double t0 = wall_sec();
    PetscCall(KSPSolve(ksp, b, x));
    const double dt = wall_sec() - t0;
    PetscReal nrm = 0.0, rnorm = 0.0;
    KSPConvergedReason reason;
    PetscCall(VecNorm(x, NORM_2, &nrm));
    PetscCall(KSPGetResidualNorm(ksp, &rnorm));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    const int finite = (reason > 0) && isfinite((double)nrm) && isfinite((double)rnorm) ? 1 : 0;
    all_finite &= finite;
    if (dt < best) best = dt;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "PETSc direct %s dim=%" PetscInt_FMT " n=%" PetscInt_FMT " dofs=%" PetscInt_FMT ": %.6f s  |x|=%.6e rnorm=%.6e finite=%d reason=%d\n",
                          package, dim, n, dofs, dt, (double)nrm, (double)rnorm, finite, (int)reason));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "BEST direct %s dim=%" PetscInt_FMT " n=%" PetscInt_FMT " dofs=%" PetscInt_FMT ": %.6f s  finite=%d\n",
                        package, dim, n, dofs, best, all_finite));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return all_finite ? 0 : 1;
}
