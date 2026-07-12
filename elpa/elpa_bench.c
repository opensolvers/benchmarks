/*
 * elpa_bench.c - single-process ELPA eigensolver micro-benchmark
 *
 * Purpose
 *   Time a dense real-symmetric eigenproblem solved by ELPA (1-stage) as a
 *   proxy for the performance AND numerical correctness of the underlying
 *   BLAS/LAPACK backend. ELPA's tridiagonalization and back-transform are
 *   dominated by BLAS-2/BLAS-3 (dsymv / dsyr2k / dgemm / dtrmm), so swapping
 *   the BLAS backend (e.g. via FlexiBLAS) changes both the runtime and - if
 *   the backend is buggy - whether the eigenvalues come back finite.
 *
 *   It was written to validate a RISC-V RVV (SpaceMiT X60 / K1) OpenBLAS
 *   build against a scalar baseline: a broken vector gemv_n kernel makes the
 *   eigenvalues NaN, so finite=0 is a real correctness failure, not merely a
 *   slowdown. The program exits nonzero on a non-finite result so it can be
 *   used as a CI correctness gate.
 *
 * Design
 *   Single MPI rank, trivial 1x1 process grid (na_rows=na_cols=na). This is a
 *   shared-memory benchmark: parallelism comes from the threaded BLAS backend
 *   (OMP_NUM_THREADS / OPENBLAS_NUM_THREADS), not from MPI. ELPA still requires
 *   an MPI communicator, hence MPI_Init + the 1x1 grid.
 *
 * Build (with an EESSI/EasyBuild ELPA module loaded):
 *   make                 # uses $EBROOT{ELPA,SCALAPACK,FLEXIBLAS}
 *
 * Run:
 *   OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
 *     mpirun --bind-to none -np 1 ./elpa_bench [na] [nev] [nblk]
 *   defaults: na=3000  nev=na  nblk=64
 *   (--bind-to none lets the threaded BLAS spread across all cores.)
 *
 * A/B a BLAS backend through FlexiBLAS (no rebuild):
 *   scalar : OPENBLAS_CORETYPE=RISCV64_GENERIC ...
 *   vector : FLEXIBLAS=/path/to/libopenblas.so ...
 *
 * Output (one line):
 *   ELPA 1stage na=.. nev=.. nblk=..: <sec> s  ev0=.. evN=.. finite=<0|1>
 *   lower time = faster backend; finite=0 = backend produced NaN (FAIL).
 *
 * SPDX-License-Identifier: MIT
 */
#include <elpa/elpa.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char** argv){
  int na   = argc>1 ? atoi(argv[1]) : 3000;
  int nev  = argc>2 ? atoi(argv[2]) : na;
  int nblk = argc>3 ? atoi(argv[3]) : 64;

  MPI_Init(&argc,&argv);

  /* Single rank, 1x1 process grid: all parallelism is in the threaded BLAS. */
  int my_prow=0, my_pcol=0, na_rows=na, na_cols=na, err;

  double *a  = malloc((size_t)na_rows*na_cols*sizeof(double));
  double *q  = malloc((size_t)na_rows*na_cols*sizeof(double));
  double *ev = malloc((size_t)na*sizeof(double));
  if(!a||!q||!ev){ fprintf(stderr,"alloc fail (na=%d)\n",na); return 2; }

  /* Deterministic real-symmetric test matrix. */
  srand(12345);
  for(int j=0;j<na;j++)
    for(int i=j;i<na;i++){
      double v=(double)rand()/RAND_MAX;
      a[i+(size_t)j*na_rows]=v;
      a[j+(size_t)i*na_rows]=v;
    }

  if(elpa_init(20211125)!=ELPA_OK){ fprintf(stderr,"elpa_init: API 20211125 unsupported\n"); return 1; }
  elpa_t h = elpa_allocate(&err);
  elpa_set(h,"na",na,&err);               elpa_set(h,"nev",nev,&err);
  elpa_set(h,"local_nrows",na_rows,&err); elpa_set(h,"local_ncols",na_cols,&err);
  elpa_set(h,"nblk",nblk,&err);
  elpa_set(h,"mpi_comm_parent",MPI_Comm_c2f(MPI_COMM_WORLD),&err);
  elpa_set(h,"process_row",my_prow,&err); elpa_set(h,"process_col",my_pcol,&err);
  err=elpa_setup(h); if(err!=ELPA_OK){ fprintf(stderr,"elpa_setup err=%d\n",err); return 1; }
  elpa_set(h,"solver",ELPA_SOLVER_1STAGE,&err);

  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  elpa_eigenvectors(h,a,ev,q,&err);
  clock_gettime(CLOCK_MONOTONIC,&t1);
  if(err!=ELPA_OK) fprintf(stderr,"elpa_eigenvectors err=%d\n",err);

  double dt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
  int finite = isfinite(ev[0]) && isfinite(ev[na-1]);
  printf("ELPA 1stage na=%d nev=%d nblk=%d: %.2f s  ev0=%.5f evN=%.5f finite=%d\n",
         na,nev,nblk,dt,ev[0],ev[na-1],finite);

  elpa_deallocate(h,&err); elpa_uninit(&err);
  free(a); free(q); free(ev);
  MPI_Finalize();
  return finite ? 0 : 3;   /* nonzero exit on NaN -> usable as a CI correctness gate */
}
