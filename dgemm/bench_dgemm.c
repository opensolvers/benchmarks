/*
 * bench_dgemm.c - minimal DGEMM throughput benchmark.
 *
 * Times a square double-precision C = A*B (level-3 BLAS dgemm, 3 reps) and
 * reports GFLOP/s. Links against any BLAS that provides the Fortran dgemm_
 * symbol; use FlexiBLAS (or OPENBLAS_CORETYPE / OPENBLAS_NUM_THREADS) to select
 * and A/B the backend without recompiling. Prints C[0] so a NaN/garbage backend
 * is immediately visible.
 *
 * Build:  gcc -O2 bench_dgemm.c -o bench_dgemm -lflexiblas   (or -lopenblas)
 * Run:    OPENBLAS_NUM_THREADS=8 ./bench_dgemm [N]           # default N=4096
 * A/B:    scalar : OPENBLAS_CORETYPE=RISCV64_GENERIC ./bench_dgemm
 *         vector : FLEXIBLAS=/path/to/libopenblas.so  ./bench_dgemm
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern void dgemm_(const char*,const char*,const int*,const int*,const int*,
  const double*,const double*,const int*,const double*,const int*,
  const double*,double*,const int*);
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
  int n=argc>1?atoi(argv[1]):4096;
  double *A=malloc(8L*n*n),*B=malloc(8L*n*n),*C=malloc(8L*n*n);
  if(!A||!B||!C){printf("alloc fail\n");return 1;}
  for(long i=0;i<(long)n*n;i++){A[i]=(i%7)*0.1;B[i]=(i%5)*0.2;C[i]=0;}
  double al=1.0,be=0.0;
  dgemm_("N","N",&n,&n,&n,&al,A,&n,B,&n,&be,C,&n);
  int reps=3; double t0=now();
  for(int r=0;r<reps;r++) dgemm_("N","N",&n,&n,&n,&al,A,&n,B,&n,&be,C,&n);
  double t=now()-t0; double gf=2.0*(double)n*n*n*reps/t/1e9;
  printf("N=%d threads=%s DGEMM=%.2f GFLOP/s (t=%.2fs C[0]=%g)\n",
    n,getenv("OPENBLAS_NUM_THREADS")?getenv("OPENBLAS_NUM_THREADS"):"?",gf,t,C[0]);
  return 0;
}
