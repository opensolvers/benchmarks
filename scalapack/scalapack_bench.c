/*
 * scalapack_bench.c - distributed ScaLAPACK pdsyev eigensolver benchmark
 *
 * Times a dense real-symmetric eigenproblem solved by ScaLAPACK PDSYEV on a
 * 2D block-cyclic grid, as a pure-MPI proxy for the performance and numerical
 * correctness of the (node-local) BLAS backend. Companion to elpa_bench.c:
 * same problem, different library / parallelism model (ScaLAPACK = many MPI
 * ranks x 1 thread; ELPA driver here = 1 rank x threaded BLAS).
 *
 * Each rank fills only its local block-cyclic entries of a deterministic
 * symmetric, diagonally-dominant matrix (global index math -> no scatter).
 *
 * Build (with a ScaLAPACK module loaded):  make -f Makefile.scalapack
 * Run:   OMP_NUM_THREADS=1 mpirun --bind-to core -np 8 ./scalapack_bench [na] [nb] [nprow]
 *        defaults: na=3000 nb=64 grid auto-factored near-square from np
 * A/B a BLAS backend via FlexiBLAS (scalar vs vector) exactly as for elpa_bench.
 * Exits nonzero if an eigenvalue is non-finite (CI correctness gate).
 * SPDX-License-Identifier: MIT
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void Cblacs_pinfo(int*, int*);
extern void Cblacs_get(int, int, int*);
extern void Cblacs_gridinit(int*, const char*, int, int);
extern void Cblacs_gridinfo(int, int*, int*, int*, int*);
extern void Cblacs_gridexit(int);
extern int  numroc_(int*, int*, int*, int*, int*);
extern void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
extern void pdsyev_(const char*, const char*, int*, double*, int*, int*, int*,
                    double*, double*, int*, int*, int*, double*, int*, int*);

static double sym_val(long gi, long gj){
  long a = gi<gj?gi:gj, b = gi<gj?gj:gi;
  unsigned long h = (unsigned long)a*2654435761UL + (unsigned long)b*40503UL;
  return (double)(h % 100000UL) / 100000.0;   /* symmetric in (i,j), in [0,1) */
}

int main(int argc, char** argv){
  MPI_Init(&argc,&argv);
  int myid, nprocs;
  Cblacs_pinfo(&myid,&nprocs);

  int na = argc>1 ? atoi(argv[1]) : 3000;
  int nb = argc>2 ? atoi(argv[2]) : 64;
  int nprow = (int)floor(sqrt((double)nprocs));
  while(nprow>1 && nprocs%nprow) nprow--;
  int npcol = nprocs/nprow;
  if(argc>3){ nprow=atoi(argv[3]); npcol=nprocs/nprow; }

  int ctxt; Cblacs_get(0,0,&ctxt);
  Cblacs_gridinit(&ctxt,"Row-major",nprow,npcol);
  int prow,pcol; Cblacs_gridinfo(ctxt,&nprow,&npcol,&prow,&pcol);
  if(prow<0||pcol<0){ fprintf(stderr,"rank %d not in %dx%d grid\n",myid,nprow,npcol); MPI_Abort(MPI_COMM_WORLD,2); }

  int izero=0, ione=1;
  int nrows = numroc_(&na,&nb,&prow,&izero,&nprow);
  int ncols = numroc_(&na,&nb,&pcol,&izero,&npcol);
  int lld = nrows>1?nrows:1;

  double *A = malloc((size_t)lld*ncols*sizeof(double));
  double *Z = malloc((size_t)lld*ncols*sizeof(double));
  double *W = malloc((size_t)na*sizeof(double));
  if(!A||!Z||!W){ fprintf(stderr,"alloc fail\n"); MPI_Abort(MPI_COMM_WORLD,2); }

  for(int lj=0; lj<ncols; lj++){
    long gj = ((long)(lj/nb)*npcol + pcol)*nb + (lj%nb);
    for(int li=0; li<nrows; li++){
      long gi = ((long)(li/nb)*nprow + prow)*nb + (li%nb);
      double v = sym_val(gi,gj);
      if(gi==gj) v += (double)na;    /* diagonally dominant -> well-conditioned */
      A[li + (size_t)lj*lld] = v;
    }
  }

  int desca[9], descz[9], info;
  descinit_(desca,&na,&na,&nb,&nb,&izero,&izero,&ctxt,&lld,&info);
  descinit_(descz,&na,&na,&nb,&nb,&izero,&izero,&ctxt,&lld,&info);

  double wkopt; int lwork=-1;                 /* workspace query */
  pdsyev_("V","U",&na,A,&ione,&ione,desca,W,Z,&ione,&ione,descz,&wkopt,&lwork,&info);
  lwork=(int)wkopt + 1;
  double *work = malloc((size_t)lwork*sizeof(double));
  if(!work){ fprintf(stderr,"work alloc fail (lwork=%d)\n",lwork); MPI_Abort(MPI_COMM_WORLD,2); }

  MPI_Barrier(MPI_COMM_WORLD);
  double t0=MPI_Wtime();
  pdsyev_("V","U",&na,A,&ione,&ione,desca,W,Z,&ione,&ione,descz,work,&lwork,&info);
  MPI_Barrier(MPI_COMM_WORLD);
  double dt=MPI_Wtime()-t0;

  int finite = isfinite(W[0]) && isfinite(W[na-1]);
  if(myid==0)
    printf("ScaLAPACK pdsyev na=%d nb=%d grid=%dx%d: %.2f s  ev0=%.5f evN=%.5f finite=%d info=%d\n",
           na,nb,nprow,npcol,dt,W[0],W[na-1],finite,info);

  free(work); free(A); free(Z); free(W);
  Cblacs_gridexit(ctxt);
  MPI_Finalize();
  return finite ? 0 : 3;
}
