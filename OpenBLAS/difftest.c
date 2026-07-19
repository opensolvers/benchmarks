/*
 * difftest.c - differential BLAS correctness test (single-threaded).
 *
 * dlopen()s a given BLAS .so and calls a spread of level-1/2/3 routines on
 * fixed pseudo-random inputs, printing sum / sumsq / NaN-count / Inf-count for
 * each. Point it at two backends (or one backend under two OPENBLAS_CORETYPE
 * settings) and diff the output to localize a numerically broken kernel.
 *
 * This is the test that isolated the OpenBLAS 0.3.30 RISC-V vector gemv_n bug:
 * on a SpaceMiT X60 the stock DYNAMIC_ARCH build's RVV (ZVL256B) cblas_dgemv
 * returns NaN (VFILL_ZERO zeroes an uninitialized vector register) while
 * dgemm/dtrsm are correct - which is exactly why HPL (leans on dgemv in panel
 * factorization) NaNs while a standalone dgemm benchmark looks fine. Forcing
 * OPENBLAS_CORETYPE=RISCV64_GENERIC, or using a fixed/patched RVV build, makes
 * dgemv match the scalar reference bit-for-bit (nan=0).
 *
 * Build:  gcc -O2 difftest.c -o difftest -ldl -lm
 * Run:    ./difftest /path/to/libopenblas.so
 *         OPENBLAS_CORETYPE=RISCV64_GENERIC ./difftest /path/to/libopenblas.so
 * Requires a backend exposing the cblas_* symbols (OpenBLAS does).
 * SPDX-License-Identifier: MIT
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

enum {ColMajor=102,RowMajor=101,NoTrans=111,Trans=112,Upper=121,Lower=122,
      Left=141,Right=142,NonUnit=131,Unit=132};

typedef void   (*dgemm_t)(int,int,int,int,int,int,double,const double*,int,const double*,int,double,double*,int);
typedef void   (*dtrsm_t)(int,int,int,int,int,int,int,double,const double*,int,double*,int);
typedef void   (*dgemv_t)(int,int,int,int,double,const double*,int,const double*,int,double,double*,int);
typedef void   (*dscal_t)(int,double,double*,int);
typedef void   (*daxpy_t)(int,double,const double*,int,double*,int);
typedef size_t (*idamax_t)(int,const double*,int);
typedef double (*dnrm2_t)(int,const double*,int);
typedef double (*ddot_t)(int,const double*,int,const double*,int);
typedef void   (*setnt_t)(int);

static double rnd(long i){
  unsigned long x=(unsigned long)i*2654435761UL+1013904223UL;
  x^=x>>13; x*=0xD1B54A32D192ED03UL; x^=x>>29;
  return ((double)(x>>40)/(double)(1UL<<24))*2.0-1.0; /* [-1,1) */
}

static void report(const char*name,const double*v,long n){
  double s=0,s2=0; long nan=0,inf=0;
  for(long i=0;i<n;i++){double x=v[i];
    if(isnan(x))nan++; else if(isinf(x))inf++; else {s+=x; s2+=x*x;}}
  printf("%-8s sum=%.17g sumsq=%.17g nan=%ld inf=%ld\n",name,s,s2,nan,inf);
}

int main(int argc,char**argv){
  if(argc<2){fprintf(stderr,"usage: %s lib.so\n",argv[0]);return 2;}
  void*h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen: %s\n",dlerror());return 3;}
  #define SYM(t,n) t n=(t)dlsym(h,#n); if(!n){fprintf(stderr,"missing %s\n",#n);return 4;}
  SYM(setnt_t,openblas_set_num_threads); openblas_set_num_threads(1);
  SYM(dgemm_t,cblas_dgemm)
  SYM(dtrsm_t,cblas_dtrsm)
  SYM(dgemv_t,cblas_dgemv)
  SYM(dscal_t,cblas_dscal)
  SYM(daxpy_t,cblas_daxpy)
  SYM(idamax_t,cblas_idamax)
  SYM(dnrm2_t,cblas_dnrm2)
  SYM(ddot_t,cblas_ddot)

  /* ---- level-1 vector ops ---- */
  long nv=4096;
  double*x=malloc(nv*sizeof(double)),*y=malloc(nv*sizeof(double));
  for(long i=0;i<nv;i++){x[i]=rnd(i); y[i]=rnd(i+100000);}
  size_t im=cblas_idamax(nv,x,1);
  printf("idamax   idx=%zu val=%.17g\n",im,(im<(size_t)nv?x[im]:0.0/0.0));
  printf("dnrm2    val=%.17g\n",cblas_dnrm2(nv,x,1));
  printf("ddot     val=%.17g\n",cblas_ddot(nv,x,1,y,1));
  double*xs=malloc(nv*sizeof(double)); memcpy(xs,x,nv*sizeof(double));
  cblas_dscal(nv,3.14159,xs,1); report("dscal",xs,nv);
  double*ya=malloc(nv*sizeof(double)); memcpy(ya,y,nv*sizeof(double));
  cblas_daxpy(nv,-2.5,x,1,ya,1); report("daxpy",ya,nv);

  /* ---- dgemv (col-major, NoTrans) ---- */
  int M=768,N=768;
  double*A=malloc((size_t)M*N*sizeof(double));
  for(long i=0;i<(long)M*N;i++)A[i]=rnd(i+1);
  double*xv=malloc(N*sizeof(double)),*yv=malloc(M*sizeof(double));
  for(int i=0;i<N;i++)xv[i]=rnd(i+7);
  for(int i=0;i<M;i++)yv[i]=rnd(i+9);
  cblas_dgemv(ColMajor,NoTrans,M,N,1.3,A,M,xv,1,0.7,yv,1); report("dgemv",yv,M);

  /* ---- dgemm sanity ---- */
  int mm=256;
  double*Ag=malloc((size_t)mm*mm*sizeof(double)),*Bg=malloc((size_t)mm*mm*sizeof(double)),*Cg=malloc((size_t)mm*mm*sizeof(double));
  for(long i=0;i<(long)mm*mm;i++){Ag[i]=rnd(i+3);Bg[i]=rnd(i+5);Cg[i]=0;}
  cblas_dgemm(ColMajor,NoTrans,NoTrans,mm,mm,mm,1.0,Ag,mm,Bg,mm,0.0,Cg,mm); report("dgemm",Cg,(long)mm*mm);

  /* ---- dtrsm (Left,Lower,NoTrans,NonUnit) diag-dominant ---- */
  int tm=512,tn=64;
  double*T=malloc((size_t)tm*tm*sizeof(double));
  for(int j=0;j<tm;j++)for(int i=0;i<tm;i++){
    double v=(i==j)?(tm+1.0):(i>j?rnd((long)j*tm+i)*0.1:0.0);
    T[(size_t)j*tm+i]=v; }
  double*B=malloc((size_t)tm*tn*sizeof(double));
  for(long i=0;i<(long)tm*tn;i++)B[i]=rnd(i+13);
  cblas_dtrsm(ColMajor,Left,Lower,NoTrans,NonUnit,tm,tn,1.0,T,tm,B,tm); report("dtrsm",B,(long)tm*tn);

  printf("DONE\n");
  return 0;
}
