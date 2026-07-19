/*
 * verify_ctrsm.c - CTRSM (complex single) numerical-correctness sweep.
 *
 * Solves op(A) X = alpha B (Left) or X op(A) = alpha B (Right) for a triangular
 * A across the full parameter space - {Left,Right} x {Upper,Lower} x
 * {NoTrans,Trans,ConjTrans} x {NonUnit,Unit} x a size grid - then recomputes
 * op(A)*X (or X*op(A)) and checks the max residual against the original B.
 * Used to validate the RISC-V RVV (_rvv_v1) TRSM kernels: the VLEN-agnostic fix
 * in OpenMathLib/OpenBLAS#5928 passes 0 fails here on both ZVL128B (GEMM_UNROLL_M
 * =8) and ZVL256B (=16); the pre-fix kernels fail on ZVL128B (unroll 8 != the
 * VSETVL_MAX=16 the kernel tiled packed-A by).
 *
 * The S/D/Z variants are structurally identical - swap the type, the cblas_?trsm
 * call, and the element generator (real vs complex).
 *
 * Build:  gcc -O2 verify_ctrsm.c -o verify_ctrsm -I<cblas_include> -lopenblas -lm
 * Run:    ./verify_ctrsm            # exit 0 = all pass, prints "N cases, F fails"
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "cblas.h"

typedef float _Complex cd;
static double frand(void){ return (double)rand()/RAND_MAX - 0.5; }
static cd crand(void){ return frand() + frand()*I; }

static void make_tri(cd *A, int n, int uplo_upper, int unit){
    for(int j=0;j<n;j++) for(int i=0;i<n;i++){
        cd v;
        if(i==j) v = unit ? (1.0+0.0*I) : ((2.0+(double)(i%3)) + 0.5*I);
        else if((uplo_upper && i<j) || (!uplo_upper && i>j)) v = 0.3*crand();
        else v = 0.0;
        A[i+j*n]=v;
    }
}

static cd opA(int trans,int uplo,int is_unit,const cd*A,int lda,int r,int c){
    /* returns element (r,c) of op(A) given stored triangular A */
    int sr,sc;
    if(trans==CblasNoTrans){ sr=r; sc=c; }
    else if(trans==CblasTrans){ sr=c; sc=r; }
    else { sr=c; sc=r; } /* ConjTrans handled by caller via conj */
    cd a;
    if(sr==sc) a = is_unit ? (1.0+0.0*I) : A[sr+sc*lda];
    else if((uplo==CblasUpper && sr<sc)||(uplo==CblasLower && sr>sc)) a=A[sr+sc*lda];
    else a=0.0;
    if(trans==CblasConjTrans) a = conj(a);
    return a;
}

static double resid(int side,int uplo,int trans,int is_unit,int m,int n,
                    const cd*A,int lda,const cd*X,const cd*B0,int ldb){
    double maxr=0.0;
    if(side==CblasLeft){
        for(int i=0;i<m;i++) for(int jj=0;jj<n;jj++){
            cd s=0.0;
            for(int k=0;k<m;k++) s += opA(trans,uplo,is_unit,A,lda,i,k) * X[k+jj*ldb];
            double d=cabs(s - B0[i+jj*ldb]); if(d>maxr) maxr=d;
        }
    } else {
        for(int i=0;i<m;i++) for(int jj=0;jj<n;jj++){
            cd s=0.0;
            for(int k=0;k<n;k++) s += X[i+k*ldb]*opA(trans,uplo,is_unit,A,lda,k,jj);
            double d=cabs(s - B0[i+jj*ldb]); if(d>maxr) maxr=d;
        }
    }
    return maxr;
}

int main(void){
    int grid[]={1,2,4,7,8,9,10,12,15,23};
    int ng=sizeof(grid)/sizeof(grid[0]);
    int sides[]={CblasLeft,CblasRight}, uplos[]={CblasUpper,CblasLower};
    int transs[]={CblasNoTrans,CblasTrans,CblasConjTrans}, diags[]={CblasNonUnit,CblasUnit};
    int fails=0,total=0; double worst=0.0; srand(999);
    cd alpha=1.0+0.0*I;
    for(int si=0;si<2;si++) for(int ui=0;ui<2;ui++) for(int ti=0;ti<3;ti++) for(int di=0;di<2;di++)
    for(int im=0;im<ng;im++) for(int in=0;in<ng;in++){
        int m=grid[im], n=grid[in];
        int side=sides[si],uplo=uplos[ui],trans=transs[ti],unit=diags[di];
        int ao=(side==CblasLeft)?m:n; int lda=ao, ldb=m;
        cd *A=malloc(sizeof(cd)*ao*ao);
        cd *B=malloc(sizeof(cd)*m*n), *B0=malloc(sizeof(cd)*m*n);
        make_tri(A,ao, uplo==CblasUpper, unit==CblasUnit);
        for(int x=0;x<m*n;x++){ B[x]=crand(); B0[x]=B[x]; }
        cblas_ctrsm(CblasColMajor,side,uplo,trans,unit,m,n,&alpha,A,lda,B,ldb);
        double r=resid(side,uplo,trans,(unit==CblasUnit),m,n,A,lda,B,B0,ldb);
        double tol=1e-4*(ao+1);
        total++; if(r>worst) worst=r;
        if(!(r<=tol)||isnan(r)){ fails++;
            if(fails<=30)
            printf("FAIL side=%d uplo=%d trans=%d unit=%d m=%d n=%d resid=%g tol=%g\n",
                   side,uplo,trans,unit,m,n,r,tol); }
        free(A);free(B);free(B0);
    }
    printf("=== CTRSM: %d cases, %d fails, worst_resid=%g ===\n", total, fails, worst);
    return fails?1:0;
}
