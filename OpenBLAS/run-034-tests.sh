#!/bin/bash
# Run OpenBLAS 0.3.34 verification tests (after build).
set -euo pipefail
export LMOD_IGNORE_CACHE=yes EESSI_VERSION_OVERRIDE=2025.06-001 EESSI_NO_MODULE_PURGE_ON_INIT=
set +u
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load GCC/14.3.0
set -u

REAL=$HOME/ob-0.3.34/libopenblas_riscv64_zvl256bp-r0.3.34.so
mkdir -p "$HOME/openblas-0.3.34/lib"
cp -a "$REAL" "$HOME/openblas-0.3.34/lib/"
ln -sfn libopenblas_riscv64_zvl256bp-r0.3.34.so "$HOME/openblas-0.3.34/lib/libopenblas.so.0"
ln -sfn libopenblas.so.0 "$HOME/openblas-0.3.34/lib/libopenblas.so"

BENCH=$HOME/openblas-bench
SRC=$HOME/openblas-verify
INC=$HOME/ob-0.3.34
STOCK030=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/OpenBLAS/0.3.30-GCC-14.3.0/lib/libopenblas.so
STOCK029=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/OpenBLAS/0.3.29-GCC-14.2.0/lib/libopenblas.so
FIXED=$HOME/libopenblas_x60_eb_fixed.so
N=${DGEMM_N:-2048}
THREADS=${THREADS:-8}
RP=$HOME/ob-0.3.34

mkdir -p "$BENCH"
cp -f "$SRC"/{bench_dgemm.c,difftest.c,verify_ctrsm.c} "$BENCH"/
gcc -O2 "$BENCH/difftest.c" -o "$BENCH/difftest" -ldl -lm

cat > "$BENCH/syrk_psd.c" <<'C'
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
enum { RowMajor=101, Upper=121, Trans=112 };
static double rnd(long i){
  unsigned long x=(unsigned long)i*2654435761UL+1013904223UL;
  x^=x>>13; x*=0xD1B54A32D192ED03UL; x^=x>>29;
  return ((double)(x>>40)/(double)(1UL<<24))*2.0-1.0;
}
int main(void){
  const int N=50,K=50;
  double *A=malloc((size_t)K*N*sizeof(double));
  double *C=calloc((size_t)N*N,sizeof(double));
  double *Ref=calloc((size_t)N*N,sizeof(double));
  for(int i=0;i<K*N;i++) A[i]=rnd(i+1);
  cblas_dsyrk(RowMajor, Upper, Trans, N, K, 1.0, A, N, 0.0, C, N);
  for(int i=0;i<N;i++) for(int j=i;j<N;j++){
    double s=0; for(int k=0;k<K;k++) s+=A[k*N+i]*A[k*N+j];
    Ref[i*N+j]=s;
  }
  double maxerr=0; int nan=0;
  for(int i=0;i<N;i++) for(int j=i;j<N;j++){
    double e=fabs(C[i*N+j]-Ref[i*N+j]);
    if(isnan(C[i*N+j])||isinf(C[i*N+j])) nan++;
    if(e>maxerr) maxerr=e;
  }
  double mind=C[0]; for(int i=0;i<N;i++) if(C[i*N+i]<mind) mind=C[i*N+i];
  printf("syrk_psd max_err=%.6e min_diag=%.6e nan=%d %s\n", maxerr, mind, nan, (maxerr<1e-8&&nan==0)?"PASS":"FAIL");
  return (maxerr<1e-8&&nan==0)?0:1;
}
C
gcc -O2 -I"$INC" "$BENCH/syrk_psd.c" -o "$BENCH/syrk_psd" "$REAL" -lm -lpthread -fopenmp -Wl,-rpath,"$RP"
gcc -O2 -I"$INC" "$BENCH/verify_ctrsm.c" -o "$BENCH/verify_ctrsm" "$REAL" -lm -lpthread -fopenmp -Wl,-rpath,"$RP"

link_bench(){ gcc -O2 "$BENCH/bench_dgemm.c" -o "$1" "$2" -lm -lpthread -fopenmp -Wl,-rpath,"$(dirname "$2")"; }

LOG=$HOME/logs/openblas-034-results-$(date +%Y%m%d-%H%M%S).log
{
echo "=== OpenBLAS 0.3.34 RVV verify RV2 ==="
echo "LIB=$REAL"
echo ""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "=== 1) difftest ==="
for pair in "0.3.34-rvv:$REAL" "stock-0.3.30:$STOCK030" "stock-0.3.29:$STOCK029" "patched-0.3.30:$FIXED"; do
  t=${pair%%:*}; s=${pair#*:}
  [[ -f $s ]] || { echo "SKIP $t"; continue; }
  echo "--- $t ---"
  "$BENCH/difftest" "$s" | grep -E 'dgemv|dgemm|dtrsm|nan'
  echo "--- $t scalar ---"
  OPENBLAS_CORETYPE=RISCV64_GENERIC "$BENCH/difftest" "$s" 2>/dev/null | grep -E 'dgemv|dgemm|dtrsm|nan' || true
done
echo ""
echo "=== 2) SYRK PSD (#5811) ==="
set +e; "$BENCH/syrk_psd"; echo syrk_rc=$?; set -e
echo ""
echo "=== 3) verify_ctrsm ==="
set +e; "$BENCH/verify_ctrsm"; echo ctrsm_rc=$?; set -e
echo ""
echo "=== 4) DGEMM N=$N threads=$THREADS ==="
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
link_bench "$BENCH/bench_dgemm_034" "$REAL"
link_bench "$BENCH/bench_dgemm_030" "$STOCK030"
link_bench "$BENCH/bench_dgemm_fixed" "$FIXED"
echo "--- 0.3.34-rvv ---"; "$BENCH/bench_dgemm_034" "$N"
echo "--- 0.3.34-scalar ---"; OPENBLAS_CORETYPE=RISCV64_GENERIC "$BENCH/bench_dgemm_034" "$N"
echo "--- stock-0.3.30 ---"; "$BENCH/bench_dgemm_030" "$N"
echo "--- patched-0.3.30 ---"; "$BENCH/bench_dgemm_fixed" "$N"
echo DONE
} 2>&1 | tee "$LOG"
echo LOG=$LOG
