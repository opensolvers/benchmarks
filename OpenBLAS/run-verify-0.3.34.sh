#!/bin/bash
# Build OpenBLAS v0.3.34 (RISCV64_ZVL256B) on RV2 and verify RVV kernels end-to-end.
# Usage: bash run-verify-0.3.34.sh
set -euo pipefail
LOG="${LOG:-$HOME/logs/openblas-0.3.34-verify-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Iseconds) log=$LOG"

HERE=$(cd "$(dirname "$0")" && pwd)
SRC_DIR=${OB_SRC:-$HOME/ob-0.3.34}
PREFIX=${OB_PREFIX:-$HOME/openblas-0.3.34}
LIB=$PREFIX/lib/libopenblas.so
BENCH=${OB_BIN:-$HOME/openblas-bench}
N=${DGEMM_N:-2048}
THREADS=${THREADS:-8}
JOBS=${JOBS:-$(nproc)}

STOCK030=${STOCK030:-/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/OpenBLAS/0.3.30-GCC-14.3.0/lib/libopenblas.so}
STOCK029=${STOCK029:-/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/OpenBLAS/0.3.29-GCC-14.2.0/lib/libopenblas.so}

# System GCC 13 cannot compile ZVL256B SGEMM (missing RVV tuple intrinsics).
# EESSI GCCcore 14.3.0 matches the prior working X60 OpenBLAS builds.
export LMOD_IGNORE_CACHE=yes
export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_NO_MODULE_PURGE_ON_INIT="${EESSI_NO_MODULE_PURGE_ON_INIT:-}"
set +u
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module --ignore_cache load GCC/14.3.0
set -u
echo "CC=$(gcc --version | head -1) ($(which gcc))"

echo "SRC_DIR=$SRC_DIR PREFIX=$PREFIX JOBS=$JOBS"

# ---- fetch / checkout v0.3.34 ----
if [[ ! -d $SRC_DIR/.git ]]; then
  git clone --depth 1 --branch v0.3.34 https://github.com/OpenMathLib/OpenBLAS.git "$SRC_DIR"
else
  cd "$SRC_DIR"
  git fetch --tags --depth 1 origin tag v0.3.34 2>/dev/null || git fetch --tags origin
  git checkout -f v0.3.34
  git clean -fdx -e '*.so' -e '*.a' 2>/dev/null || true
fi
cd "$SRC_DIR"
echo "OpenBLAS $(git describe --tags --always) @ $(git rev-parse --short HEAD)"

# ---- build shared ZVL256B RVV ----
if [[ ! -f $LIB ]] || [[ "${FORCE_REBUILD:-0}" = 1 ]]; then
  echo "Building TARGET=RISCV64_ZVL256B ..."
  make clean >/dev/null 2>&1 || true
  # NO_LAPACK=1: harnesses only need BLAS (dgemm/dgemv/dtrsm/dsyrk/ctrsm).
  make -j"$JOBS" \
    TARGET=RISCV64_ZVL256B \
    BINARY=64 \
    CC=gcc FC=gfortran \
    HOSTCC=gcc \
    USE_OPENMP=1 \
    NO_LAPACK=1 \
    NUM_THREADS=8 \
    libs shared
  mkdir -p "$PREFIX/lib" "$PREFIX/include"
  cp -a libopenblas.so* "$PREFIX/lib/"
  # soname may be versioned; ensure libopenblas.so exists
  if [[ ! -e $PREFIX/lib/libopenblas.so ]]; then
    ln -sfn "$(ls -1 "$PREFIX/lib"/libopenblas.so.* | head -1 | xargs -n1 basename)" "$PREFIX/lib/libopenblas.so"
  fi
  cp -a cblas.h common*.h openblas_config.h "$PREFIX/include/" 2>/dev/null || true
  # config header path varies
  find . -maxdepth 2 -name 'openblas_config.h' -exec cp -a {} "$PREFIX/include/" \; 2>/dev/null || true
  echo "Installed $LIB"
  ls -la "$PREFIX/lib"/libopenblas*
else
  echo "Reusing existing $LIB (FORCE_REBUILD=1 to rebuild)"
fi

# ---- build harnesses ----
mkdir -p "$BENCH"
cp -f "$HERE"/bench_dgemm.c "$HERE"/difftest.c "$HERE"/verify_ctrsm.c "$BENCH"/
gcc -O2 "$BENCH/bench_dgemm.c" -o "$BENCH/bench_dgemm" -L"$PREFIX/lib" -Wl,-rpath,"$PREFIX/lib" -lopenblas -lm
gcc -O2 "$BENCH/difftest.c" -o "$BENCH/difftest" -ldl -lm
# verify_ctrsm needs cblas.h from the build
INC=
for d in "$PREFIX/include" "$SRC_DIR" "$SRC_DIR/cblas"; do
  [[ -f $d/cblas.h ]] && INC=$d && break
done
gcc -O2 -I"$INC" "$BENCH/verify_ctrsm.c" -o "$BENCH/verify_ctrsm" \
  -L"$PREFIX/lib" -Wl,-rpath,"$PREFIX/lib" -lopenblas -lm
echo "harnesses built (cblas.h from $INC)"

# ---- SYRK PSD reproducer from OpenBLAS#5811 (the 0.3.33 ZVL256 killer) ----
cat > "$BENCH/syrk_psd.c" <<'C'
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
/* Upper+Trans SYRK: C = A^T A, N=K=50 — fails on broken 0.3.33 ZVL256B */
enum { RowMajor=101, Upper=121, Trans=112 };
static double rnd(long i){
  unsigned long x=(unsigned long)i*2654435761UL+1013904223UL;
  x^=x>>13; x*=0xD1B54A32D192ED03UL; x^=x>>29;
  return ((double)(x>>40)/(double)(1UL<<24))*2.0-1.0;
}
int main(void){
  const int N=50, K=50;
  double *A=malloc((size_t)K*N*sizeof(double));
  double *C=calloc((size_t)N*N,sizeof(double));
  double *Ref=calloc((size_t)N*N,sizeof(double));
  for(int i=0;i<K*N;i++) A[i]=rnd(i+1);
  /* reference: naive A^T A (row-major A is K x N → col-major view for cblas) */
  /* Use ColMajor Trans: A is N x K column-major equiv of row-major K x N */
  cblas_dsyrk(RowMajor, Upper, Trans, N, K, 1.0, A, N, 0.0, C, N);
  /* naive ref */
  for(int i=0;i<N;i++) for(int j=i;j<N;j++){
    double s=0;
    for(int k=0;k<K;k++) s += A[k*N+i]*A[k*N+j]; /* A row-major KxN */
    Ref[i*N+j]=s;
  }
  double maxerr=0; int nan=0;
  for(int i=0;i<N;i++) for(int j=i;j<N;j++){
    double e=fabs(C[i*N+j]-Ref[i*N+j]);
    if(isnan(C[i*N+j])||isinf(C[i*N+j])) nan++;
    if(e>maxerr) maxerr=e;
  }
  /* crude min eig of upper via diagonal dominance check: min diag of C */
  double mind=C[0];
  for(int i=0;i<N;i++) if(C[i*N+i]<mind) mind=C[i*N+i];
  printf("syrk_psd N=%d max_err=%.6e min_diag=%.6e nan=%d %s\n",
         N, maxerr, mind, nan, (maxerr<1e-8 && nan==0)?"PASS":"FAIL");
  free(A); free(C); free(Ref);
  return (maxerr<1e-8 && nan==0)?0:1;
}
C
gcc -O2 -I"$INC" "$BENCH/syrk_psd.c" -o "$BENCH/syrk_psd" \
  -L"$PREFIX/lib" -Wl,-rpath,"$PREFIX/lib" -lopenblas -lm

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo ""
echo "========== 1) difftest correctness (1 thread) =========="
for tag_lib in \
  "0.3.34:$LIB" \
  "stock-0.3.30:$STOCK030" \
  "stock-0.3.29:$STOCK029"
do
  tag=${tag_lib%%:*}; so=${tag_lib#*:}
  [[ -f $so ]] || { echo "SKIP $tag (missing $so)"; continue; }
  echo "--- $tag ($so) ---"
  "$BENCH/difftest" "$so" | grep -E 'dgemv|dgemm|dtrsm|nan|inf|sum='
  echo "--- $tag forced scalar ---"
  OPENBLAS_CORETYPE=RISCV64_GENERIC "$BENCH/difftest" "$so" 2>/dev/null | grep -E 'dgemv|dgemm|dtrsm|nan' || true
done

echo ""
echo "========== 2) SYRK PSD (OpenBLAS#5811 repro) =========="
"$BENCH/syrk_psd"; echo "syrk_rc=$?"

echo ""
echo "========== 3) verify_ctrsm =========="
"$BENCH/verify_ctrsm"; echo "ctrsm_rc=$?"

echo ""
echo "========== 4) DGEMM perf A/B N=$N threads=$THREADS =========="
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
# Link only against libopenblas via absolute path (no FlexiBLAS required).
link_bench() {
  local out=$1 so=$2
  gcc -O2 "$BENCH/bench_dgemm.c" -o "$out" "$so" -lm -lpthread -fopenmp \
    -Wl,-rpath,"$(dirname "$so")"
}
run_one() {
  local tag=$1 bin=$2; shift 2
  echo "--- DGEMM [$tag] ---"
  env "$@" "$bin" "$N" || true
}
link_bench "$BENCH/bench_dgemm_034" "$LIB"
run_one "0.3.34-rvv" "$BENCH/bench_dgemm_034"
run_one "0.3.34-scalar" "$BENCH/bench_dgemm_034" OPENBLAS_CORETYPE=RISCV64_GENERIC
if [[ -f $STOCK030 ]]; then
  link_bench "$BENCH/bench_dgemm_030" "$STOCK030"
  run_one "stock-0.3.30" "$BENCH/bench_dgemm_030"
fi
if [[ -f $HOME/libopenblas_x60_eb_fixed.so ]]; then
  link_bench "$BENCH/bench_dgemm_fixed" "$HOME/libopenblas_x60_eb_fixed.so"
  run_one "patched-0.3.30" "$BENCH/bench_dgemm_fixed"
fi

echo ""
echo "DONE $(date -Iseconds) log=$LOG"
echo "LIB=$LIB"
