#!/bin/bash
# Build OpenBLAS TARGET=RISCV64_ZVL1024B on k3 and smoke-test DGEMM on X100 + A100.
set -eo pipefail
source "$HOME/.eessi_init"
module load foss/2025b

WORK="$HOME/openblas-zvl1024b-build"
PREFIX="$HOME/openblas-zvl1024b"
HPLDIR="$HOME/hpl-baseline"
mkdir -p "$WORK"
cd "$WORK"

TARBALL="$HOME/openblas-build/OpenBLAS-0.3.34.tar.gz"
rm -rf openblas-src
mkdir openblas-src
tar -xzf "$TARBALL" -C openblas-src --strip-components=1
cd openblas-src

echo "=== applying ZVL1024B patch $(date -Is) ==="
patch -p1 < "$HPLDIR/OpenBLAS-0.3.34_add-riscv64-zvl1024b.patch"

echo "=== building TARGET=RISCV64_ZVL1024B $(date -Is) ==="
NJOBS=$(nproc)
# libs only — do not run BLAS test suite on X100 (VLEN=256) with zvl1024b kernels
make -j"$NJOBS" libs netlib shared \
  CC=gcc FC=gfortran HOSTCC=gcc \
  TARGET=RISCV64_ZVL1024B \
  BINARY=64 \
  USE_OPENMP=1 \
  NUM_THREADS=8 \
  NO_CBLAS=0 \
  NO_LAPACK=0 \
  INTERFACE64=0 \
  DYNAMIC_ARCH=0 \
  CROSS=0

rm -rf "$PREFIX"
make PREFIX="$PREFIX" install
echo "Installed $PREFIX"
ls -la "$PREFIX/lib"/libopenblas*

cat > /tmp/dgemm_smoke.c <<'C'
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
extern char *openblas_get_corename(void);
extern char *openblas_get_config(void);
int main(void) {
  printf("corename=%s\n", openblas_get_corename());
  printf("config=%s\n", openblas_get_config());
  const int N = 512;
  double *A = calloc(N*N, sizeof(double));
  double *B = calloc(N*N, sizeof(double));
  double *C = calloc(N*N, sizeof(double));
  for (int i=0;i<N*N;i++){ A[i]=1.0; B[i]=2.0; }
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
  double expect = 2.0 * N;
  int bad = 0;
  for (int i=0;i<N*N;i++) if (C[i] != expect) { bad++; break; }
  printf("dgemm %dx%d expect=%g got=%g %s\n", N, N, expect, C[0], bad?"FAIL":"OK");
  free(A); free(B); free(C);
  return bad ? 1 : 0;
}
C
gcc -O2 /tmp/dgemm_smoke.c -I"$PREFIX/include" -L"$PREFIX/lib" \
  -Wl,-rpath,"$PREFIX/lib" -lopenblas -lpthread -lgomp -lm -o /tmp/dgemm_smoke

echo "=== A100 smoke (required; X100 VLEN too short for this TARGET) ==="
bash -c 'echo $$ > /proc/set_ai_thread; exec taskset -c 8-15 /tmp/dgemm_smoke'

echo "BUILD_OK $(date -Is)"
