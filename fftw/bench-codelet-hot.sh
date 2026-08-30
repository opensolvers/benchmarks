#!/bin/bash
# Microbench hot r5v codelet shapes: 1D power-of-two (MEASURE) + QE-like many_dft.
# Usage: LIB=r5v|patched ./bench-codelet-hot.sh
set -euo pipefail
ROOT=${ROOT:-$HOME/fftwbuild}
LIB=${LIB:-r5v}
DIR=$ROOT/src-$LIB
SO=$DIR/.libs/libfftw3.so.3.6.10
INC=$DIR/api
LOG=${LOG:-$HOME/logs/fftw-codelet-hot-$LIB-$(date +%Y%m%d-%H%M%S).log}
mkdir -p "$(dirname "$LOG")"
EESSI_SW=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software
GCC=$EESSI_SW/GCCcore/14.3.0
export PATH=$GCC/bin:$PATH
export LIBRARY_PATH=$GCC/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=$DIR/.libs:$GCC/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cat > /tmp/bench_codelet_hot.c <<'C'
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void bench_1d(int n, int reps) {
  fftw_complex *in = fftw_malloc(sizeof(*in) * (size_t)n);
  fftw_complex *out = fftw_malloc(sizeof(*out) * (size_t)n);
  for (int i = 0; i < n; i++) { in[i][0] = i * 1e-3; in[i][1] = -i * 1e-4; }
  double t0 = now();
  fftw_plan p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_MEASURE);
  double tp = now() - t0;
  t0 = now();
  for (int r = 0; r < reps; r++) fftw_execute(p);
  double te = now() - t0;
  double mflops = (5.0 * n * (n ? __builtin_log2(n) : 0) * reps) / (te * 1e6);
  printf("[1d] N=%d plan=%.4fs exec=%.4fs (%d reps) ~%.0f MFLOPS\n",
         n, tp, te, reps, mflops);
  fflush(stdout);
  fftw_destroy_plan(p);
  fftw_free(in); fftw_free(out);
}

/* QE-like: many 1D pencils along one axis of a 64^3 grid (howmany=64*64). */
static void bench_many(int n, int howmany, int reps) {
  size_t N = (size_t)n * (size_t)howmany;
  fftw_complex *in = fftw_malloc(sizeof(*in) * N);
  fftw_complex *out = fftw_malloc(sizeof(*out) * N);
  memset(in, 0, sizeof(*in) * N);
  for (size_t i = 0; i < N; i++) { in[i][0] = 1e-3 * (double)i; }
  int nn[1] = {n};
  double t0 = now();
  fftw_plan p = fftw_plan_many_dft(1, nn, howmany,
                                   in, NULL, 1, n,
                                   out, NULL, 1, n,
                                   FFTW_FORWARD, FFTW_MEASURE);
  double tp = now() - t0;
  t0 = now();
  for (int r = 0; r < reps; r++) fftw_execute(p);
  double te = now() - t0;
  double flops = 5.0 * n * __builtin_log2(n) * howmany * reps;
  printf("[many] N=%d howmany=%d plan=%.4fs exec=%.4fs (%d reps) ~%.0f MFLOPS\n",
         n, howmany, tp, te, reps, flops / (te * 1e6));
  fflush(stdout);
  fftw_destroy_plan(p);
  fftw_free(in); fftw_free(out);
}

int main(void) {
  printf("FFTW hot-codelet microbench\n");
  bench_1d(256, 4000);
  bench_1d(1024, 2000);
  bench_1d(4096, 800);
  bench_1d(65536, 80);
  /* 64-point pencils, 4096 of them — closer to QE 64^3 axis */
  bench_many(64, 4096, 200);
  bench_many(64, 64, 2000);
  return 0;
}
C

gcc -O2 -I"$INC" /tmp/bench_codelet_hot.c -o /tmp/bench_codelet_hot \
  -L"$DIR/.libs" -lfftw3 -Wl,-rpath,"$DIR/.libs" -lm
{
  echo "=== codelet-hot LIB=$LIB $(date -Iseconds) ==="
  echo "SO=$SO"
  /tmp/bench_codelet_hot
  echo "=== FFTW bench sanity (MEASURE) @ 256 1024 4096 ==="
  for s in 256 1024 4096; do
    (cd "$DIR/tests" && LD_LIBRARY_PATH="$DIR/.libs" ./bench --report-mflops -t 1.0 -s "$s" 2>&1) \
      | tee -a /dev/stderr | tr -d '()' | grep -oE '[0-9]+\.[0-9]+' | \
      awk -v s="$s" '{a[++n]=$1} END{
        asort(a); med=(n%2)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2;
        printf("[bench] N=%s med=%.2f\n", s, med)}'
  done
  echo DONE
} 2>&1 | tee "$LOG"
echo "LOG=$LOG"
