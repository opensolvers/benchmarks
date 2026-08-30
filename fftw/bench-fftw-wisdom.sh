#!/bin/bash
# Generate FFTW MEASURE wisdom and A/B ESTIMATE vs MEASURE vs wisdom-warm MEASURE.
#
# Shows that a wisdom file recovers MEASURE-quality plans at near-ESTIMATE
# planning cost — the lever that QE cannot use while it hardcodes ESTIMATE.
#
# Usage:  ./bench-fftw-wisdom.sh
# Env:    ROOT   fftwbuild root (default $HOME/fftwbuild)
#         LIB    which lib: r5v|scalar (default r5v)
#         SIZES  "256 1024 4096 ..." (default cache-resident set)
#         LOG    output log path
set -euo pipefail

ROOT=${ROOT:-$HOME/fftwbuild}
LIBNAME=${LIB:-r5v}
SIZES=${SIZES:-'256 1024 4096 16384 65536'}
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=${LOG:-$HOME/logs/fftw-wisdom-$(date +%Y%m%d-%H%M%S).log}
WISDIR=${WISDIR:-$HOME/fftw-wisdom}
mkdir -p "$(dirname "$LOG")" "$WISDIR"

case "$LIBNAME" in
  r5v)    SRC=$ROOT/src-r5v ;;
  scalar) SRC=$ROOT/src-scalar ;;
  *) echo "LIB must be r5v or scalar"; exit 1 ;;
esac
SO=$SRC/.libs/libfftw3.so.3.6.10
WISDOM=$SRC/tools/.libs/fftw-wisdom
BENCH=$SRC/tests/.libs/bench
[ -x "$SO" ] || [ -f "$SO" ] || { echo "missing $SO"; exit 1; }
[ -x "$WISDOM" ] || WISDOM=$SRC/tools/fftw-wisdom
[ -x "$BENCH" ] || BENCH=$SRC/tests/bench
export LD_LIBRARY_PATH=$SRC/.libs:${LD_LIBRARY_PATH:-}

echo "=== FFTW wisdom A/B LIB=$LIBNAME $(date -Iseconds) ===" | tee "$LOG"
echo "SO=$SO" | tee -a "$LOG"

# Build size list for fftw-wisdom: complex, out-of-place, forward+backward 1D
# (matches tests/bench default geometry for -s N).
WSIZES=()
for n in $SIZES; do
  WSIZES+=("cof${n}" "cob${n}")
done
WISFILE=$WISDIR/wisdom-${LIBNAME}-1d.fftw

echo "=== 1) generate MEASURE wisdom → $WISFILE ===" | tee -a "$LOG"
# -m = MEASURE (PATIENT is default and too slow at large N)
t0=$(date +%s)
"$WISDOM" -v -m -n -o "$WISFILE" "${WSIZES[@]}" 2>&1 | tee -a "$LOG"
echo "wisdom_wall=$(( $(date +%s) - t0 )) s" | tee -a "$LOG"
wc -c "$WISFILE" | tee -a "$LOG"

# Small C harness: plan+exec timing under three modes
cat > /tmp/fftw_wisdom_probe.c <<'C'
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void run_mode(const char *tag, int n, unsigned flags, int reps) {
  fftw_complex *in  = fftw_malloc(sizeof(fftw_complex) * (size_t)n);
  fftw_complex *out = fftw_malloc(sizeof(fftw_complex) * (size_t)n);
  for (int i = 0; i < n; i++) { in[i][0] = 0.1 * (i % 17); in[i][1] = 0.01 * (i % 9); }

  double t0 = wall_s();
  fftw_plan p = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, flags);
  double t_plan = wall_s() - t0;
  if (!p) {
    printf("[%s] N=%d PLAN_FAILED flags=0x%x\n", tag, n, flags);
    fftw_free(in); fftw_free(out);
    return;
  }

  /* warmup */
  fftw_execute(p);
  t0 = wall_s();
  for (int r = 0; r < reps; r++) fftw_execute(p);
  double t_exec = wall_s() - t0;
  double mflops = (5.0 * n * __builtin_log2(n) * reps) / (t_exec * 1e6);

  printf("[%s] N=%d plan=%.4fs exec=%.4fs (%d reps) ~%.0f MFLOPS flags=0x%x\n",
         tag, n, t_plan, t_exec, reps, mflops, flags);
  fftw_destroy_plan(p);
  fftw_free(in); fftw_free(out);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <wisdom.fftw|NONE> N [N...]\n", argv[0]);
    return 2;
  }
  const char *wfile = argv[1];
  int have_w = strcmp(wfile, "NONE") != 0;
  if (have_w) {
    FILE *f = fopen(wfile, "r");
    if (!f) { perror(wfile); return 1; }
    if (!fftw_import_wisdom_from_file(f)) {
      fprintf(stderr, "wisdom import failed: %s\n", wfile);
      return 1;
    }
    fclose(f);
    printf("imported wisdom from %s\n", wfile);
  }

  for (int a = 2; a < argc; a++) {
    int n = atoi(argv[a]);
    int reps = (n <= 4096) ? 2000 : (n <= 16384) ? 400 : 80;

    /* forget between modes so ESTIMATE cannot accidentally reuse MEASURE plans
       from the same process — wisdom file is re-imported when needed. */
    if (have_w) {
      fftw_forget_wisdom();
      FILE *f = fopen(wfile, "r");
      fftw_import_wisdom_from_file(f);
      fclose(f);
    } else {
      fftw_forget_wisdom();
    }

    run_mode("ESTIMATE", n, FFTW_ESTIMATE, reps);

    if (have_w) {
      fftw_forget_wisdom();
      FILE *f = fopen(wfile, "r");
      fftw_import_wisdom_from_file(f);
      fclose(f);
      /* MEASURE with wisdom: planning should be nearly free, quality = MEASURE */
      run_mode("MEASURE+wisdom", n, FFTW_MEASURE, reps);
      fftw_forget_wisdom();
      f = fopen(wfile, "r");
      fftw_import_wisdom_from_file(f);
      fclose(f);
      /* WISDOM_ONLY|ESTIMATE: use wisdom if present (app-friendly fast path) */
      run_mode("WISDOM_ONLY", n, FFTW_ESTIMATE | FFTW_WISDOM_ONLY, reps);
    } else {
      run_mode("MEASURE-cold", n, FFTW_MEASURE, reps);
    }
  }
  return 0;
}
C

echo "=== 2) build probe ===" | tee -a "$LOG"
INC=$SRC/api
gcc -O2 -I"$INC" /tmp/fftw_wisdom_probe.c -o "$WISDIR/fftw_wisdom_probe" \
  "$SO" -lm -lpthread -Wl,-rpath,"$SRC/.libs"
cp -f "$WISDIR/fftw_wisdom_probe" "$HERE/fftw_wisdom_probe" 2>/dev/null || true

echo "=== 3) cold (no wisdom): ESTIMATE vs MEASURE ===" | tee -a "$LOG"
"$WISDIR/fftw_wisdom_probe" NONE $SIZES 2>&1 | tee -a "$LOG"

echo "=== 4) with MEASURE wisdom file ===" | tee -a "$LOG"
"$WISDIR/fftw_wisdom_probe" "$WISFILE" $SIZES 2>&1 | tee -a "$LOG"

echo "=== 5) FFTW bench sanity (estimate vs default=MEASURE) @ 4096 ===" | tee -a "$LOG"
cd "$SRC/tests"
raw_e=$(LD_LIBRARY_PATH="$SRC/.libs" ./bench --report-mflops -oestimate -t 1.0 -s 4096 2>&1)
raw_m=$(LD_LIBRARY_PATH="$SRC/.libs" ./bench --report-mflops -t 1.0 -s 4096 2>&1)
echo "bench estimate: $raw_e" | tee -a "$LOG"
echo "bench MEASURE:  $raw_m" | tee -a "$LOG"

echo "DONE log=$LOG wisdom=$WISFILE" | tee -a "$LOG"
