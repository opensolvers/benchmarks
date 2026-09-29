#!/usr/bin/env bash
# Build + run tiny radix-2 FFT × TCM A/B on RV2 (cluster 0).
# Needs: group tcm, /dev/tcm, GCC 14 (EESSI).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
IME="${IME_DIR:-$HERE/../ime}"
OUT="${OUT:-$HERE/tiny-fft-tcm-bench}"
LOGDIR="${LOGDIR:-$HOME/logs}"
mkdir -p "$LOGDIR"

GCC14="${GCC14:-$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0}"
if [[ ! -x "$GCC14/bin/gcc" ]]; then
  GCC14=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0
fi
export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$GCC14/lib64:${LD_LIBRARY_PATH:-}"
CC="${CC:-$GCC14/bin/gcc}"

echo "Building $OUT with $CC ..."
"$CC" -O3 -std=c11 -Wall -Wextra -march=rv64gcv_zvl256b \
  -I"$IME" -B"$GCC14/bin" -L"$GCC14/lib64" -Wl,-rpath,"$GCC14/lib64" \
  "$HERE/bench_tiny_fft_tcm.c" "$IME/tcm.c" -lm -o "$OUT"

ts=$(date +%Y%m%d-%H%M%S)
log="$LOGDIR/tiny-fft-tcm-ab-$ts.log"
exec > >(tee -a "$log") 2>&1
echo "LOG=$log"
ls -la /dev/tcm 2>&1 || true

run() {
  echo
  echo "===== N=$* ====="
  taskset -c 0 "$OUT" "$@"
}

# Working set ≈ 24·N bytes (work + twiddles; signal filled in-place)
run 256 200
run 1024 100
run 4096 80
run 8192 60
run 16384 40   # ~384 KiB — fits after dropping resident src

echo "DONE"
