#!/bin/bash
# OpenBLAS microbench + correctness A/B on RV2 (FlexiBLAS).
set +e
LOG="${LOG:-$HOME/logs/openblas-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export LMOD_IGNORE_CACHE=yes
export EESSI_VERSION_OVERRIDE=2025.06-001
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module --ignore_cache load FlexiBLAS/3.4.5-GCC-14.3.0

HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${OB_SRC:-$HERE}
BIN=${OB_BIN:-$HOME/openblas-bench}
RVV_LIB=${RVV_LIB:-$HOME/libopenblas_x60_eb_fixed.so}
N=${DGEMM_N:-2048}
THREADS=${THREADS:-8}

mkdir -p "$BIN"
cp -f "$SRC"/bench_dgemm.c "$SRC"/difftest.c "$BIN"/
gcc -O2 "$BIN/bench_dgemm.c" -o "$BIN/bench_dgemm" -lflexiblas
gcc -O2 "$BIN/difftest.c" -o "$BIN/difftest" -ldl -lm
echo "build_rc=$?"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

# resolve stock .so from FlexiBLAS default backend
STOCK_SO=$(python3 -c "
import ctypes, os
os.environ.setdefault('FLEXIBLAS','OPENBLAS')
# ldd flexiblas path
" 2>/dev/null)
STOCK_SO=$(ldd "$(command -v bench_dgemm 2>/dev/null || echo /dev/null)" 2>/dev/null | awk '/libopenblas/{print $3; exit}')
if [[ -z "$STOCK_SO" || ! -f "$STOCK_SO" ]]; then
  STOCK_SO=$(find /cvmfs/dev.eessi.io/riscv -path '*/OpenBLAS/*/lib/libopenblas.so' 2>/dev/null | head -1)
fi
echo "STOCK_SO=$STOCK_SO"

run_dgemm() {
  local tag=$1
  shift
  echo ""
  echo "========== DGEMM [$tag] N=$N env: $* =========="
  env "$@" "$BIN/bench_dgemm" "$N"
}

run_dgemm scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
run_dgemm stock  FLEXIBLAS=OPENBLAS
if [[ -f "$RVV_LIB" ]]; then
  run_dgemm patched FLEXIBLAS="$RVV_LIB"
fi

echo ""
echo "========== difftest correctness =========="
if [[ -n "$STOCK_SO" && -f "$STOCK_SO" ]]; then
  echo "--- stock $STOCK_SO ---"
  "$BIN/difftest" "$STOCK_SO" | grep -E 'dgemv|dgemm|dtrsm|nan'
fi
if [[ -f "$RVV_LIB" ]]; then
  echo "--- patched $RVV_LIB ---"
  "$BIN/difftest" "$RVV_LIB" | grep -E 'dgemv|dgemm|dtrsm|nan'
fi
echo "--- scalar via env ---"
OPENBLAS_CORETYPE=RISCV64_GENERIC FLEXIBLAS=OPENBLAS "$BIN/difftest" "${STOCK_SO:-$RVV_LIB}" 2>/dev/null | grep -E 'dgemv|dgemm|dtrsm|nan' || true

echo "DONE $(date -Iseconds)"
