#!/usr/bin/env bash
# A/B SoA LBM-style microbench: walberla::simd::double4_t RVV vs FORCE_SCALAR
# on Orange Pi (rv64gcv). Standalone TU — no full waLBerla rebuild.
set -e
# Avoid pipefail: `cmd | head` exits 141 under pipefail on this board image.
export PS1="${PS1:-}"

ROOT="${ROOT:-/home/orangepi/walberla-bench}"
SRC_TREE="$ROOT/src/walberla-7.2"
BUILD="$ROOT/build-gcv"
LOCAL_CPP="${LOCAL_CPP:-$ROOT/microbench/soa_lbm_simd.cpp}"
OUT_DIR="${OUT_DIR:-$ROOT/results/soa-microbench}"
NX="${NX:-262144}"
ITERS="${ITERS:-80}"
REPS="${REPS:-3}"

export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL=$HOME/eessi-overlay
export EESSI_NO_MODULE_PURGE_ON_INIT=1
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load EasyBuild/5.3.1 EESSI-extend/2025.06-easybuild foss/2025b CMake/4.0.3 Boost.MPI/1.88.0-gompi-2025b
export LD_LIBRARY_PATH="$EBROOTGCCCORE/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT_DIR" "$(dirname "$LOCAL_CPP")"
if [[ ! -f "$LOCAL_CPP" ]]; then
  echo "Missing source: $LOCAL_CPP" >&2
  exit 1
fi

DEF=$(find "$BUILD" -name 'waLBerlaDefinitions.h' 2>/dev/null | head -1)
INCS=(-I"$SRC_TREE/src" -I"$BUILD/src")
[[ -n "$DEF" ]] && INCS+=(-I"$(dirname "$DEF")")

CXXFLAGS_COMMON=(-std=c++20 -O2 -march=rv64gcv -mabi=lp64d -fno-math-errno)
echo "g++ $(g++ --version | head -1)"
echo "DEF=$DEF"
echo "SRC=$LOCAL_CPP"

build_one() {
  local name="$1"
  shift
  local bin="$OUT_DIR/soa_lbm_$name"
  echo "=== BUILD $name ===" >&2
  set +e
  g++ "${CXXFLAGS_COMMON[@]}" "$@" "${INCS[@]}" "$LOCAL_CPP" -o "$bin" 2>"$OUT_DIR/compile_$name.err"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "COMPILE_FAIL $name rc=$rc" >&2
    cat "$OUT_DIR/compile_$name.err" >&2
    exit $rc
  fi
  echo "COMPILE_OK $name -> $bin" >&2
  printf '%s\n' "$bin"
}

BIN_RVV=$(build_one rvv)
BIN_SCALAR=$(build_one scalar -DWALBERLA_SIMD_FORCE_SCALAR)

run_one() {
  local name="$1"
  local bin="$2"
  local log="$OUT_DIR/run_$name.log"
  echo "=== RUN $name ==="
  set +e
  # Board image may lack GNU time; binary prints its own WALL via chrono.
  "$bin" --nx "$NX" --iters "$ITERS" --reps "$REPS" >"$log" 2>"$OUT_DIR/time_$name.log"
  local rc=$?
  set -e
  echo "RUN_RC=$rc name=$name"
  cat "$log"
  [[ -s "$OUT_DIR/time_$name.log" ]] && cat "$OUT_DIR/time_$name.log"
}

run_one rvv "$BIN_RVV"
run_one scalar "$BIN_SCALAR"

# Objdump evidence: RVV should show vector loads/FP ops; scalar should not (or far fewer).
dump_hot() {
  local name="$1"
  local bin="$2"
  local dump="$OUT_DIR/objdump_$name.txt"
  local full="$OUT_DIR/objdump_${name}_full.txt"
  echo "=== OBJDUMP $name (collideSoA) ==="
  objdump -d "$bin" 2>/dev/null | c++filt >"$full"
  # Extract collideSoA function body (definition line, not call sites in main).
  awk '
    /^[0-9a-f]+ <.*collideSoA.*>:/ {p=1}
    p {print}
    p && NR>1 && /^[0-9a-f]+ <.*>:/ && $0 !~ /collideSoA/ {exit}
  ' "$full" >"$dump"
  local vle vse vfadd vfmul vfmacc vfsqrt
  vle=$(grep -cE '[[:space:]]vle64' "$dump" 2>/dev/null || true)
  vse=$(grep -cE '[[:space:]]vse64' "$dump" 2>/dev/null || true)
  vfadd=$(grep -cE '[[:space:]]vfadd' "$dump" 2>/dev/null || true)
  vfmul=$(grep -cE '[[:space:]]vfmul' "$dump" 2>/dev/null || true)
  vfmacc=$(grep -cE '[[:space:]]vfmacc|[[:space:]]vfmadd' "$dump" 2>/dev/null || true)
  vfsqrt=$(grep -cE '[[:space:]]vfsqrt' "$dump" 2>/dev/null || true)
  local full_vle full_vfadd
  full_vle=$(grep -cE '[[:space:]]vle64' "$full" 2>/dev/null || true)
  full_vfadd=$(grep -cE '[[:space:]]vfadd' "$full" 2>/dev/null || true)
  {
    echo "name=$name"
    echo "hot_dump=$dump"
    echo "hot_lines=$(wc -l <"$dump")"
    echo "hot_vle64=$vle hot_vse64=$vse hot_vfadd=$vfadd hot_vfmul=$vfmul hot_vfmacc=$vfmacc hot_vfsqrt=$vfsqrt"
    echo "binary_vle64_total=$full_vle binary_vfadd_total=$full_vfadd"
  } | tee "$OUT_DIR/objdump_${name}_summary.txt"
  echo "--- first 30 lines of hot dump ---"
  head -30 "$dump" || true
  echo "--- sample RVV ops in hot (if any) ---"
  grep -E '[[:space:]](vle64|vse64|vfadd|vfmul|vfmacc|vfmadd|vfsqrt)' "$dump" | head -20 || echo "(none)"
}

dump_hot rvv "$BIN_RVV"
dump_hot scalar "$BIN_SCALAR"

# Write combined results summary
SUMMARY="$OUT_DIR/summary.txt"
{
  echo "soa_lbm_simd A/B — RVV vector_size(32) vs WALBERLA_SIMD_FORCE_SCALAR"
  echo "date=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo "backend=GCC f64x4_t __attribute__((vector_size(32))) inside double4_t"
  echo "nx=$NX iters=$ITERS reps=$REPS"
  echo "cxxflags=${CXXFLAGS_COMMON[*]}"
  echo
  echo "----- RVV run -----"
  cat "$OUT_DIR/run_rvv.log"
  echo
  echo "----- SCALAR run -----"
  cat "$OUT_DIR/run_scalar.log"
  echo
  echo "----- stderr (if any) -----"
  echo -n "rvv: "; cat "$OUT_DIR/time_rvv.log" 2>/dev/null || true
  echo -n "scalar: "; cat "$OUT_DIR/time_scalar.log" 2>/dev/null || true
  echo
  echo "----- objdump -----"
  cat "$OUT_DIR/objdump_rvv_summary.txt"
  cat "$OUT_DIR/objdump_scalar_summary.txt"
  echo
  # Speedup from BEST_WALL
  rvv_w=$(awk -F= '/BEST_WALL_s=/{print $2}' "$OUT_DIR/run_rvv.log" | tail -1)
  sc_w=$(awk -F= '/BEST_WALL_s=/{print $2}' "$OUT_DIR/run_scalar.log" | tail -1)
  echo "BEST_WALL_rvv_s=$rvv_w"
  echo "BEST_WALL_scalar_s=$sc_w"
  echo "PREV_POD_RVV_BEST_WALL_s=${PREV_POD_RVV_WALL:-22.802704}"
  if command -v python3 >/dev/null 2>&1 && [[ -n "$rvv_w" && -n "$sc_w" ]]; then
    python3 - <<PY
rvv=float("$rvv_w"); sc=float("$sc_w"); old=float("${PREV_POD_RVV_WALL:-22.802704}")
print(f"speedup_vs_old_POD_RVV={old/rvv:.4f}x" if rvv>0 else "speedup=n/a")
print(f"speedup_rvv_over_scalar={sc/rvv:.4f}x" if rvv>0 else "speedup=n/a")
print(f"speedup_scalar_over_rvv={rvv/sc:.4f}x" if sc>0 else "speedup=n/a")
PY
  fi
} | tee "$SUMMARY"

echo "SUMMARY=$SUMMARY"
