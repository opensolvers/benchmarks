#!/usr/bin/env bash
# A/B SoA LBM-style microbench: GCC AUTOVEC (-ftree-vectorize) vs NOVEC (-fno-tree-vectorize)
# on Orange Pi (rv64gcv). Plain double SoA — no walberla::simd.
set -e
# Avoid pipefail: `cmd | head` exits 141 under pipefail on this board image.
export PS1="${PS1:-}"

ROOT="${ROOT:-/home/orangepi/walberla-bench}"
LOCAL_CPP="${LOCAL_CPP:-$ROOT/microbench/soa_lbm_autovec.cpp}"
OUT_DIR="${OUT_DIR:-$ROOT/results/soa-autovec}"
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

CXXFLAGS_COMMON=(-std=c++20 -O2 -march=rv64gcv -mabi=lp64d -fno-math-errno)
echo "g++ $(g++ --version | head -1)"
echo "SRC=$LOCAL_CPP"
echo "OUT_DIR=$OUT_DIR"

build_one() {
  local name="$1"
  shift
  local bin="$OUT_DIR/soa_lbm_$name"
  echo "=== BUILD $name ===" >&2
  set +e
  # Vec reports go to stderr; keep compile errors separate from opt-info.
  g++ "${CXXFLAGS_COMMON[@]}" "$@" "$LOCAL_CPP" -o "$bin" \
    2>"$OUT_DIR/compile_$name.err"
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

# Explicit vectorize / no-vectorize; capture opt-info for the hot loop.
# Note: GCC rejects two -fopt-info-*=FILE in one invocation — use one file each.
BIN_AUTO=$(build_one autovec \
  -ftree-vectorize \
  -fopt-info-vec-optimized="$OUT_DIR/vec_optimized_autovec.txt")
# Second pass for missed (overwrite binary is fine; we already have the first).
echo "=== BUILD autovec (vec-missed report) ===" >&2
g++ "${CXXFLAGS_COMMON[@]}" -ftree-vectorize \
  -fopt-info-vec-missed="$OUT_DIR/vec_missed_autovec.txt" \
  "$LOCAL_CPP" -o "$OUT_DIR/soa_lbm_autovec" 2>"$OUT_DIR/compile_autovec_missed.err" || true
BIN_AUTO="$OUT_DIR/soa_lbm_autovec"

BIN_NOVEC=$(build_one novec \
  -fno-tree-vectorize \
  -fopt-info-vec-all="$OUT_DIR/vec_all_novec.txt")

run_one() {
  local name="$1"
  local bin="$2"
  local log="$OUT_DIR/run_$name.log"
  echo "=== RUN $name ==="
  set +e
  "$bin" --nx "$NX" --iters "$ITERS" --reps "$REPS" >"$log" 2>"$OUT_DIR/time_$name.log"
  local rc=$?
  set -e
  echo "RUN_RC=$rc name=$name"
  cat "$log"
  [[ -s "$OUT_DIR/time_$name.log" ]] && cat "$OUT_DIR/time_$name.log"
}

run_one autovec "$BIN_AUTO"
run_one novec "$BIN_NOVEC"

dump_hot() {
  local name="$1"
  local bin="$2"
  local dump="$OUT_DIR/objdump_$name.txt"
  local full="$OUT_DIR/objdump_${name}_full.txt"
  echo "=== OBJDUMP $name (collideSoA) ==="
  objdump -d "$bin" 2>/dev/null | c++filt >"$full"
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
  {
    echo "name=$name"
    echo "hot_dump=$dump"
    echo "hot_lines=$(wc -l <"$dump")"
    echo "hot_vle64=$vle hot_vse64=$vse hot_vfadd=$vfadd hot_vfmul=$vfmul hot_vfmacc=$vfmacc hot_vfsqrt=$vfsqrt"
  } | tee "$OUT_DIR/objdump_${name}_summary.txt"
  echo "--- first 40 lines of hot dump ---"
  head -40 "$dump" || true
  echo "--- sample RVV ops in hot (if any) ---"
  grep -E '[[:space:]](vle64|vse64|vfadd|vfmul|vfmacc|vfmadd|vfsqrt|vsetvli|vsetivli)' "$dump" | head -30 || echo "(none)"
}

dump_hot autovec "$BIN_AUTO"
dump_hot novec "$BIN_NOVEC"

# Vec-report excerpts mentioning collideSoA / the source file
{
  echo "===== vec-optimized (autovec) — collideSoA / soa_lbm_autovec ====="
  cat "$OUT_DIR/vec_optimized_autovec.txt" 2>/dev/null || true
  echo
  echo "===== vec-missed (autovec) — hot-loop sample ====="
  grep -E 'soa_lbm_autovec\.cpp:(59|62|63|103)' "$OUT_DIR/vec_missed_autovec.txt" 2>/dev/null | head -40 || true
  echo
  echo "===== novec: -fno-tree-vectorize (no vec-optimized report expected) ====="
} | tee "$OUT_DIR/vec_report_excerpt.txt"

SUMMARY="$OUT_DIR/summary.txt"
{
  echo "soa_lbm_autovec A/B — GCC AUTOVEC vs NOVEC (plain double SoA)"
  echo "date=$(date -Iseconds)"
  echo "host=$(hostname)"
  echo "nx=$NX iters=$ITERS reps=$REPS"
  echo "cxxflags_common=${CXXFLAGS_COMMON[*]}"
  echo "autovec_extra=-ftree-vectorize"
  echo "novec_extra=-fno-tree-vectorize"
  echo "source=$LOCAL_CPP"
  echo
  echo "----- AUTOVEC run -----"
  cat "$OUT_DIR/run_autovec.log"
  echo
  echo "----- NOVEC run -----"
  cat "$OUT_DIR/run_novec.log"
  echo
  echo "----- objdump -----"
  cat "$OUT_DIR/objdump_autovec_summary.txt"
  cat "$OUT_DIR/objdump_novec_summary.txt"
  echo
  auto_w=$(awk -F= '/BEST_WALL_s=/{print $2}' "$OUT_DIR/run_autovec.log" | tail -1)
  nov_w=$(awk -F= '/BEST_WALL_s=/{print $2}' "$OUT_DIR/run_novec.log" | tail -1)
  auto_m=$(awk -F= '/^MLUPS=/{print $2}' "$OUT_DIR/run_autovec.log" | tail -1)
  nov_m=$(awk -F= '/^MLUPS=/{print $2}' "$OUT_DIR/run_novec.log" | tail -1)
  echo "BEST_WALL_autovec_s=$auto_w MLUPS_autovec=$auto_m"
  echo "BEST_WALL_novec_s=$nov_w MLUPS_novec=$nov_m"
  echo "PRIOR_simd_vector_size_WALL_s=14.595330"
  echo "PRIOR_FORCE_SCALAR_autovec_WALL_s=8.335790"
  echo "PRIOR_POD_RVV_WALL_s=22.802704"
  if command -v python3 >/dev/null 2>&1 && [[ -n "$auto_w" && -n "$nov_w" ]]; then
    python3 - <<PY
auto=float("$auto_w"); nov=float("$nov_w")
simd=14.595330; sc=8.335790; pod=22.802704
print(f"speedup_autovec_over_novec={nov/auto:.4f}x" if auto>0 else "n/a")
print(f"speedup_autovec_vs_simd={simd/auto:.4f}x" if auto>0 else "n/a")
print(f"speedup_autovec_vs_FORCE_SCALAR={sc/auto:.4f}x" if auto>0 else "n/a")
print(f"speedup_autovec_vs_POD_RVV={pod/auto:.4f}x" if auto>0 else "n/a")
PY
  fi
} | tee "$SUMMARY"

echo "SUMMARY=$SUMMARY"
