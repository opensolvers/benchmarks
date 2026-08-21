#!/bin/bash
# Voro++ RVV auto-vec A/B on SpaceMiT X60:
#   novec  = -O3 -march=rv64gc  -fno-tree-vectorize
#   gcv    = -O3 -march=rv64gcv -ftree-vectorize
#
# Rebuilds libvoro++ from upstream 0.4.6 via its Makefile (stock EESSI is
# rv64gc only). Usage: VORO_N=20000 VORO_REPS=5 bash run-voro-autovec-ab.sh
set +e
LOG="${LOG:-$HOME/logs/voro-autovec-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export EESSI_VERSION_OVERRIDE=2025.06-001
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load GCCcore/14.3.0
echo "CXX=$(command -v g++)"
g++ --version | head -1

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HOME/voro-ab}
SRC_TGZ=${SRC_TGZ:-$WORK/voro++-0.4.6.tar.gz}
URL=${URL:-https://math.lbl.gov/voro++/download/dir/voro++-0.4.6.tar.gz}
N=${VORO_N:-20000}
REPS=${VORO_REPS:-5}
SEED=${VORO_SEED:-42}

mkdir -p "$WORK"
cd "$WORK"

if [[ ! -f "$SRC_TGZ" ]]; then
  echo "Downloading $URL"
  wget -q -O "$SRC_TGZ" "$URL" || curl -fsSL -o "$SRC_TGZ" "$URL"
fi

NOVEC_FLAGS="-Wall -O3 -march=rv64gc -mabi=lp64d -fno-tree-vectorize -fno-math-errno"
GCV_FLAGS="-Wall -O3 -march=rv64gcv -mabi=lp64d -ftree-vectorize -fno-math-errno"

build_variant() {
  local tag=$1
  local flags=$2
  local tree="$WORK/tree-$tag"
  echo "==== build [$tag] CFLAGS=$flags ===="
  rm -rf "$tree"
  mkdir -p "$tree"
  tar -xzf "$SRC_TGZ" -C "$tree" --strip-components=0
  # tarball extracts voro++-0.4.6/
  local src="$tree/voro++-0.4.6"
  # Only build the library (skip examples)
  ( cd "$src/src" && make clean && make libvoro++.a CXX=g++ CFLAGS="$flags" -j1 )
  echo "lib size: $(wc -c <"$src/src/libvoro++.a")"
  echo "---- link bench [$tag] ----"
  # shellcheck disable=SC2086
  g++ $flags -I"$src/src" -o "$WORK/bench_tess-$tag" "$HERE/bench_tess.cc" "$src/src/libvoro++.a" -lm
  file "$WORK/bench_tess-$tag"
}

build_variant novec "$NOVEC_FLAGS"
build_variant gcv   "$GCV_FLAGS"

echo ""
echo "---- objdump RVV-ish insn counts in cell.o ----"
for tag in novec gcv; do
  o="$WORK/tree-$tag/voro++-0.4.6/src/cell.o"
  if [[ -f "$o" ]]; then
    n=$(objdump -d "$o" 2>/dev/null | grep -cE '\bv(setvli|le[0-9]|se[0-9]|fadd|fmul|fmadd|fsub|fmacc|fred)' || true)
    echo "[$tag] cell.o matches: $n"
  fi
done

run_one() {
  local tag=$1
  echo ""
  echo "========== [$tag] N=$N REPS=$REPS =========="
  "$WORK/bench_tess-$tag" "$N" "$REPS" "$SEED"
  echo "RC=$?"
}

run_one novec
run_one gcv
echo "DONE $(date -Iseconds)"
