#!/bin/bash
# Compare EESSI Espresso vs ~/espresso-opt on dense_large / lattice512.
# Env: ESP_CASE=dense_large|lattice512|both  ESP_REPS=N  ESP_STEPS=...
set +e
LOG="${LOG:-$HOME/logs/espresso-opt-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds)"

export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL="${EESSI_USER_INSTALL:-$HOME/eessi-overlay}"
export EESSI_NO_MODULE_PURGE_ON_INIT=1
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load foss/2025b 2>/dev/null || true

HERE=$(cd "$(dirname "$0")" && pwd)
SCALAR=${SCALAR:-$HOME/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=RISCV64_GENERIC
export LD_PRELOAD="$SCALAR"

CASE=${ESP_CASE:-both}
REPS=${ESP_REPS:-1}
OPT=${OPT:-$HOME/espresso-opt}

setup_case() {
  local c=$1
  case "$c" in
    dense_large)
      export ESP_BOX=24 ESP_DENSITY=0.4 ESP_STEPS=${ESP_STEPS:-200} ESP_ACCURACY=1e-3
      export ESP_MESH=32 ESP_CAO=6 ESP_RCUT=3.4847 ESP_ALPHA=0.751886 ESP_TUNE=0
      SCRIPT=$HERE/mpi_dense_large.py
      ;;
    lattice512)
      export ESP_N=512 ESP_BOX=20 ESP_STEPS=${ESP_STEPS:-600} ESP_ACCURACY=1e-3
      export ESP_MESH=18 ESP_CAO=5 ESP_RCUT=4.4896 ESP_ALPHA=0.53745 ESP_TUNE=0
      # unset dense-only knobs
      unset ESP_DENSITY
      SCRIPT=$HERE/p3m_lj.py
      ;;
    *)
      echo "unknown case: $c" >&2
      return 1
      ;;
  esac
  echo "CASE=$c SCRIPT=$SCRIPT STEPS=$ESP_STEPS REPS=$REPS"
}

run_one() {
  local tag=$1
  local py=$2
  local c=$3
  local rep=$4
  echo ""
  echo "========== [$tag] case=$c rep=$rep/$REPS pypresso=$py =========="
  if [ "$rep" = 1 ]; then
    "$py" -c "import espressomd; print('features', sorted(espressomd.features()))" 2>/dev/null | tail -1
  fi
  "$py" "$SCRIPT"
  echo "RC_${tag}_${c}_r${rep}=$?"
}

run_case() {
  local c=$1
  setup_case "$c" || return 1

  module load ESPResSo/4.2.2-foss-2025b 2>/dev/null || module load ESPResSo/4.2.2-foss-2025b
  local eessi_py
  eessi_py="$(command -v pypresso)"

  # shellcheck disable=SC1090
  [ -f "$HOME/espresso-opt-env.sh" ] && . "$HOME/espresso-opt-env.sh"
  export PATH="$OPT/bin:$PATH"
  local pyver
  pyver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  export PYTHONPATH="$OPT/lib/python${pyver}/site-packages:${PYTHONPATH:-}"

  local r
  for r in $(seq 1 "$REPS"); do
    # Prefer EESSI pypresso via absolute path so PATH prepend for opt does not steal it.
    run_one eessi "$eessi_py" "$c" "$r"
    run_one opt "$OPT/bin/pypresso" "$c" "$r"
  done
}

cases=()
if [ "$CASE" = both ]; then
  cases=(dense_large lattice512)
else
  cases=("$CASE")
fi

for c in "${cases[@]}"; do
  # lattice512 uses its own default steps unless ESP_STEPS was set for a single case
  if [ "$CASE" = both ]; then
    unset ESP_STEPS
  fi
  run_case "$c"
done

echo "DONE $(date -Iseconds)"
echo "LOG=$LOG"
