#!/bin/sh
# A100 IME2 vs RVV on the shapes reported in the README.
# The A100 is fenced: set_ai_thread must happen before exec of the bench,
# or ld.so caches the X100 VLEN and taskset -c 8 returns EINVAL.
set -eu
cd "$(dirname "$0")"
BIN=${BIN:-./ime2-bench}
REPS=${REPS:-15}

if [ ! -x "$BIN" ]; then
  echo "missing $BIN — run: make a100" >&2
  exit 1
fi

a100() {
  bin=$1
  shift
  # shellcheck disable=SC2016
  sh -c 'echo $$ > /proc/set_ai_thread && exec taskset -c 8 "$0" "$@"' "$bin" "$@"
}

echo "=== A100 core 8 ==="
a100 "$BIN" 512 512 512 "$REPS" 1
a100 "$BIN" 768 768 512 "$REPS" 1
a100 "$BIN" 1024 1024 512 "$REPS" 1

echo "=== X100 core 0, RVV only ==="
taskset -c 0 "$BIN" 512 512 512 8 0
taskset -c 0 "$BIN" 768 768 512 8 0
taskset -c 0 "$BIN" 1024 1024 512 8 0
