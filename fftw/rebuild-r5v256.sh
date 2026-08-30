#!/bin/bash
# Rebuild only r5v256 codelets + relink libfftw3 (other VL archives untouched).
set -euo pipefail
EESSI_SW=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software
export PATH=$EESSI_SW/GCCcore/14.3.0/bin:$PATH
export LIBRARY_PATH=$EESSI_SW/GCCcore/14.3.0/lib64
export LD_LIBRARY_PATH=$EESSI_SW/GCCcore/14.3.0/lib64
ROOT=/home/orangepi/fftwbuild/src-r5v
cd "$ROOT"

# Force remake of r5v256 objects only
find dft/simd/r5v256 rdft/simd/r5v256 -name '*.o' -delete
find dft/simd/r5v256 rdft/simd/r5v256 -name '*.lo' -delete
make -j4 -C dft/simd/r5v256
make -j4 -C rdft/simd/r5v256 2>/dev/null || true

# Prevent make from descending into other VL dirs
for d in dft/simd/r5v* rdft/simd/r5v*; do
  [ -d "$d" ] || continue
  case "$d" in */r5v256) continue ;; esac
  make -t -C "$d" >/dev/null 2>&1 || true
done

rm -f .libs/libfftw3.so .libs/libfftw3.so.3 .libs/libfftw3.so.3.6.10
make -j4
ls -la .libs/libfftw3.so.3.6.10
echo RELINK_OK
