#!/bin/bash
# Build R5V (RVV) + scalar FFTW 3.3.10 (rdolbeau r5v-test-release-005) with
# EESSI GCC 14.3.0 + -march=rv64imafdcv_zvl256b. Double precision, shared.
# Two in-tree builds from ONE source (identical compiler/flags) so the ONLY
# variable is --enable-r5v -> clean A/B.
# Deprioritized (nice/ionice) so it does not disturb a concurrent build.
# NOTE: never pipe `source`/`module` (subshell loses env) -- EESSI gotcha #1.
#
# TARGET: Orange Pi RV2 (SpaceMiT X60 / K1), EESSI 2025.06-001.
# On the RV2, `module load` does NOT repath gcc (compat GCC 13.4.0 keeps
# winning), so the real GCC14 bindir is prepended to PATH explicitly below --
# EESSI gotcha #2.  On the Banana Pi BPI-F3 (same K1/X60 SoC) a plain
# `module load GCC/14.3.0` repaths correctly and the PATH pin is unnecessary.

export EESSI_VERSION_OVERRIDE=2025.06-001
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load GCCcore/14.3.0 2>/dev/null || true
# module load does not repath gcc on RV2 -> pin the real GCC14 binary dir.
GCC14=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0
export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$GCC14/lib64:$LD_LIBRARY_PATH"

echo "=== toolchain $(date -Is) ==="
gcc --version | head -1
GCCPATH="$(command -v gcc)"
echo "gcc=$GCCPATH"
case "$(gcc -dumpversion 2>/dev/null)" in
  14.*) echo "GCC 14.x OK" ;;
  *) echo "ERROR: wrong gcc ($GCCPATH), expected GCC 14.x"; exit 1 ;;
esac

CF="-O3 -march=rv64imafdcv_zvl256b"
ROOT=$HOME/fftwbuild
TARBALL=$HOME/fftw-r5v.tar.gz
NICE="nice -n 19 ionice -c3"

[ -f "$TARBALL" ] || { echo "ERROR: tarball missing $TARBALL"; exit 1; }
rm -rf "$ROOT"; mkdir -p "$ROOT" || exit 1
cd "$ROOT" || exit 1
tar xzf "$TARBALL" || { echo "ERROR: untar failed"; exit 1; }
mv fftw-3.3.10 src-r5v || exit 1
cp -a src-r5v src-scalar || exit 1

build(){ tag=$1; shift
  cd "$ROOT/src-$tag" || return 1
  echo "=== [$tag] configure $(date -Is) ==="
  ./configure --enable-shared --disable-fortran --disable-doc --disable-mpi "$@" \
      CC=gcc CFLAGS="$CF" > cfg.log 2>&1
  rc=$?
  echo "[$tag] configure RC=$rc (simd: $(grep -iE 'r5v|simd' config.log 2>/dev/null | grep -iE 'enabl|yes' | head -1))"
  [ $rc -eq 0 ] || { echo "[$tag] CONFIGURE FAILED; tail:"; tail -20 cfg.log; return 1; }
  echo "=== [$tag] make -j4 (niced) $(date -Is) ==="
  $NICE make -j4 > make.log 2>&1
  rc=$?
  echo "[$tag] make RC=$rc $(date -Is)"
  [ $rc -eq 0 ] || { echo "[$tag] MAKE FAILED; tail:"; tail -25 make.log; return 1; }
  lib=$(ls "$ROOT/src-$tag"/.libs/libfftw3.so.3.*.* 2>/dev/null | head -1)
  echo "[$tag] lib=$lib ($(du -h "$lib" 2>/dev/null | cut -f1))"
  echo "[$tag] bench=$(ls "$ROOT/src-$tag"/tests/bench 2>/dev/null)"
  # inline RVV sanity: count vector mnemonics (scalar~0, r5v>>0)
  if [ -n "$lib" ]; then
    v=$(objdump -d "$lib" 2>/dev/null | grep -cE 'vsetvli|vsetivli|vfmacc|vfmadd|vfmul|vfadd|vle64\.v|vse64\.v|vrgather')
    echo "[$tag] RVV_INSTR_COUNT=$v"
  fi
  return 0
}

build r5v --enable-r5v || { echo "DONE_WITH_ERROR r5v $(date -Is)"; exit 1; }
build scalar           || { echo "DONE_WITH_ERROR scalar $(date -Is)"; exit 1; }
echo "=== ALL BUILDS DONE $(date -Is) ==="
