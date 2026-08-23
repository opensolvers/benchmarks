#!/bin/bash
# Fast incremental rebuild of llama.cpp IME with i8i8 + piped INNER (no full EasyBuild).
# Produces ~/llama-pipe/{lib,bin} for A/B vs stock -x60-ime-q8_0.
set -eo pipefail
ROOT="$HOME/llama-pipe"
SRC="$ROOT/src"
BUILD="$ROOT/build"
ASDIR="$ROOT/xsmtvdot-as-only"
TAR="$HOME/x60-work/easybuild/sources/l/llama.cpp/ad8d821.tar.gz"
EC_PATCHES="$HOME/x60-ec"
PIPE_PATCH="$HOME/llama-ime1-i8i8-pipe.patch"
SMT_AS="$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/binutils/2.46.1-xsmtvdot/bin/as"

export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_NO_MODULE_PURGE_ON_INIT="${EESSI_NO_MODULE_PURGE_ON_INIT:-}"
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load foss/2025b CMake/3.31.8-GCCcore-14.3.0 cURL/8.14.1-GCCcore-14.3.0

rm -rf "$SRC" "$BUILD"
mkdir -p "$ROOT" "$ASDIR"
ln -sfn "$SMT_AS" "$ASDIR/as"
# Do NOT put -march in global C/CXXFLAGS — ggml-cpu sets its own
# rv64gcv_zfh_zvfh_…_zba march via ARCH_FLAGS. A global -march without zba
# overrides that and breaks ime.cpp (#error GGML_RV_ZBA).
export CFLAGS="${CFLAGS:--O2 -ftree-vectorize -fno-math-errno} -B${ASDIR}/"
export CXXFLAGS="${CXXFLAGS:--O2 -ftree-vectorize -fno-math-errno} -B${ASDIR}/"
# Strip any -march= from module-provided flags
CFLAGS="$(echo "$CFLAGS" | sed -E 's/-march=[^ ]+//g')"
CXXFLAGS="$(echo "$CXXFLAGS" | sed -E 's/-march=[^ ]+//g')"
export CFLAGS CXXFLAGS

mkdir -p "$SRC"
tar -xzf "$TAR" -C "$SRC" --strip-components=1
cd "$SRC"
patch -p1 < "$EC_PATCHES/llama.cpp-x60-ime-upstream-binutils.patch"
patch -p1 < "$EC_PATCHES/llama-ime1-scalebuild-opt.patch"
patch -p1 < "$EC_PATCHES/llama-ime1-q8_0-i8i8.patch"
python3 - "$SRC/ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
t = p.read_text()
macro = r'''
// Software-pipelined INNER (ime-bench lesson): overlap marching B/A loads with
// vmadot. Full dual-bank needs ~20 operand regs; FP acc lives in v24..v31, so
// interleave on one bank (sequential pointer).
#define IME1_I8I8_INNER_PIPED                   \
    "vsetvli      t0, zero, e8, m1        \n\t" \
    "vle8.v       v2, (s1)               \n\t" \
    "addi         s1, s1, 32              \n\t" \
    "vle8.v       v3, (s1)               \n\t" \
    "addi         s1, s1, 32              \n\t" \
    "vle8.v       v4, (s1)               \n\t" \
    "addi         s1, s1, 32              \n\t" \
    "vle8.v       v5, (s1)               \n\t" \
    "addi         s1, s1, 32              \n\t" \
    "vle8.v       v10, (a1)              \n\t" \
    "addi         a1, a1, 32              \n\t" \
    "smt.vmadot       v16, v10, v2            \n\t" \
    "vle8.v       v6, (s1)               \n\t" \
    "addi         s1, s1, 32              \n\t" \
    "smt.vmadot       v18, v10, v3            \n\t" \
    "vle8.v       v7, (s1)               \n\t" \
    "addi         s1, s1, 32              \n\t" \
    "smt.vmadot       v20, v10, v4            \n\t" \
    "vle8.v       v8, (s1)               \n\t" \
    "addi         s1, s1, 32              \n\t" \
    "smt.vmadot       v22, v10, v5            \n\t" \
    "vle8.v       v9, (s1)               \n\t" \
    "addi         s1, s1, 32              \n\t" \
    "vle8.v       v11, (a1)              \n\t" \
    "addi         a1, a1, 32              \n\t" \
    "smt.vmadot       v16, v11, v6            \n\t" \
    "smt.vmadot       v18, v11, v7            \n\t" \
    "smt.vmadot       v20, v11, v8            \n\t" \
    "smt.vmadot       v22, v11, v9            \n\t"

'''
anchor = '// v2..v9 <- 8 contiguous 32B B tiles (slice-lo groups 0..3, slice-hi groups 0..3)'
if 'IME1_I8I8_INNER_PIPED' not in t:
    if anchor not in t:
        raise SystemExit('anchor missing')
    t = t.replace(anchor, macro + anchor, 1)
old = '''            LOAD_B_16x8x2_I8

            "vsetvli            t0, zero, e8, m1            \\n\\t"
            "vle8.v             v10, (a1)                   \\n\\t"
            "addi               a1, a1, 32                  \\n\\t"
            "vle8.v             v11, (a1)                   \\n\\t"
            "addi               a1, a1, 32                  \\n\\t"

            SQ4BIT_KERNEL_COMP_4x16x16
'''
new = '            IME1_I8I8_INNER_PIPED\n'
if old not in t:
    raise SystemExit('INNER block not found for replacement')
t = t.replace(old, new, 1)
p.write_text(t)
print('piped INNER applied')
PY

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_RVV=ON \
  -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZVFH=ON \
  -DGGML_RV_ZBA=ON \
  -DGGML_RV_ZICBOP=ON \
  -DGGML_CPU_RISCV64_SPACEMIT=ON \
  -DRISCV64_SPACEMIT_IME_SPEC=RISCV64_SPACEMIT_IME1 \
  -DGGML_BLAS=OFF \
  -DCMAKE_C_FLAGS="$CFLAGS" \
  -DCMAKE_CXX_FLAGS="$CXXFLAGS"

cmake --build "$BUILD" -j"$(nproc)" --target ggml-cpu llama-bench llama-cli
mkdir -p "$ROOT/lib" "$ROOT/bin"
cp -a "$BUILD"/bin/llama-bench "$BUILD"/bin/llama-cli "$ROOT/bin/" 2>/dev/null || \
  cp -a "$BUILD"/tools/llama-bench/llama-bench "$BUILD"/tools/llama-cli/llama-cli "$ROOT/bin/"
# libs live under build/bin or build/ggml/src
find "$BUILD" -name 'libggml*.so*' -exec cp -a {} "$ROOT/lib/" \;
find "$BUILD" -name 'libllama*.so*' -exec cp -a {} "$ROOT/lib/" \;
echo "PIPE_BUILD_DONE $(date)"
ls -la "$ROOT/bin" "$ROOT/lib" | head -40
nm -D "$ROOT/lib/libggml-cpu.so" | grep i8i8 | head
