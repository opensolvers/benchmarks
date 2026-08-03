#!/usr/bin/env bash
set -eo pipefail
export PS1="${PS1:-}"
export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL=$HOME/eessi-overlay
export EESSI_NO_MODULE_PURGE_ON_INIT=1
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load EasyBuild/5.3.1 EESSI-extend/2025.06-easybuild foss/2025b CMake/4.0.3 Boost.MPI/1.88.0-gompi-2025b
export LD_LIBRARY_PATH="$EBROOTGCCCORE/lib64:${LD_LIBRARY_PATH:-}"

SRC=/home/orangepi/walberla-bench/src/walberla-7.2
BUILD=/home/orangepi/walberla-bench/build-gcv
SMOKE=/tmp/rvv_simd_smoke
mkdir -p "$SMOKE"

DEF=$(find "$BUILD" -name 'waLBerlaDefinitions.h' 2>/dev/null | head -1)
echo "DEF=$DEF"

cat > "$SMOKE/smoke.cpp" << 'EOF'
#include "simd/SIMD.h"
#include <cstdio>
#include <cmath>
#include <cstring>

using namespace walberla::simd;

int main() {
   const char* isa = usedInstructionSet();
   std::printf("usedInstructionSet=%s\n", isa);
   if (!isa || isa[0] != 'R') {
      std::printf("FAIL: expected RVV-like ISA, got %s\n", isa ? isa : "(null)");
      return 1;
   }

   alignas(32) double mem[4] = {1.0, 2.0, 3.0, 4.0};
   double4_t a = load_aligned(mem);
   double4_t b = make_double4(8.0, 7.0, 6.0, 5.0); // mem [5,6,7,8]
   double4_t c = a + b;
   alignas(32) double out[4];
   store_aligned(out, c);
   for (int i = 0; i < 4; ++i) {
      double expect = mem[i] + (5.0 + i);
      if (std::fabs(out[i] - expect) > 1e-12) {
         std::printf("FAIL add lane %d: got %g expect %g\n", i, out[i], expect);
         return 2;
      }
   }

   double4_t d = a * make_double4(2.0);
   store_aligned(out, d);
   for (int i = 0; i < 4; ++i) {
      if (std::fabs(out[i] - 2.0 * mem[i]) > 1e-12) {
         std::printf("FAIL mul lane %d: got %g\n", i, out[i]);
         return 3;
      }
   }

   double4_t eq = compareEQ(a, load_aligned(mem));
   if (movemask(eq) != 0xF) {
      std::printf("FAIL movemask compareEQ=%d\n", movemask(eq));
      return 4;
   }

   double4_t s = sqrt(make_double4(16.0, 9.0, 4.0, 1.0)); // mem [1,2,3,4]
   store_aligned(out, s);
   const double sexpect[4] = {1.0, 2.0, 3.0, 4.0};
   for (int i = 0; i < 4; ++i) {
      if (std::fabs(out[i] - sexpect[i]) > 1e-12) {
         std::printf("FAIL sqrt lane %d: got %g expect %g\n", i, out[i], sexpect[i]);
         return 5;
      }
   }

   double4_t h = hadd(a, b); // r0=3, r1=11, r2=7, r3=15
   store_aligned(out, h);
   const double hexpect[4] = {3.0, 11.0, 7.0, 15.0};
   for (int i = 0; i < 4; ++i) {
      if (std::fabs(out[i] - hexpect[i]) > 1e-12) {
         std::printf("FAIL hadd lane %d: got %g expect %g\n", i, out[i], hexpect[i]);
         return 6;
      }
   }

   std::printf("SMOKE_OK\n");
   return 0;
}
EOF

INCS=(-I"$SRC/src" -I"$BUILD/src")
[ -n "$DEF" ] && INCS+=(-I"$(dirname "$DEF")")

echo "Compiling with g++ $(g++ --version | head -1)"
set +e
g++ -std=c++17 -O2 -march=rv64gcv -mabi=lp64d \
  "${INCS[@]}" \
  "$SMOKE/smoke.cpp" -o "$SMOKE/smoke" 2>"$SMOKE/compile.err"
rc=$?
set -e
if [ $rc -ne 0 ]; then
  echo "COMPILE_FAIL rc=$rc"
  cat "$SMOKE/compile.err"
  exit $rc
fi
echo "COMPILE_OK"
"$SMOKE/smoke"
echo "RUN_RC=$?"
