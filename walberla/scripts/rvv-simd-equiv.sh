#!/usr/bin/env bash
# Local RVV vs Scalar equivalence harness (does not need waLBerla tests enabled).
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
OUT=/tmp/rvv_simd_equiv
mkdir -p "$OUT"

cat > "$OUT/equiv.cpp" << 'EOF'
#include "simd/SIMD.h"
#include "simd/Scalar.h"
#include <cmath>
#include <cstdio>
#include <cstring>

namespace R = walberla::simd;          // active backend (RVV)
namespace S = walberla::simd::scalar;  // reference

static bool near(double a, double b, double eps = 1e-12) {
   if (std::isnan(a) && std::isnan(b)) return true;
   return std::fabs(a - b) <= eps * (1.0 + std::fabs(b));
}

static bool sameNumeric(const R::double4_t& a, const S::double4_t& b, const char* tag) {
   for (int i = 0; i < 4; ++i) {
      double av = R::getComponent(a, i);
      double bv = S::getComponent(b, i);
      if (!near(av, bv)) {
         std::printf("MISMATCH %s lane %d: rvv=%g scalar=%g\n", tag, i, av, bv);
         return false;
      }
   }
   return true;
}

static bool sameMask(const R::double4_t& a, const S::double4_t& b, const char* tag) {
   // Compare masks via bool lanes / movemask — bit patterns differ (all-ones vs NaN).
   for (int i = 0; i < 4; ++i) {
      bool ab = R::getBoolComponent(a, i);
      bool bb = S::getBoolComponent(b, i);
      if (ab != bb) {
         std::printf("MISMATCH %s bool lane %d: rvv=%d scalar=%d\n", tag, i, (int)ab, (int)bb);
         return false;
      }
   }
   if (R::movemask(a) != S::movemask(b)) {
      std::printf("MISMATCH %s movemask: rvv=%d scalar=%d\n", tag, R::movemask(a), S::movemask(b));
      return false;
   }
   return true;
}

int main() {
   const char* isa = R::usedInstructionSet();
   std::printf("usedInstructionSet=%s\n", isa);
   if (!isa || std::strncmp(isa, "RVV", 3) != 0) {
      std::printf("FAIL: not RVV\n");
      return 1;
   }

   R::double4_t a = R::make_double4(4.0, 3.0, 2.0, 1.0); // mem [1,2,3,4]
   R::double4_t b = R::make_double4(8.0, 7.0, 6.0, 5.0); // mem [5,6,7,8]
   S::double4_t sa = S::make_double4(4.0, 3.0, 2.0, 1.0);
   S::double4_t sb = S::make_double4(8.0, 7.0, 6.0, 5.0);

   if (!sameNumeric(a + b, sa + sb, "add")) return 2;
   if (!sameNumeric(a - b, sa - sb, "sub")) return 3;
   if (!sameNumeric(a * b, sa * sb, "mul")) return 4;
   if (!sameNumeric(a / b, sa / sb, "div")) return 5;

   if (!sameNumeric(R::sqrt(a), S::sqrt(sa), "sqrt")) return 6;
   if (!sameNumeric(R::hadd(a, b), S::hadd(sa, sb), "hadd")) return 7;
   if (!sameNumeric(R::horizontalSum(a), S::horizontalSum(sa), "hsum")) return 8;
   if (!sameNumeric(R::rotateLeft(a), S::rotateLeft(sa), "rotL")) return 9;
   if (!sameNumeric(R::rotateRight(a), S::rotateRight(sa), "rotR")) return 10;
   if (!sameNumeric(R::exchangeLowerUpperHalf(a), S::exchangeLowerUpperHalf(sa), "xchg")) return 11;

   if (!sameMask(R::compareEQ(a, a), S::compareEQ(sa, sa), "eq")) return 12;
   if (!sameMask(R::compareNEQ(a, b), S::compareNEQ(sa, sb), "neq")) return 13;
   if (!sameMask(R::compareGE(a, b), S::compareGE(sa, sb), "ge")) return 14;
   if (!sameMask(R::compareLE(a, b), S::compareLE(sa, sb), "le")) return 15;

   auto mask = R::compareGE(a, R::make_double4(2.5));
   auto smask = S::compareGE(sa, S::make_double4(2.5));
   if (!sameMask(mask, smask, "ge_mask")) return 16;
   if (!sameNumeric(R::blendv(a, b, mask), S::blendv(sa, sb, smask), "blendv")) return 17;
   if (!sameNumeric(R::blend<0b1010>(a, b), S::blend<0b1010>(sa, sb), "blend")) return 18;

   // load/store roundtrip (RVV only — Scalar load brace-init is historically odd)
   alignas(32) double mem[4] = {1.5, 2.5, 3.5, 4.5};
   alignas(32) double out[4] = {};
   R::store_aligned(out, R::load_aligned(mem));
   for (int i = 0; i < 4; ++i) {
      if (!near(out[i], mem[i])) {
         std::printf("MISMATCH load/store lane %d\n", i);
         return 19;
      }
   }

   std::printf("EQUIV_OK\n");
   return 0;
}
EOF

INCS=(-I"$SRC/src" -I"$BUILD/src")
set +e
g++ -std=c++20 -O2 -march=rv64gcv -mabi=lp64d "${INCS[@]}" \
  "$OUT/equiv.cpp" -o "$OUT/equiv" 2>"$OUT/compile.err"
rc=$?
set -e
if [ $rc -ne 0 ]; then
  echo "EQUIV_COMPILE_FAIL"
  cat "$OUT/compile.err"
  exit $rc
fi
"$OUT/equiv" | tee "$OUT/equiv.log"
