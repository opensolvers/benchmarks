//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \file RVV.h
//! \ingroup simd
//! \brief RISC-V Vector backend for waLBerla's fixed double4_t SIMD API (GCC)
//!
//! Stores a GCC fixed-size vector (`__attribute__((vector_size(32)))`) inside
//! `double4_t` so arithmetic can stay in registers. With `-march=rv64gcv`, GCC
//! lowers GNU vector ops to RVV. Lane order matches Scalar.h / SSE:
//! make_double4(d,c,b,a) → memory [a,b,c,d] at indices [0],[1],[2],[3].
//!
//! Unlike scalable `vfloat64m2_t`, GNU vectors are valid as struct members /
//! array elements (no per-op POD spill).
//
//======================================================================================================================

#pragma once

#include "waLBerlaDefinitions.h"
#include "core/DataTypes.h"
#include "core/math/FastInvSqrt.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__riscv_vector) || defined(__riscv_v)

namespace walberla {
namespace simd {
namespace rvv {

// Fixed VL=4 × f64. Prefer this over scalable RVV types as struct members.
typedef double   f64x4_t __attribute__((vector_size(32)));
typedef long long i64x4_t __attribute__((vector_size(32)));

namespace detail {

inline constexpr uint64_t MSB = 0x8000000000000000ull;

inline i64x4_t asI64(f64x4_t v)
{
   i64x4_t r;
   __builtin_memcpy(&r, &v, sizeof(r));
   return r;
}

inline f64x4_t asF64(i64x4_t v)
{
   f64x4_t r;
   __builtin_memcpy(&r, &v, sizeof(r));
   return r;
}

inline f64x4_t load4(const double* p)
{
   f64x4_t r;
   __builtin_memcpy(&r, p, sizeof(r));
   return r;
}

inline void store4(double* p, f64x4_t v)
{
   __builtin_memcpy(p, &v, sizeof(v));
}

} // namespace detail

struct double4_t
{
   f64x4_t v;

   double4_t() = default;

   explicit double4_t(f64x4_t x) : v(x) {}

   inline double4_t operator+(const double4_t& o) const { return double4_t(v + o.v); }
   inline double4_t operator-(const double4_t& o) const { return double4_t(v - o.v); }
   inline double4_t operator*(const double4_t& o) const { return double4_t(v * o.v); }
   inline double4_t operator/(const double4_t& o) const { return double4_t(v / o.v); }
};

inline const char* usedInstructionSet() { return "RVV"; }

inline double4_t make_double4(double hi3, double hi2, double lo1, double lo0)
{
   // memory order [0]=lo0,[1]=lo1,[2]=hi2,[3]=hi3  (same as Scalar.h / SSE)
   return double4_t(f64x4_t{ lo0, lo1, hi2, hi3 });
}

inline double4_t make_double4_r(double a, double b, double c, double d)
{
   return make_double4(d, c, b, a);
}

inline double4_t make_double4(double a)
{
   return double4_t(f64x4_t{ a, a, a, a });
}

inline double4_t make_zero() { return make_double4(0.0); }

inline double4_t load_aligned(double const* m)
{
   return double4_t(detail::load4(m));
}

inline double4_t load_unaligned(double const* m)
{
   return double4_t(detail::load4(m));
}

inline void store_aligned(double* m, double4_t a)
{
   detail::store4(m, a.v);
}

inline void loadNeighbors(const double* p, double4_t& r_left, double4_t& r_center, double4_t& r_right)
{
   r_left   = load_unaligned(p - 1);
   r_center = load_aligned(p);
   r_right  = load_unaligned(p + 1);
}

inline double getComponent(const double4_t& a, int i)
{
   return a.v[i];
}

inline double getComponent(const double4_t& a, unsigned long i)
{
   return a.v[static_cast<int>(i)];
}

inline bool getBoolComponent(const double4_t& a, int i)
{
   uint64_t u;
   const double d = a.v[i];
   std::memcpy(&u, &d, sizeof(u));
   return u != 0;
}

inline bool getBoolComponent(const double4_t& a, unsigned long i)
{
   return getBoolComponent(a, static_cast<int>(i));
}

inline double4_t hadd(double4_t a, double4_t b)
{
   // matches Scalar: r0=a1+a0, r1=b1+b0, r2=a3+a2, r3=b3+b2
   return make_double4(b.v[3] + b.v[2], a.v[3] + a.v[2], b.v[1] + b.v[0], a.v[1] + a.v[0]);
}

inline double4_t horizontalSum(double4_t a)
{
   const double s = a.v[0] + a.v[1] + a.v[2] + a.v[3];
   return make_double4(s);
}

inline double4_t exchangeLowerUpperHalf(double4_t a)
{
   // Scalar: make_double4(a[1], a[0], a[3], a[2]) -> mem [a2,a3,a0,a1]
   return make_double4(a.v[1], a.v[0], a.v[3], a.v[2]);
}

inline void extract(double4_t in, double4_t& d, double4_t& c, double4_t& b, double4_t& a)
{
   a = make_double4(in.v[0]);
   b = make_double4(in.v[1]);
   c = make_double4(in.v[2]);
   d = make_double4(in.v[3]);
}

inline double4_t rotateRight(double4_t a)
{
   // Scalar: make_double4(a[0], a[3], a[2], a[1]) -> mem [a1,a2,a3,a0]
   return make_double4(a.v[0], a.v[3], a.v[2], a.v[1]);
}

inline double4_t rotateLeft(double4_t a)
{
   // Scalar: make_double4(a[2], a[1], a[0], a[3]) -> mem [a3,a0,a1,a2]
   return make_double4(a.v[2], a.v[1], a.v[0], a.v[3]);
}

// GCC float-vector compares yield a same-width signed-integer vector with
// all-ones / zero lanes — matches SSE mask bit patterns for blend/movemask.
inline double4_t compareEQ(double4_t a, double4_t b)
{
   return double4_t(detail::asF64(a.v == b.v));
}

inline double4_t compareNEQ(double4_t a, double4_t b)
{
   return double4_t(detail::asF64(a.v != b.v));
}

inline double4_t compareGE(double4_t a, double4_t b)
{
   return double4_t(detail::asF64(a.v >= b.v));
}

inline double4_t compareLE(double4_t a, double4_t b)
{
   return double4_t(detail::asF64(a.v <= b.v));
}

inline double4_t logicalAND(double4_t a, double4_t b)
{
   return double4_t(detail::asF64(detail::asI64(a.v) & detail::asI64(b.v)));
}

inline double4_t logicalOR(double4_t a, double4_t b)
{
   return double4_t(detail::asF64(detail::asI64(a.v) | detail::asI64(b.v)));
}

inline int movemask(double4_t a)
{
   int result = 0;
   const i64x4_t u = detail::asI64(a.v);
   for (int i = 0; i < 4; ++i)
   {
      if (static_cast<uint64_t>(u[i]) & detail::MSB)
         result |= (1 << i);
   }
   return result;
}

inline double4_t blendv(double4_t a, double4_t b, double4_t mask)
{
   // Select b where mask MSB set (SSE blendv). Arithmetic >>63 splat of MSB.
   const i64x4_t mm = detail::asI64(mask.v) >> 63;
   const i64x4_t r  = (detail::asI64(a.v) & ~mm) | (detail::asI64(b.v) & mm);
   return double4_t(detail::asF64(r));
}

template< int mask >
inline double4_t blend(double4_t a, double4_t b)
{
   return make_double4((mask & (1 << 3)) ? b.v[3] : a.v[3],
                       (mask & (1 << 2)) ? b.v[2] : a.v[2],
                       (mask & (1 << 1)) ? b.v[1] : a.v[1],
                       (mask & (1 << 0)) ? b.v[0] : a.v[0]);
}

inline double4_t sqrt(double4_t a)
{
   // Per-lane builtin; GCC with -march=rv64gcv typically emits vfsqrt and
   // keeps the value in a vector register (no POD spill around the op).
   return double4_t(f64x4_t{
      __builtin_sqrt(a.v[0]),
      __builtin_sqrt(a.v[1]),
      __builtin_sqrt(a.v[2]),
      __builtin_sqrt(a.v[3])
   });
}

template< unsigned int numIter >
inline double4_t invSqrt(double4_t a)
{
   return make_double4(math::fastInvSqrt< numIter >(a.v[3]),
                       math::fastInvSqrt< numIter >(a.v[2]),
                       math::fastInvSqrt< numIter >(a.v[1]),
                       math::fastInvSqrt< numIter >(a.v[0]));
}

} // namespace rvv
} // namespace simd
} // namespace walberla

#endif // __riscv_vector || __riscv_v
