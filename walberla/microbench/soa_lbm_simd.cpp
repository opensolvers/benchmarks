//======================================================================================================================
//! \file soa_lbm_simd.cpp
//! \brief SoA D2Q9-ish stream-collide microbench exercising walberla::simd::double4_t (VL=4).
//!
//! Layout: contiguous double arrays per PDF component, stride along x.
//! Hot loop: load_aligned × Q, density/velocity moments, BGK-like collide (mul/add),
//! optional sqrt on |u|, store_aligned. Sized for multi-MB working set.
//======================================================================================================================

#include "simd/SIMD.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

using namespace walberla::simd;

namespace {

constexpr int Q   = 9;
constexpr int VL  = 4; // double4_t lane count
constexpr std::size_t ALIGN = 32;

// Lattice weights / velocities (D2Q9), broadcast as double4 splats in the kernel.
constexpr double W[Q] = {
   4.0 / 9.0,
   1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
   1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
constexpr int CX[Q] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
constexpr int CY[Q] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };

inline double* allocAligned(std::size_t nDoubles)
{
   void* p = nullptr;
   if (posix_memalign(&p, ALIGN, nDoubles * sizeof(double)) != 0 || !p)
      throw std::bad_alloc();
   return static_cast<double*>(p);
}

// Marked noinline so objdump can target the hot symbol cleanly.
__attribute__((noinline))
double collideSoA(double* __restrict__ f[Q], std::size_t nx, int iters, double omega)
{
   const double4_t vOmega  = make_double4(omega);
   const double4_t vOne    = make_double4(1.0);
   const double4_t vThree  = make_double4(3.0);
   const double4_t vNineHalf = make_double4(4.5); // 9/2
   const double4_t vMinus15 = make_double4(-1.5);

   double4_t vW[Q];
   double4_t vCx[Q];
   double4_t vCy[Q];
   for (int q = 0; q < Q; ++q) {
      vW[q]  = make_double4(W[q]);
      vCx[q] = make_double4(static_cast<double>(CX[q]));
      vCy[q] = make_double4(static_cast<double>(CY[q]));
   }

   // Sink: accumulate a few density samples so the compiler cannot DCE the loop.
   double4_t sink = make_zero();

   for (int it = 0; it < iters; ++it) {
      for (std::size_t i = 0; i + VL <= nx; i += VL) {
         double4_t fq[Q];
         for (int q = 0; q < Q; ++q)
            fq[q] = load_aligned(f[q] + i);

         // Density ρ = Σ f_q
         double4_t rho = fq[0];
         for (int q = 1; q < Q; ++q)
            rho = rho + fq[q];

         // Momentum / velocity
         double4_t jx = make_zero();
         double4_t jy = make_zero();
         for (int q = 1; q < Q; ++q) {
            jx = jx + fq[q] * vCx[q];
            jy = jy + fq[q] * vCy[q];
         }
         // invRho ≈ 1/ρ (stable for our init ρ≈1); use mul of reciprocal via / for honesty
         double4_t invRho = vOne / rho;
         double4_t ux = jx * invRho;
         double4_t uy = jy * invRho;

         double4_t u2 = ux * ux + uy * uy;
         // Touch sqrt so RVV vfsqrt shows up (speed magnitude–ish)
         double4_t uspeed = sqrt(u2 + make_double4(1e-30));
         sink = sink + uspeed * make_double4(1e-12);

         // BGK collide: f' = f - ω (f - f_eq)
         const double4_t oneMinusOmega = vOne - vOmega;
         for (int q = 0; q < Q; ++q) {
            const double4_t cu  = vThree * (vCx[q] * ux + vCy[q] * uy);
            const double4_t feq = rho * vW[q] * (vOne + cu + vNineHalf * (cu * cu) + vMinus15 * u2);
            const double4_t fout = oneMinusOmega * fq[q] + vOmega * feq;
            store_aligned(f[q] + i, fout);
         }
      }
   }

   // Horizontal reduction of sink → scalar checksum
   alignas(32) double tmp[4];
   store_aligned(tmp, sink);
   return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

void initSoA(double* f[Q], std::size_t nx)
{
   for (std::size_t i = 0; i < nx; ++i) {
      const double x = static_cast<double>(i) / static_cast<double>(nx);
      const double ux = 0.02 * std::sin(6.283185307179586 * x);
      const double uy = 0.01 * std::cos(6.283185307179586 * x);
      const double rho = 1.0;
      const double u2 = ux * ux + uy * uy;
      for (int q = 0; q < Q; ++q) {
         const double cu = 3.0 * (CX[q] * ux + CY[q] * uy);
         f[q][i] = rho * W[q] * (1.0 + cu + 4.5 * cu * cu - 1.5 * u2);
      }
   }
}

} // namespace

int main(int argc, char** argv)
{
   // Defaults: ~18.9 MiB PDF working set, enough iters for multi-second runs on Orange Pi.
   std::size_t nx    = 262144; // must be multiple of 4
   int         iters = 80;
   double      omega = 1.0;    // τ=1 BGK
   int         reps  = 1;

   for (int a = 1; a < argc; ++a) {
      const std::string s = argv[a];
      auto next = [&](double& d) {
         if (a + 1 < argc) d = std::atof(argv[++a]);
      };
      auto nexti = [&](int& d) {
         if (a + 1 < argc) d = std::atoi(argv[++a]);
      };
      auto nextz = [&](std::size_t& d) {
         if (a + 1 < argc) d = static_cast<std::size_t>(std::atoll(argv[++a]));
      };
      if (s == "--nx") nextz(nx);
      else if (s == "--iters") nexti(iters);
      else if (s == "--omega") next(omega);
      else if (s == "--reps") nexti(reps);
      else if (s == "--help") {
         std::printf("Usage: %s [--nx N] [--iters I] [--omega W] [--reps R]\n", argv[0]);
         return 0;
      }
   }

   if (nx < static_cast<std::size_t>(VL) || (nx % VL) != 0) {
      std::fprintf(stderr, "nx must be multiple of %d and >= %d\n", VL, VL);
      return 2;
   }

   std::printf("usedInstructionSet=%s\n", usedInstructionSet());
   std::printf("nx=%zu Q=%d VL=%d iters=%d omega=%g reps=%d\n", nx, Q, VL, iters, omega, reps);

   const std::size_t bytes = Q * nx * sizeof(double);
   std::printf("working_set_MiB=%.2f\n", bytes / (1024.0 * 1024.0));

   double* f[Q];
   for (int q = 0; q < Q; ++q)
      f[q] = allocAligned(nx);

   initSoA(f, nx);

   // Warmup (not timed)
   volatile double warm = collideSoA(f, nx, 1, omega);
   (void)warm;
   initSoA(f, nx); // reset after warmup

   double bestWall = 1e300;
   double lastChecksum = 0.0;
   for (int r = 0; r < reps; ++r) {
      initSoA(f, nx);
      const auto t0 = std::chrono::steady_clock::now();
      lastChecksum = collideSoA(f, nx, iters, omega);
      const auto t1 = std::chrono::steady_clock::now();
      const double wall = std::chrono::duration<double>(t1 - t0).count();
      if (wall < bestWall) bestWall = wall;
      std::printf("rep=%d WALL_s=%.6f checksum=%.17g\n", r, wall, lastChecksum);
   }

   const double cells = static_cast<double>(nx) * static_cast<double>(iters);
   const double mlups = (cells / bestWall) / 1.0e6;
   // Rough bytes touched: read Q + write Q per cell per iter
   const double gbs = (2.0 * static_cast<double>(bytes) * static_cast<double>(iters) / bestWall) / 1.0e9;

   std::printf("BEST_WALL_s=%.6f\n", bestWall);
   std::printf("MLUPS=%.3f\n", mlups);
   std::printf("approx_GB_s=%.3f\n", gbs);
   std::printf("checksum=%.17g\n", lastChecksum);

   for (int q = 0; q < Q; ++q)
      std::free(f[q]);
   return 0;
}
