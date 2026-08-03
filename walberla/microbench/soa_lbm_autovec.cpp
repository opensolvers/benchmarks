//======================================================================================================================
//! \file soa_lbm_autovec.cpp
//! \brief SoA D2Q9-ish BGK collide microbench written for GCC auto-vectorization (plain double, no walberla::simd).
//!
//! Layout: contiguous double arrays per PDF component, stride along x.
//! Hot loop: load Q PDFs, density/velocity moments, BGK collide, optional sqrt sink, store.
//! Same problem size as soa_lbm_simd.cpp (nx=262144, 80 iters) for fair A/B.
//======================================================================================================================

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

namespace {

constexpr int Q = 9;
constexpr std::size_t ALIGN = 64; // cache-line / RVV-friendly

constexpr double W[Q] = {
   4.0 / 9.0,
   1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
   1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
constexpr double CX[Q] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
constexpr double CY[Q] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };

inline double* allocAligned(std::size_t nDoubles)
{
   void* p = nullptr;
   if (posix_memalign(&p, ALIGN, nDoubles * sizeof(double)) != 0 || !p)
      throw std::bad_alloc();
   return static_cast<double*>(p);
}

// Marked noinline so objdump / -fopt-info can target the hot symbol cleanly.
__attribute__((noinline))
double collideSoA(
   double* __restrict__ f0, double* __restrict__ f1, double* __restrict__ f2,
   double* __restrict__ f3, double* __restrict__ f4, double* __restrict__ f5,
   double* __restrict__ f6, double* __restrict__ f7, double* __restrict__ f8,
   std::size_t nx, int iters, double omega)
{
   const double oneMinusOmega = 1.0 - omega;
   const double w0 = W[0], w1 = W[1], w2 = W[2], w3 = W[3], w4 = W[4];
   const double w5 = W[5], w6 = W[6], w7 = W[7], w8 = W[8];
   const double cx1 = CX[1], cx2 = CX[2], cx3 = CX[3], cx4 = CX[4];
   const double cx5 = CX[5], cx6 = CX[6], cx7 = CX[7], cx8 = CX[8];
   const double cy1 = CY[1], cy2 = CY[2], cy3 = CY[3], cy4 = CY[4];
   const double cy5 = CY[5], cy6 = CY[6], cy7 = CY[7], cy8 = CY[8];

   // Sink: accumulate speed samples so the compiler cannot DCE the loop.
   double sink = 0.0;

   for (int it = 0; it < iters; ++it) {
      // Contiguous x-stride; no loop-carried deps across cells (collide-only).
#pragma GCC ivdep
      for (std::size_t i = 0; i < nx; ++i) {
         const double fq0 = f0[i];
         const double fq1 = f1[i];
         const double fq2 = f2[i];
         const double fq3 = f3[i];
         const double fq4 = f4[i];
         const double fq5 = f5[i];
         const double fq6 = f6[i];
         const double fq7 = f7[i];
         const double fq8 = f8[i];

         const double rho =
            fq0 + fq1 + fq2 + fq3 + fq4 + fq5 + fq6 + fq7 + fq8;

         const double jx =
            fq1 * cx1 + fq2 * cx2 + fq3 * cx3 + fq4 * cx4 +
            fq5 * cx5 + fq6 * cx6 + fq7 * cx7 + fq8 * cx8;
         const double jy =
            fq1 * cy1 + fq2 * cy2 + fq3 * cy3 + fq4 * cy4 +
            fq5 * cy5 + fq6 * cy6 + fq7 * cy7 + fq8 * cy8;

         const double invRho = 1.0 / rho;
         const double ux = jx * invRho;
         const double uy = jy * invRho;
         const double u2 = ux * ux + uy * uy;

         // Touch sqrt so RVV vfsqrt can appear under auto-vec (matches simd microbench).
         const double uspeed = std::sqrt(u2 + 1e-30);
         sink += uspeed * 1e-12;

         // BGK: f' = (1-ω) f + ω f_eq
         const double cu0 = 0.0;
         const double cu1 = 3.0 * (cx1 * ux + cy1 * uy);
         const double cu2 = 3.0 * (cx2 * ux + cy2 * uy);
         const double cu3 = 3.0 * (cx3 * ux + cy3 * uy);
         const double cu4 = 3.0 * (cx4 * ux + cy4 * uy);
         const double cu5 = 3.0 * (cx5 * ux + cy5 * uy);
         const double cu6 = 3.0 * (cx6 * ux + cy6 * uy);
         const double cu7 = 3.0 * (cx7 * ux + cy7 * uy);
         const double cu8 = 3.0 * (cx8 * ux + cy8 * uy);

         f0[i] = oneMinusOmega * fq0 + omega * (rho * w0 * (1.0 + cu0 + 4.5 * cu0 * cu0 - 1.5 * u2));
         f1[i] = oneMinusOmega * fq1 + omega * (rho * w1 * (1.0 + cu1 + 4.5 * cu1 * cu1 - 1.5 * u2));
         f2[i] = oneMinusOmega * fq2 + omega * (rho * w2 * (1.0 + cu2 + 4.5 * cu2 * cu2 - 1.5 * u2));
         f3[i] = oneMinusOmega * fq3 + omega * (rho * w3 * (1.0 + cu3 + 4.5 * cu3 * cu3 - 1.5 * u2));
         f4[i] = oneMinusOmega * fq4 + omega * (rho * w4 * (1.0 + cu4 + 4.5 * cu4 * cu4 - 1.5 * u2));
         f5[i] = oneMinusOmega * fq5 + omega * (rho * w5 * (1.0 + cu5 + 4.5 * cu5 * cu5 - 1.5 * u2));
         f6[i] = oneMinusOmega * fq6 + omega * (rho * w6 * (1.0 + cu6 + 4.5 * cu6 * cu6 - 1.5 * u2));
         f7[i] = oneMinusOmega * fq7 + omega * (rho * w7 * (1.0 + cu7 + 4.5 * cu7 * cu7 - 1.5 * u2));
         f8[i] = oneMinusOmega * fq8 + omega * (rho * w8 * (1.0 + cu8 + 4.5 * cu8 * cu8 - 1.5 * u2));
      }
   }

   return sink;
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
   std::size_t nx    = 262144;
   int         iters = 80;
   double      omega = 1.0;
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

   if (nx < 1) {
      std::fprintf(stderr, "nx must be >= 1\n");
      return 2;
   }

#if defined(__riscv_v)
   std::printf("build_has_riscv_v=1\n");
#else
   std::printf("build_has_riscv_v=0\n");
#endif
#ifdef __OPTIMIZE__
   std::printf("optimize=1\n");
#endif
   std::printf("nx=%zu Q=%d iters=%d omega=%g reps=%d\n", nx, Q, iters, omega, reps);
   std::printf("kernel=plain_double_SoA_autovec (no walberla::simd)\n");

   const std::size_t bytes = Q * nx * sizeof(double);
   std::printf("working_set_MiB=%.2f\n", bytes / (1024.0 * 1024.0));

   double* f[Q];
   for (int q = 0; q < Q; ++q)
      f[q] = allocAligned(nx);

   initSoA(f, nx);

   volatile double warm = collideSoA(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8],
                                     nx, 1, omega);
   (void)warm;
   initSoA(f, nx);

   double bestWall = 1e300;
   double lastChecksum = 0.0;
   for (int r = 0; r < reps; ++r) {
      initSoA(f, nx);
      const auto t0 = std::chrono::steady_clock::now();
      lastChecksum = collideSoA(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8],
                                nx, iters, omega);
      const auto t1 = std::chrono::steady_clock::now();
      const double wall = std::chrono::duration<double>(t1 - t0).count();
      if (wall < bestWall) bestWall = wall;
      std::printf("rep=%d WALL_s=%.6f checksum=%.17g\n", r, wall, lastChecksum);
   }

   const double cells = static_cast<double>(nx) * static_cast<double>(iters);
   const double mlups = (cells / bestWall) / 1.0e6;
   const double gbs = (2.0 * static_cast<double>(bytes) * static_cast<double>(iters) / bestWall) / 1.0e9;

   std::printf("BEST_WALL_s=%.6f\n", bestWall);
   std::printf("MLUPS=%.3f\n", mlups);
   std::printf("approx_GB_s=%.3f\n", gbs);
   std::printf("checksum=%.17g\n", lastChecksum);

   for (int q = 0; q < Q; ++q)
      std::free(f[q]);
   return 0;
}
