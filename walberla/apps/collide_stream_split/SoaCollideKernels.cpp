//======================================================================================================================
//! \file SoaCollideKernels.cpp
//! \brief Contiguous SoA D2Q9 incompressible SRT collide row kernel (GCC auto-vec target).
//!
//! Flag gating uses arithmetic blend (keep/drop) instead of `continue`, so the x-loop stays
//! auto-vectorizable under `-O2 -march=rv64gcv -ftree-vectorize`.
//======================================================================================================================

#include "SoaCollideKernels.h"

namespace cssplit {

void collideSoARowD2Q9IncompSRT(
   double* __restrict__ fC,
   double* __restrict__ fN,
   double* __restrict__ fS,
   double* __restrict__ fW,
   double* __restrict__ fE,
   double* __restrict__ fNW,
   double* __restrict__ fNE,
   double* __restrict__ fSW,
   double* __restrict__ fSE,
   const std::uint8_t* __restrict__ flags,
   std::uint8_t fluidMask,
   std::size_t nx,
   double omega)
{
   const double omega_trm = 1.0 - omega;
   const double omega_w0  = 3.0 * (4.0 / 9.0)  * omega;
   const double omega_w1  = 3.0 * (1.0 / 9.0)  * omega;
   const double omega_w2  = 3.0 * (1.0 / 36.0) * omega;
   const double one_third = 1.0 / 3.0;

#pragma GCC ivdep
   for (std::size_t i = 0; i < nx; ++i) {
      const double keep = ((flags[i] & fluidMask) != 0) ? 1.0 : 0.0;
      const double drop = 1.0 - keep;

      const double vC  = fC[i];
      const double vN  = fN[i];
      const double vS  = fS[i];
      const double vW  = fW[i];
      const double vE  = fE[i];
      const double vNW = fNW[i];
      const double vNE = fNE[i];
      const double vSW = fSW[i];
      const double vSE = fSE[i];

      // Incompressible dens/vel (CellwiseSweep D2Q9)
      const double velXTerm = vE + vNE + vSE;
      const double velYTerm = vN + vNW;
      const double rho      = vC + vS + vW + vSW + velXTerm + velYTerm;
      const double velX     = velXTerm - vW - vNW - vSW;
      const double velY     = velYTerm + vNE - vS - vSW - vSE;

      const double velXX = velX * velX;
      const double velYY = velY * velY;
      const double dir_indep_trm = one_third * rho - 0.5 * (velXX + velYY);

      const double nC = omega_trm * vC + omega_w0 * dir_indep_trm;

      const double vel_trm_E_W = dir_indep_trm + 1.5 * velXX;
      const double vel_trm_N_S = dir_indep_trm + 1.5 * velYY;

      const double nE = omega_trm * vE + omega_w1 * (vel_trm_E_W + velX);
      const double nW = omega_trm * vW + omega_w1 * (vel_trm_E_W - velX);
      const double nN = omega_trm * vN + omega_w1 * (vel_trm_N_S + velY);
      const double nS = omega_trm * vS + omega_w1 * (vel_trm_N_S - velY);

      const double velXmY        = velX - velY;
      const double vel_trm_NW_SE = dir_indep_trm + 1.5 * velXmY * velXmY;

      const double nNW = omega_trm * vNW + omega_w2 * (vel_trm_NW_SE - velXmY);
      const double nSE = omega_trm * vSE + omega_w2 * (vel_trm_NW_SE + velXmY);

      const double velXpY        = velX + velY;
      const double vel_trm_NE_SW = dir_indep_trm + 1.5 * velXpY * velXpY;

      const double nNE = omega_trm * vNE + omega_w2 * (vel_trm_NE_SW + velXpY);
      const double nSW = omega_trm * vSW + omega_w2 * (vel_trm_NE_SW - velXpY);

      fC[i]  = keep * nC  + drop * vC;
      fN[i]  = keep * nN  + drop * vN;
      fS[i]  = keep * nS  + drop * vS;
      fW[i]  = keep * nW  + drop * vW;
      fE[i]  = keep * nE  + drop * vE;
      fNW[i] = keep * nNW + drop * vNW;
      fNE[i] = keep * nNE + drop * vNE;
      fSW[i] = keep * nSW + drop * vSW;
      fSE[i] = keep * nSE + drop * vSE;
   }
}

} // namespace cssplit
