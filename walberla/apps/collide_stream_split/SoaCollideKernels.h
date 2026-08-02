//======================================================================================================================
//! \file SoaCollideKernels.h
//! \brief Contiguous SoA (fzyx) D2Q9 incompressible SRT collide — plain doubles for GCC auto-vec.
//!
//! Physics matches waLBerla `lbm::CellwiseSweep` D2Q9 SRT specialization (incompressible, no force):
//!   omega_trm = 1-ω;  omega_w0/w1/w2 = 3 * {4/9,1/9,1/36} * ω
//!   dens/vel as WALBERLA_LBM_CELLWISE_SWEEP_D2Q9_DENSITY_VELOCITY_INCOMP
//======================================================================================================================

#pragma once

#include <cstddef>
#include <cstdint>

namespace cssplit {

//! Hot collide over one x-row of length nx (contiguous fzyx PDF planes + flag bytes).
//! Writes post-collide PDFs in place. Skips cells where (flags[i] & fluidMask) == 0.
//! Marked noinline so -fopt-info / objdump can attribute vectorization cleanly.
__attribute__((noinline))
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
   double omega);

} // namespace cssplit
