/* -*- c++ -*- ----------------------------------------------------------
   Pair style eam/rvv — RVV cubic-spline EAM (SpaceMiT X60).
   Inherits PairEAM; overrides compute() for single-type force-only fast path.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(eam/rvv,PairEAMRVV);
// clang-format on
#else

#ifndef LMP_PAIR_EAM_RVV_H
#define LMP_PAIR_EAM_RVV_H

#include "pair_eam.h"

namespace LAMMPS_NS {

class PairEAMRVV : public PairEAM {
 public:
  PairEAMRVV(class LAMMPS *);
  void compute(int, int) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
