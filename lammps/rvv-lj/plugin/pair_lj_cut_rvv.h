/* -*- c++ -*- ----------------------------------------------------------
   Pair style lj/cut/rvv — RVV SoA-tiled LJ/cut (SpaceMiT X60 / RVV 1.0).
   Inherits PairLJCut; overrides compute() only.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut/rvv,PairLJCutRVV);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_RVV_H
#define LMP_PAIR_LJ_CUT_RVV_H

#include "pair_lj_cut.h"

namespace LAMMPS_NS {

class PairLJCutRVV : public PairLJCut {
 public:
  PairLJCutRVV(class LAMMPS *);
  void compute(int, int) override;
};

}    // namespace LAMMPS_NS

#endif
#endif
