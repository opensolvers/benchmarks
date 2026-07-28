/* RVV LJ/cut pair microkernel — SoA neighbor tiles (X60 / VLEN=256).
 *
 * Scope (step 1): accumulate force on atom i only (FULL neigh or newton off).
 * Neighbor list is built outside; this only evaluates the Pair hot loop.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double lj1, lj2; /* LAMMPS-style: forcelj = r6inv*(lj1*r6inv - lj2) */
  double cutsq;
} LjParams;

/* Scalar reference: for each i, loop neighbors, accumulate f[i]. */
void lj_pair_scalar(int nlocal, const double *x, /* nall x 3, xyzxyz... or SoA? use AoS xyz */
                    const int *ilist, const int *numneigh, const int *const *firstneigh,
                    const LjParams *params, double *f /* nlocal x 3, zeroed by caller */);

/* Same contract; RVV SoA-tiled inner loop. */
void lj_pair_rvv(int nlocal, const double *x, const int *ilist, const int *numneigh,
                 const int *const *firstneigh, const LjParams *params, double *f);

/* Flat neighbor CSR variant (easier for benches): neighs[off[i] .. off[i+1]). */
void lj_pair_scalar_csr(int nlocal, const double *x, const int *off, const int *neighs,
                        const LjParams *params, double *f);
void lj_pair_rvv_csr(int nlocal, const double *x, const int *off, const int *neighs,
                     const LjParams *params, double *f);

#ifdef __cplusplus
}
#endif
