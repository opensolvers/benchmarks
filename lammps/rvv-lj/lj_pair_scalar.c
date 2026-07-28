#include "lj_pair.h"

#include <math.h>

static inline void accum_i(const double *xi, const double *xj, const LjParams *p, double *fi) {
  const double delx = xi[0] - xj[0];
  const double dely = xi[1] - xj[1];
  const double delz = xi[2] - xj[2];
  const double rsq = delx * delx + dely * dely + delz * delz;
  if (!(rsq < p->cutsq) || rsq == 0.0)
    return;
  const double r2inv = 1.0 / rsq;
  const double r6inv = r2inv * r2inv * r2inv;
  const double forcelj = r6inv * (p->lj1 * r6inv - p->lj2);
  const double fpair = forcelj * r2inv;
  fi[0] += delx * fpair;
  fi[1] += dely * fpair;
  fi[2] += delz * fpair;
}

void lj_pair_scalar_csr(int nlocal, const double *x, const int *off, const int *neighs,
                        const LjParams *params, double *f) {
  for (int i = 0; i < nlocal; i++) {
    const double *xi = x + 3 * i;
    double *fi = f + 3 * i;
    for (int jj = off[i]; jj < off[i + 1]; jj++) {
      const int j = neighs[jj];
      accum_i(xi, x + 3 * j, params, fi);
    }
  }
}

void lj_pair_scalar(int nlocal, const double *x, const int *ilist, const int *numneigh,
                    const int *const *firstneigh, const LjParams *params, double *f) {
  for (int ii = 0; ii < nlocal; ii++) {
    const int i = ilist ? ilist[ii] : ii;
    const double *xi = x + 3 * i;
    double *fi = f + 3 * i;
    const int *nlist = firstneigh[i];
    const int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = nlist[jj];
      j &= 0x3FFFFFFF; /* NEIGHMASK-ish; microbench usually plain indices */
      accum_i(xi, x + 3 * j, params, fi);
    }
  }
}
