// clang-format off
/* ----------------------------------------------------------------------
   Pair style lj/cut/rvv — hand RVV LJ/cut compute (force + energy/virial).

   Hot path (1 type, force-only): RVV indexed gather from contiguous atom->x,
   vector LJ → fpair/Δ, scalar apply to f[i] and newton f[j].
------------------------------------------------------------------------- */

#include "pair_lj_cut_rvv.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cstdint>
#include <riscv_vector.h>

using namespace LAMMPS_NS;

PairLJCutRVV::PairLJCutRVV(LAMMPS *lmp) : PairLJCut(lmp) {}

enum { VMAX = 16 };

static inline int aos_contig(double **a, int nprobe)
{
  if (nprobe < 2) return 1;
  double *base = a[0];
  for (int i = 1; i < nprobe; i++)
    if (a[i] != base + 3 * i) return 0;
  return 1;
}

void PairLJCutRVV::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  const int fast = (atom->ntypes == 1) && (special_lj[1] == 1.0) && (special_lj[2] == 1.0) &&
      (special_lj[3] == 1.0);
  const int contig =
      fast && aos_contig(x, nall > 2 ? 3 : nall) && aos_contig(f, nlocal > 2 ? 3 : nlocal);

  if (fast && !eflag && !vflag && contig) {
    const double lj1v = lj1[1][1];
    const double lj2v = lj2[1][1];
    const double cutsqv = cutsq[1][1];
    double *xbase = x[0];
    double *fbase = f[0];
    uint64_t jidx[VMAX];
    double fpair_s[VMAX], dx_s[VMAX], dy_s[VMAX], dz_s[VMAX];

    for (int ii = 0; ii < inum; ii++) {
      const int i = ilist[ii];
      const double xi = xbase[3 * i];
      const double yi = xbase[3 * i + 1];
      const double zi = xbase[3 * i + 2];
      double *fi = &fbase[3 * i];
      int *jlist = firstneigh[i];
      const int jnum = numneigh[i];

      for (int base = 0; base < jnum;) {
        size_t vl0 = __riscv_vsetvl_e64m2((size_t) (jnum - base));
        if (vl0 > (size_t) VMAX) vl0 = VMAX;
        const int n = (int) vl0;
        for (int t = 0; t < n; t++) jidx[t] = (uint64_t) (jlist[base + t] & NEIGHMASK);

        size_t vl;
        for (size_t k = 0; k < (size_t) n; k += vl) {
          vl = __riscv_vsetvl_e64m2((size_t) n - k);

          vuint64m2_t vj = __riscv_vle64_v_u64m2(&jidx[k], vl);
          vuint64m2_t vox = __riscv_vmul_vx_u64m2(vj, 24, vl);
          vuint64m2_t voy = __riscv_vadd_vx_u64m2(vox, 8, vl);
          vuint64m2_t voz = __riscv_vadd_vx_u64m2(vox, 16, vl);

          vfloat64m2_t vxj = __riscv_vloxei64_v_f64m2(xbase, vox, vl);
          vfloat64m2_t vyj = __riscv_vloxei64_v_f64m2(xbase, voy, vl);
          vfloat64m2_t vzj = __riscv_vloxei64_v_f64m2(xbase, voz, vl);

          vfloat64m2_t vdx = __riscv_vfrsub_vf_f64m2(vxj, xi, vl);
          vfloat64m2_t vdy = __riscv_vfrsub_vf_f64m2(vyj, yi, vl);
          vfloat64m2_t vdz = __riscv_vfrsub_vf_f64m2(vzj, zi, vl);

          vfloat64m2_t vrsq = __riscv_vfmul_vv_f64m2(vdx, vdx, vl);
          vrsq = __riscv_vfmacc_vv_f64m2(vrsq, vdy, vdy, vl);
          vrsq = __riscv_vfmacc_vv_f64m2(vrsq, vdz, vdz, vl);

          vbool32_t m_cut = __riscv_vmflt_vf_f64m2_b32(vrsq, cutsqv, vl);
          vbool32_t m_pos = __riscv_vmfgt_vf_f64m2_b32(vrsq, 0.0, vl);
          vbool32_t m = __riscv_vmand_mm_b32(m_cut, m_pos, vl);

          vfloat64m2_t vone = __riscv_vfmv_v_f_f64m2(1.0, vl);
          vfloat64m2_t vzero = __riscv_vfmv_v_f_f64m2(0.0, vl);
          vfloat64m2_t vrsq_safe = __riscv_vmerge_vvm_f64m2(vone, vrsq, m, vl);
          vfloat64m2_t vr2inv = __riscv_vfdiv_vv_f64m2(vone, vrsq_safe, vl);
          vr2inv = __riscv_vmerge_vvm_f64m2(vzero, vr2inv, m, vl);

          vfloat64m2_t vr6inv = __riscv_vfmul_vv_f64m2(vr2inv, vr2inv, vl);
          vr6inv = __riscv_vfmul_vv_f64m2(vr6inv, vr2inv, vl);

          vfloat64m2_t vt = __riscv_vfmul_vf_f64m2(vr6inv, lj1v, vl);
          vt = __riscv_vfsub_vf_f64m2(vt, lj2v, vl);
          vfloat64m2_t vforcelj = __riscv_vfmul_vv_f64m2(vr6inv, vt, vl);
          vfloat64m2_t vfpair = __riscv_vfmul_vv_f64m2(vforcelj, vr2inv, vl);
          vfpair = __riscv_vmerge_vvm_f64m2(vzero, vfpair, m, vl);

          __riscv_vse64_v_f64m2(&fpair_s[k], vfpair, vl);
          __riscv_vse64_v_f64m2(&dx_s[k], vdx, vl);
          __riscv_vse64_v_f64m2(&dy_s[k], vdy, vl);
          __riscv_vse64_v_f64m2(&dz_s[k], vdz, vl);
        }

        for (int t = 0; t < n; t++) {
          const double fp = fpair_s[t];
          if (fp == 0.0) continue;
          const double delx = dx_s[t], dely = dy_s[t], delz = dz_s[t];
          fi[0] += delx * fp;
          fi[1] += dely * fp;
          fi[2] += delz * fp;
          const int j = (int) jidx[t];
          if (newton_pair || j < nlocal) {
            fbase[3 * j + 0] -= delx * fp;
            fbase[3 * j + 1] -= dely * fp;
            fbase[3 * j + 2] -= delz * fp;
          }
        }
        base += n;
      }
    }
    return;
  }

  /* Fallback: scalar PairLJCut-equivalent (energy/virial/multi-type). */
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];
    const int itype = type[i];
    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      const double factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      const double delx = xtmp - x[j][0];
      const double dely = ytmp - x[j][1];
      const double delz = ztmp - x[j][2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      const int jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        const double r2inv = 1.0 / rsq;
        const double r6inv = r2inv * r2inv * r2inv;
        const double forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        const double fpair = factor_lj * forcelj * r2inv;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        double evdwl = 0.0;
        if (eflag) {
          evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
          evdwl *= factor_lj;
        }
        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}
