/* Hand RVV LJ/cut — gather → SoA tiles, masked cutoff, vector LJ math.
 * Target: SpaceMiT X60, RVV 1.0, VLEN=256 (4×f64 / m1).
 * Accumulates force on i only (FULL / newton-off style).
 */
#include "lj_pair.h"

#include <riscv_vector.h>

enum { TILE = 64 };

static void lj_rvv_i(const double *xi, const double *x, const int *neighs, int jnum,
                     const LjParams *p, double *fi) {
  double dx[TILE], dy[TILE], dz[TILE];
  const double cutsq = p->cutsq;
  const double lj1 = p->lj1;
  const double lj2 = p->lj2;

  double fxt = 0.0, fyt = 0.0, fzt = 0.0;

  for (int base = 0; base < jnum; base += TILE) {
    const int n = (jnum - base < TILE) ? (jnum - base) : TILE;

    for (int t = 0; t < n; t++) {
      const int j = neighs[base + t] & 0x3FFFFFFF;
      const double *xj = x + 3 * j;
      dx[t] = xi[0] - xj[0];
      dy[t] = xi[1] - xj[1];
      dz[t] = xi[2] - xj[2];
    }

    size_t vl;
    for (size_t k = 0; k < (size_t)n; k += vl) {
      vl = __riscv_vsetvl_e64m1((size_t)n - k);

      vfloat64m1_t vdx = __riscv_vle64_v_f64m1(&dx[k], vl);
      vfloat64m1_t vdy = __riscv_vle64_v_f64m1(&dy[k], vl);
      vfloat64m1_t vdz = __riscv_vle64_v_f64m1(&dz[k], vl);

      vfloat64m1_t vrsq = __riscv_vfmul_vv_f64m1(vdx, vdx, vl);
      vrsq = __riscv_vfmacc_vv_f64m1(vrsq, vdy, vdy, vl);
      vrsq = __riscv_vfmacc_vv_f64m1(vrsq, vdz, vdz, vl);

      vbool64_t m_cut = __riscv_vmflt_vf_f64m1_b64(vrsq, cutsq, vl);
      vbool64_t m_pos = __riscv_vmfgt_vf_f64m1_b64(vrsq, 0.0, vl);
      vbool64_t m = __riscv_vmand_mm_b64(m_cut, m_pos, vl);

      /* Safe reciprocal: inactive lanes use rsq=1 → r2inv later zeroed. */
      vfloat64m1_t vone = __riscv_vfmv_v_f_f64m1(1.0, vl);
      vfloat64m1_t vrsq_safe = __riscv_vmerge_vvm_f64m1(vone, vrsq, m, vl);
      vfloat64m1_t vr2inv = __riscv_vfdiv_vv_f64m1(vone, vrsq_safe, vl);
      vfloat64m1_t vzero = __riscv_vfmv_v_f_f64m1(0.0, vl);
      vr2inv = __riscv_vmerge_vvm_f64m1(vzero, vr2inv, m, vl);

      vfloat64m1_t vr6inv = __riscv_vfmul_vv_f64m1(vr2inv, vr2inv, vl);
      vr6inv = __riscv_vfmul_vv_f64m1(vr6inv, vr2inv, vl);

      /* forcelj = r6inv * (lj1*r6inv - lj2) */
      vfloat64m1_t vt = __riscv_vfmul_vf_f64m1(vr6inv, lj1, vl);
      vt = __riscv_vfsub_vf_f64m1(vt, lj2, vl);
      vfloat64m1_t vforcelj = __riscv_vfmul_vv_f64m1(vr6inv, vt, vl);
      vfloat64m1_t vfpair = __riscv_vfmul_vv_f64m1(vforcelj, vr2inv, vl);

      vfloat64m1_t vfx = __riscv_vfmul_vv_f64m1(vdx, vfpair, vl);
      vfloat64m1_t vfy = __riscv_vfmul_vv_f64m1(vdy, vfpair, vl);
      vfloat64m1_t vfz = __riscv_vfmul_vv_f64m1(vdz, vfpair, vl);

      vfloat64m1_t z0 = __riscv_vfmv_v_f_f64m1(0.0, 1);
      fxt += __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(vfx, z0, vl));
      fyt += __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(vfy, z0, vl));
      fzt += __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m1_f64m1(vfz, z0, vl));
    }
  }

  fi[0] += fxt;
  fi[1] += fyt;
  fi[2] += fzt;
}

void lj_pair_rvv_csr(int nlocal, const double *x, const int *off, const int *neighs,
                     const LjParams *params, double *f) {
  for (int i = 0; i < nlocal; i++) {
    const int jnum = off[i + 1] - off[i];
    if (jnum <= 0)
      continue;
    lj_rvv_i(x + 3 * i, x, neighs + off[i], jnum, params, f + 3 * i);
  }
}

void lj_pair_rvv(int nlocal, const double *x, const int *ilist, const int *numneigh,
                 const int *const *firstneigh, const LjParams *params, double *f) {
  for (int ii = 0; ii < nlocal; ii++) {
    const int i = ilist ? ilist[ii] : ii;
    const int jnum = numneigh[i];
    if (jnum <= 0)
      continue;
    lj_rvv_i(x + 3 * i, x, firstneigh[i], jnum, params, f + 3 * i);
  }
}
