// clang-format off
/* ----------------------------------------------------------------------
   Pair style eam/rvv — hand RVV EAM density + force (cubic spline tables).
   Fast path: single type, force-only, contiguous AoS x/f.
   Falls back to PairEAM::compute otherwise.
------------------------------------------------------------------------- */

#include "pair_eam_rvv.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <cstdint>
#include <riscv_vector.h>

using namespace LAMMPS_NS;

PairEAMRVV::PairEAMRVV(LAMMPS *lmp) : PairEAM(lmp) {}

enum { VMAX = 16 };

static inline int aos_contig(double **a, int nprobe)
{
  if (nprobe < 2) return 1;
  double *base = a[0];
  for (int i = 1; i < nprobe; i++)
    if (a[i] != base + 3 * i) return 0;
  return 1;
}

/* Density cubic value ((c3 p+c4)p+c5)p+c6 from rhor spline. */
static inline void spline_dens_tile(const double *spl, double rdr, int nr, double cutforcesq,
                                    double *val, const double *dx, const double *dy, const double *dz,
                                    int n)
{
  size_t vl;
  for (size_t k = 0; k < (size_t) n; k += vl) {
    vl = __riscv_vsetvl_e64m2((size_t) n - k);
    vfloat64m2_t vdx = __riscv_vle64_v_f64m2(&dx[k], vl);
    vfloat64m2_t vdy = __riscv_vle64_v_f64m2(&dy[k], vl);
    vfloat64m2_t vdz = __riscv_vle64_v_f64m2(&dz[k], vl);
    vfloat64m2_t vrsq = __riscv_vfmul_vv_f64m2(vdx, vdx, vl);
    vrsq = __riscv_vfmacc_vv_f64m2(vrsq, vdy, vdy, vl);
    vrsq = __riscv_vfmacc_vv_f64m2(vrsq, vdz, vdz, vl);

    vbool32_t m_cut = __riscv_vmflt_vf_f64m2_b32(vrsq, cutforcesq, vl);
    vbool32_t m_pos = __riscv_vmfgt_vf_f64m2_b32(vrsq, 0.0, vl);
    vbool32_t m = __riscv_vmand_mm_b32(m_cut, m_pos, vl);

    vfloat64m2_t vone = __riscv_vfmv_v_f_f64m2(1.0, vl);
    vfloat64m2_t vzero = __riscv_vfmv_v_f_f64m2(0.0, vl);
    vfloat64m2_t vrsq_safe = __riscv_vmerge_vvm_f64m2(vone, vrsq, m, vl);
    vfloat64m2_t vr = __riscv_vfsqrt_v_f64m2(vrsq_safe, vl);
    vfloat64m2_t vp = __riscv_vfmadd_vf_f64m2(vr, rdr, vone, vl);
    vint64m2_t vm_i = __riscv_vfcvt_rtz_x_f_v_i64m2(vp, vl);
    /* Match PairEAM: m = MIN((int)p, nr-1); p = MIN(p-m, 1) */
    vm_i = __riscv_vmin_vv_i64m2(vm_i, __riscv_vmv_v_x_i64m2((int64_t) (nr - 1), vl), vl);
    vfloat64m2_t vpfrac =
        __riscv_vfmin_vf_f64m2(__riscv_vfsub_vv_f64m2(vp, __riscv_vfcvt_f_x_v_f64m2(vm_i, vl), vl), 1.0, vl);

    vuint64m2_t voff = __riscv_vmul_vx_u64m2(__riscv_vreinterpret_v_i64m2_u64m2(vm_i), 56, vl);
    vfloat64m2_t c3 = __riscv_vloxei64_v_f64m2(spl, __riscv_vadd_vx_u64m2(voff, 24, vl), vl);
    vfloat64m2_t c4 = __riscv_vloxei64_v_f64m2(spl, __riscv_vadd_vx_u64m2(voff, 32, vl), vl);
    vfloat64m2_t c5 = __riscv_vloxei64_v_f64m2(spl, __riscv_vadd_vx_u64m2(voff, 40, vl), vl);
    vfloat64m2_t c6 = __riscv_vloxei64_v_f64m2(spl, __riscv_vadd_vx_u64m2(voff, 48, vl), vl);

    vfloat64m2_t vv = __riscv_vfmadd_vv_f64m2(c3, vpfrac, c4, vl);
    vv = __riscv_vfmadd_vv_f64m2(vv, vpfrac, c5, vl);
    vv = __riscv_vfmadd_vv_f64m2(vv, vpfrac, c6, vl);
    __riscv_vse64_v_f64m2(&val[k], __riscv_vmerge_vvm_f64m2(vzero, vv, m, vl), vl);
  }
}

/* Force geometry + rhor deriv + z2 val/deriv for a tile. */
static inline void force_spline_tile(const double *rhor_spl, const double *z2r_spl, double rdr, int nr,
                                     double cutforcesq, const double *dx, const double *dy,
                                     const double *dz, double *rhojp, double *z2, double *z2p,
                                     double *r_out, double *rsq_out, int n)
{
  size_t vl;
  for (size_t k = 0; k < (size_t) n; k += vl) {
    vl = __riscv_vsetvl_e64m2((size_t) n - k);
    vfloat64m2_t vdx = __riscv_vle64_v_f64m2(&dx[k], vl);
    vfloat64m2_t vdy = __riscv_vle64_v_f64m2(&dy[k], vl);
    vfloat64m2_t vdz = __riscv_vle64_v_f64m2(&dz[k], vl);
    vfloat64m2_t vrsq = __riscv_vfmul_vv_f64m2(vdx, vdx, vl);
    vrsq = __riscv_vfmacc_vv_f64m2(vrsq, vdy, vdy, vl);
    vrsq = __riscv_vfmacc_vv_f64m2(vrsq, vdz, vdz, vl);
    __riscv_vse64_v_f64m2(&rsq_out[k], vrsq, vl);

    vbool32_t m_cut = __riscv_vmflt_vf_f64m2_b32(vrsq, cutforcesq, vl);
    vbool32_t m_pos = __riscv_vmfgt_vf_f64m2_b32(vrsq, 0.0, vl);
    vbool32_t m = __riscv_vmand_mm_b32(m_cut, m_pos, vl);

    vfloat64m2_t vone = __riscv_vfmv_v_f_f64m2(1.0, vl);
    vfloat64m2_t vzero = __riscv_vfmv_v_f_f64m2(0.0, vl);
    vfloat64m2_t vrsq_safe = __riscv_vmerge_vvm_f64m2(vone, vrsq, m, vl);
    vfloat64m2_t vr = __riscv_vfsqrt_v_f64m2(vrsq_safe, vl);
    __riscv_vse64_v_f64m2(&r_out[k], vr, vl);

    vfloat64m2_t vp = __riscv_vfmadd_vf_f64m2(vr, rdr, vone, vl);
    vint64m2_t vm_i = __riscv_vfcvt_rtz_x_f_v_i64m2(vp, vl);
    vm_i = __riscv_vmin_vv_i64m2(vm_i, __riscv_vmv_v_x_i64m2((int64_t) (nr - 1), vl), vl);
    vfloat64m2_t vpfrac =
        __riscv_vfmin_vf_f64m2(__riscv_vfsub_vv_f64m2(vp, __riscv_vfcvt_f_x_v_f64m2(vm_i, vl), vl), 1.0, vl);
    vuint64m2_t voff = __riscv_vmul_vx_u64m2(__riscv_vreinterpret_v_i64m2_u64m2(vm_i), 56, vl);

    /* rhor deriv (c0,c1,c2) */
    vfloat64m2_t c0 = __riscv_vloxei64_v_f64m2(rhor_spl, voff, vl);
    vfloat64m2_t c1 = __riscv_vloxei64_v_f64m2(rhor_spl, __riscv_vadd_vx_u64m2(voff, 8, vl), vl);
    vfloat64m2_t c2 = __riscv_vloxei64_v_f64m2(rhor_spl, __riscv_vadd_vx_u64m2(voff, 16, vl), vl);
    vfloat64m2_t vd = __riscv_vfmadd_vv_f64m2(c0, vpfrac, c1, vl);
    vd = __riscv_vfmadd_vv_f64m2(vd, vpfrac, c2, vl);
    __riscv_vse64_v_f64m2(&rhojp[k], __riscv_vmerge_vvm_f64m2(vzero, vd, m, vl), vl);

    /* z2 deriv + value */
    c0 = __riscv_vloxei64_v_f64m2(z2r_spl, voff, vl);
    c1 = __riscv_vloxei64_v_f64m2(z2r_spl, __riscv_vadd_vx_u64m2(voff, 8, vl), vl);
    c2 = __riscv_vloxei64_v_f64m2(z2r_spl, __riscv_vadd_vx_u64m2(voff, 16, vl), vl);
    vfloat64m2_t c3 = __riscv_vloxei64_v_f64m2(z2r_spl, __riscv_vadd_vx_u64m2(voff, 24, vl), vl);
    vfloat64m2_t c4 = __riscv_vloxei64_v_f64m2(z2r_spl, __riscv_vadd_vx_u64m2(voff, 32, vl), vl);
    vfloat64m2_t c5 = __riscv_vloxei64_v_f64m2(z2r_spl, __riscv_vadd_vx_u64m2(voff, 40, vl), vl);
    vfloat64m2_t c6 = __riscv_vloxei64_v_f64m2(z2r_spl, __riscv_vadd_vx_u64m2(voff, 48, vl), vl);
    vd = __riscv_vfmadd_vv_f64m2(c0, vpfrac, c1, vl);
    vd = __riscv_vfmadd_vv_f64m2(vd, vpfrac, c2, vl);
    vfloat64m2_t vv = __riscv_vfmadd_vv_f64m2(c3, vpfrac, c4, vl);
    vv = __riscv_vfmadd_vv_f64m2(vv, vpfrac, c5, vl);
    vv = __riscv_vfmadd_vv_f64m2(vv, vpfrac, c6, vl);
    __riscv_vse64_v_f64m2(&z2p[k], __riscv_vmerge_vvm_f64m2(vzero, vd, m, vl), vl);
    __riscv_vse64_v_f64m2(&z2[k], __riscv_vmerge_vvm_f64m2(vzero, vv, m, vl), vl);
  }
}

void PairEAMRVV::compute(int eflag, int vflag)
{
  if (eflag || vflag || atom->ntypes != 1) {
    PairEAM::compute(eflag, vflag);
    return;
  }

  ev_init(eflag, vflag);

  if (atom->nmax > nmax) {
    memory->destroy(rho);
    memory->destroy(fp);
    memory->destroy(numforce);
    nmax = atom->nmax;
    memory->create(rho, nmax, "pair:rho");
    memory->create(fp, nmax, "pair:fp");
    memory->create(numforce, nmax, "pair:numforce");
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int newton_pair = force->newton_pair;

  if (!aos_contig(x, nall > 2 ? 3 : nall) || !aos_contig(f, nlocal > 2 ? 3 : nlocal)) {
    PairEAM::compute(eflag, vflag);
    return;
  }

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  double *xbase = x[0];
  double *fbase = f[0];

  const double *rhor_spl = &rhor_spline[type2rhor[1][1]][0][0];
  const double *z2r_spl = &z2r_spline[type2z2r[1][1]][0][0];
  const double sc = scale[1][1];

  if (newton_pair) {
    for (int i = 0; i < nall; i++) rho[i] = 0.0;
  } else {
    for (int i = 0; i < nlocal; i++) rho[i] = 0.0;
  }

  uint64_t jidx[VMAX];
  double dx[VMAX], dy[VMAX], dz[VMAX], dens[VMAX];

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const double xi = xbase[3 * i];
    const double yi = xbase[3 * i + 1];
    const double zi = xbase[3 * i + 2];
    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];

    for (int base = 0; base < jnum;) {
      size_t vl0 = __riscv_vsetvl_e64m2((size_t) (jnum - base));
      if (vl0 > (size_t) VMAX) vl0 = VMAX;
      const int n = (int) vl0;
      for (int t = 0; t < n; t++) {
        const int j = jlist[base + t] & NEIGHMASK;
        jidx[t] = (uint64_t) j;
        dx[t] = xi - xbase[3 * j];
        dy[t] = yi - xbase[3 * j + 1];
        dz[t] = zi - xbase[3 * j + 2];
      }
      spline_dens_tile(rhor_spl, rdr, nr, cutforcesq, dens, dx, dy, dz, n);

      double rho_i_add = 0.0;
      for (int t = 0; t < n; t++) {
        const double d = dens[t];
        if (d == 0.0) continue;
        rho_i_add += d;
        const int j = (int) jidx[t];
        if (newton_pair || j < nlocal) rho[j] += d;
      }
      rho[i] += rho_i_add;
      base += n;
    }
  }

  if (newton_pair) comm->reverse_comm(this);

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    double p = rho[i] * rdrho + 1.0;
    int m = static_cast<int>(p);
    m = m < 1 ? 1 : (m > nrho - 1 ? nrho - 1 : m);
    p -= m;
    if (p > 1.0) p = 1.0;
    double *coeff = frho_spline[type2frho[type[i]]][m];
    fp[i] = (coeff[0] * p + coeff[1]) * p + coeff[2];
  }

  comm->forward_comm(this);
  embedstep = update->ntimestep;

  double rhojp_a[VMAX], z2_a[VMAX], z2p_a[VMAX], r_a[VMAX], rsq_a[VMAX];

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const double xi = xbase[3 * i];
    const double yi = xbase[3 * i + 1];
    const double zi = xbase[3 * i + 2];
    const double fpi = fp[i];
    double *fi = &fbase[3 * i];
    int *jlist = firstneigh[i];
    const int jnum = numneigh[i];
    numforce[i] = 0;

    for (int base = 0; base < jnum;) {
      size_t vl0 = __riscv_vsetvl_e64m2((size_t) (jnum - base));
      if (vl0 > (size_t) VMAX) vl0 = VMAX;
      const int n = (int) vl0;
      for (int t = 0; t < n; t++) {
        const int j = jlist[base + t] & NEIGHMASK;
        jidx[t] = (uint64_t) j;
        dx[t] = xi - xbase[3 * j];
        dy[t] = yi - xbase[3 * j + 1];
        dz[t] = zi - xbase[3 * j + 2];
      }

      force_spline_tile(rhor_spl, z2r_spl, rdr, nr, cutforcesq, dx, dy, dz, rhojp_a, z2_a, z2p_a, r_a,
                        rsq_a, n);

      for (int t = 0; t < n; t++) {
        if (rsq_a[t] >= cutforcesq || rsq_a[t] <= 0.0) continue;
        ++numforce[i];
        const int j = (int) jidx[t];
        const double r = r_a[t];
        const double recip = 1.0 / r;
        const double phi = z2_a[t] * recip;
        const double phip = z2p_a[t] * recip - phi * recip;
        /* unary EAM: rhoip == rhojp */
        const double psip = fpi * rhojp_a[t] + fp[j] * rhojp_a[t] + phip;
        const double fpair = -sc * psip * recip;
        const double delx = dx[t], dely = dy[t], delz = dz[t];
        fi[0] += delx * fpair;
        fi[1] += dely * fpair;
        fi[2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          fbase[3 * j + 0] -= delx * fpair;
          fbase[3 * j + 1] -= dely * fpair;
          fbase[3 * j + 2] -= delz * fpair;
        }
      }
      base += n;
    }
  }
}
