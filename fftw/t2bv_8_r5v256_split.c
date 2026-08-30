/* Hand-specialized t2bv_8 for r5v256 double: split re/im via vlseg2/vsseg2.
 * Avoids FLIP_RI vrgather in BYTW2 / VFMAI (K1 gather is expensive).
 * Algorithm matches genfft FMA t2bv_8 (sign -1, VTW2 twiddles).
 *
 * Drop in as dft/simd/r5v256/t2bv_8.c after:
 *   #define SIMD_HEADER "simd-support/simd-r5v256.h"
 */
#include "dft/codelet-dft.h"
#include "dft/simd/t2b.h"

#if !defined(FFTW_SINGLE) && (R5V_SIZE == 256)

#include <riscv_vector.h>

#define HVL VL /* 2 complexes => 2 f64 lanes when split */

typedef vfloat64m1_t Hv;

static inline void LD2(const R *p, Hv *re, Hv *im)
{
     vfloat64m1x2_t s = __riscv_vlseg2e64_v_f64m1x2((const double *)p, HVL);
     *re = __riscv_vget_v_f64m1x2_f64m1(s, 0);
     *im = __riscv_vget_v_f64m1x2_f64m1(s, 1);
}

static inline void ST2(R *p, Hv re, Hv im)
{
     vfloat64m1x2_t s = __riscv_vcreate_v_f64m1x2(re, im);
     __riscv_vsseg2e64_v_f64m1x2((double *)p, s, HVL);
}

/* VTW2 @ t: [c0,c0,c1,c1, -s0,s0, -s1,s1] */
static inline void BYTW2s(const R *t, Hv sr, Hv si, Hv *or, Hv *oi)
{
     Hv wr = __riscv_vlse64_v_f64m1(t, (ptrdiff_t)(2 * sizeof(R)), HVL);
     Hv wi = __riscv_vlse64_v_f64m1(t + 5, (ptrdiff_t)(2 * sizeof(R)), HVL);
     Hv out_r = __riscv_vfmul_vv_f64m1(wr, sr, HVL);
     out_r = __riscv_vfnmsac_vv_f64m1(out_r, wi, si, HVL);
     Hv out_i = __riscv_vfmul_vv_f64m1(wr, si, HVL);
     out_i = __riscv_vfmacc_vv_f64m1(out_i, wi, sr, HVL);
     *or = out_r;
     *oi = out_i;
}

static inline Hv HADD(Hv a, Hv b) { return __riscv_vfadd_vv_f64m1(a, b, HVL); }
static inline Hv HSUB(Hv a, Hv b) { return __riscv_vfsub_vv_f64m1(a, b, HVL); }
static inline Hv HFMA(Hv a, Hv b, Hv c) { return __riscv_vfmacc_vv_f64m1(c, a, b, HVL); }
static inline Hv HFNMS(Hv a, Hv b, Hv c) { return __riscv_vfnmsac_vv_f64m1(c, a, b, HVL); }

static inline void VFMAI2(Hv br, Hv bi, Hv cr, Hv ci, Hv *or, Hv *oi)
{
     *or = HSUB(cr, bi);
     *oi = HADD(ci, br);
}
static inline void VFNMSI2(Hv br, Hv bi, Hv cr, Hv ci, Hv *or, Hv *oi)
{
     *or = HADD(cr, bi);
     *oi = HSUB(ci, br);
}

static void t2bv_8(R *ri, R *ii, const R *W, stride rs, INT mb, INT me, INT ms)
{
     const Hv k707 = __riscv_vfmv_v_f_f64m1(+0.70710678118654752440, HVL);
     INT m;
     R *x = ii;
     (void)ri;
     for (m = mb, W = W + (mb * ((TWVL / VL) * 14)); m < me;
          m = m + VL, x = x + (VL * ms), W = W + (TWVL * 14), MAKE_VOLATILE_STRIDE(8, rs)) {
          Hv T1r, T1i, T2r, T2i, T3r, T3i, T4r, T4i, Tqr, Tqi;
          Hv Tir, Tii, Tkr, Tki, Tlr, Tli, Trr, Tri;
          Hv T5r, T5i, T6r, T6i, T7r, T7i, T8r, T8i, T9r, T9i, Ttr, Tti;
          Hv Tar, Tai, Tbr, Tbi, Tcr, Tci, Tdr, Tdi, Ter, Tei, Tur, Tui;
          Hv Tsr, Tsi, Tvr, Tvi, Twr, Twi, Txr, Txi;
          Hv Tfr, Tfi, Tmr, Tmi, Tnr, Tni, Tpr, Tpi, Tgr, Tgi, Tor, Toi;

          LD2(&(x[0]), &T1r, &T1i);
          LD2(&(x[WS(rs, 4)]), &T2r, &T2i);
          BYTW2s(&(W[TWVL * 6]), T2r, T2i, &T3r, &T3i);
          T4r = HSUB(T1r, T3r); T4i = HSUB(T1i, T3i);
          Tqr = HADD(T1r, T3r); Tqi = HADD(T1i, T3i);

          LD2(&(x[WS(rs, 2)]), &Tir, &Tii);
          BYTW2s(&(W[TWVL * 2]), Tir, Tii, &Tir, &Tii);
          LD2(&(x[WS(rs, 6)]), &Tkr, &Tki);
          BYTW2s(&(W[TWVL * 10]), Tkr, Tki, &Tkr, &Tki);
          Tlr = HSUB(Tir, Tkr); Tli = HSUB(Tii, Tki);
          Trr = HADD(Tir, Tkr); Tri = HADD(Tii, Tki);

          LD2(&(x[WS(rs, 1)]), &T5r, &T5i);
          BYTW2s(&(W[0]), T5r, T5i, &T6r, &T6i);
          LD2(&(x[WS(rs, 5)]), &T7r, &T7i);
          BYTW2s(&(W[TWVL * 8]), T7r, T7i, &T8r, &T8i);
          T9r = HSUB(T6r, T8r); T9i = HSUB(T6i, T8i);
          Ttr = HADD(T6r, T8r); Tti = HADD(T6i, T8i);

          LD2(&(x[WS(rs, 7)]), &Tar, &Tai);
          BYTW2s(&(W[TWVL * 12]), Tar, Tai, &Tbr, &Tbi);
          LD2(&(x[WS(rs, 3)]), &Tcr, &Tci);
          BYTW2s(&(W[TWVL * 4]), Tcr, Tci, &Tdr, &Tdi);
          Ter = HSUB(Tbr, Tdr); Tei = HSUB(Tbi, Tdi);
          Tur = HADD(Tbr, Tdr); Tui = HADD(Tbi, Tdi);

          Tsr = HSUB(Tqr, Trr); Tsi = HSUB(Tqi, Tri);
          Tvr = HSUB(Ttr, Tur); Tvi = HSUB(Tti, Tui);
          VFNMSI2(Tvr, Tvi, Tsr, Tsi, &Twr, &Twi);
          ST2(&(x[WS(rs, 6)]), Twr, Twi);
          VFMAI2(Tvr, Tvi, Tsr, Tsi, &Txr, &Txi);
          ST2(&(x[WS(rs, 2)]), Txr, Txi);
          Twr = HADD(Tqr, Trr); Twi = HADD(Tqi, Tri);
          Txr = HADD(Ttr, Tur); Txi = HADD(Tti, Tui);
          ST2(&(x[WS(rs, 4)]), HSUB(Twr, Txr), HSUB(Twi, Txi));
          ST2(&(x[0]), HADD(Twr, Txr), HADD(Twi, Txi));

          Tfr = HADD(T9r, Ter); Tfi = HADD(T9i, Tei);
          Tgr = HFNMS(k707, Tfr, T4r); Tgi = HFNMS(k707, Tfi, T4i);
          Tor = HFMA(k707, Tfr, T4r); Toi = HFMA(k707, Tfi, T4i);
          Tmr = HSUB(T9r, Ter); Tmi = HSUB(T9i, Tei);
          Tnr = HFNMS(k707, Tmr, Tlr); Tni = HFNMS(k707, Tmi, Tli);
          Tpr = HFMA(k707, Tmr, Tlr); Tpi = HFMA(k707, Tmi, Tli);
          VFNMSI2(Tnr, Tni, Tgr, Tgi, &Twr, &Twi);
          ST2(&(x[WS(rs, 3)]), Twr, Twi);
          VFNMSI2(Tpr, Tpi, Tor, Toi, &Twr, &Twi);
          ST2(&(x[WS(rs, 7)]), Twr, Twi);
          VFMAI2(Tnr, Tni, Tgr, Tgi, &Twr, &Twi);
          ST2(&(x[WS(rs, 5)]), Twr, Twi);
          VFMAI2(Tpr, Tpi, Tor, Toi, &Twr, &Twi);
          ST2(&(x[WS(rs, 1)]), Twr, Twi);
     }
     VLEAVE();
}

static const tw_instr twinstr[] = {
     VTW(0, 1), VTW(0, 2), VTW(0, 3), VTW(0, 4),
     VTW(0, 5), VTW(0, 6), VTW(0, 7),
     {TW_NEXT, VL, 0}
};

static const ct_desc desc = {8, XSIMD_STRING("t2bv_8"), twinstr, &GENUS, {23, 14, 10, 0}, 0, 0, 0};

void XSIMD(codelet_t2bv_8)(planner *p)
{
     X(kdft_dit_register)(p, t2bv_8, &desc);
}

#else
#include "../common/t2bv_8.c"
#endif
