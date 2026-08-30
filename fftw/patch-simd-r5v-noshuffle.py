#!/usr/bin/env python3
"""Patch simd-r5v.h: avoid vrgather on VLEN=256 double (VL=2) hot helpers."""
from pathlib import Path
import sys

p = Path(sys.argv[1])
text = p.read_text()
old = """static inline V VDUPL(const V x) {
	Vint idx = TYPEINT(vid_v)(2*VL); // (0, 1, 2, 3, ...)
	//Vint vnotone = TYPEINT(vmv_v_x)(DS(~1ull,~1), 2*VL);
	Vint hidx = TYPEINT(vand_vx)(idx, DS(~1ull,~1), 2*VL); // (0, 0, 2, 2, ...)
	return TYPE(vrgather_vv)(x, hidx, 2*VL);
}
static inline V VDUPH(const V x) {
	Vint idx = TYPEINT(vid_v)(2*VL); // (0, 1, 2, 3, ...)
	//Vint vone = TYPEINT(vmv_v_x)(1, 2*VL);
	Vint hidx = TYPEINT(vor_vx)(idx, 1, 2*VL); // (1, 1, 3, 3, ...)
	return TYPE(vrgather_vv)(x, hidx, 2*VL);
}

static inline V FLIP_RI(const V x) {
	Vint idx = TYPEINT(vid_v)(2*VL); // (0, 1, 2, 3, ...)
	//Vint vone = TYPEINT(vmv_v_x)(1, 2*VL);
	Vint hidx = TYPEINT(vxor_vx)(idx, 1, 2*VL); // (1, 0, 3, 2, ...)
	return TYPE(vrgather_vv)(x, hidx, 2*VL);
}
"""
new = """/* SpacemiT K1/X60: vrgather is costly even at LMUL=1. For native VLEN=256
 * double (VL=2 complex => 4 f64 lanes), store/shuffle/load beats gather for
 * the hot DUxx/FLIP/VBYI helpers used by VZMUL. Larger R5V_SIZE keeps gather. */
#if (R5V_SIZE == 256) && !defined(FFTW_SINGLE)
static inline V VDUPL(const V x) {
  R a[4] __attribute__((aligned(32)));
  __riscv_vse64_v_f64m1(a, x, 4);
  a[1] = a[0];
  a[3] = a[2];
  return __riscv_vle64_v_f64m1(a, 4);
}
static inline V VDUPH(const V x) {
  R a[4] __attribute__((aligned(32)));
  __riscv_vse64_v_f64m1(a, x, 4);
  a[0] = a[1];
  a[2] = a[3];
  return __riscv_vle64_v_f64m1(a, 4);
}
static inline V FLIP_RI(const V x) {
  R a[4] __attribute__((aligned(32)));
  __riscv_vse64_v_f64m1(a, x, 4);
  R t0 = a[0]; a[0] = a[1]; a[1] = t0;
  R t2 = a[2]; a[2] = a[3]; a[3] = t2;
  return __riscv_vle64_v_f64m1(a, 4);
}
static inline V VBYI(V x) {
  /* (re,im) -> (-im, re) without gather */
  R a[4] __attribute__((aligned(32)));
  __riscv_vse64_v_f64m1(a, x, 4);
  R re0 = a[0], im0 = a[1], re1 = a[2], im1 = a[3];
  a[0] = -im0; a[1] = re0; a[2] = -im1; a[3] = re1;
  return __riscv_vle64_v_f64m1(a, 4);
}
#else /* portable gather path */
static inline V VDUPL(const V x) {
	Vint idx = TYPEINT(vid_v)(2*VL); // (0, 1, 2, 3, ...)
	Vint hidx = TYPEINT(vand_vx)(idx, DS(~1ull,~1), 2*VL); // (0, 0, 2, 2, ...)
	return TYPE(vrgather_vv)(x, hidx, 2*VL);
}
static inline V VDUPH(const V x) {
	Vint idx = TYPEINT(vid_v)(2*VL); // (0, 1, 2, 3, ...)
	Vint hidx = TYPEINT(vor_vx)(idx, 1, 2*VL); // (1, 1, 3, 3, ...)
	return TYPE(vrgather_vv)(x, hidx, 2*VL);
}

static inline V FLIP_RI(const V x) {
	Vint idx = TYPEINT(vid_v)(2*VL); // (0, 1, 2, 3, ...)
	Vint hidx = TYPEINT(vxor_vx)(idx, 1, 2*VL); // (1, 0, 3, 2, ...)
	return TYPE(vrgather_vv)(x, hidx, 2*VL);
}
#endif /* R5V_SIZE == 256 && !FFTW_SINGLE */
"""
if old not in text:
    sys.exit("VDUPL/FLIP pattern not found")
text = text.replace(old, new, 1)
old_vbyi = """/* can probably be done better */
static inline V VBYI(V x)
{
  x = VCONJ(x);
  x = FLIP_RI(x);
  return x;
}
"""
new_vbyi = """#if !((R5V_SIZE == 256) && !defined(FFTW_SINGLE))
/* can probably be done better */
static inline V VBYI(V x)
{
  x = VCONJ(x);
  x = FLIP_RI(x);
  return x;
}
#endif
"""
if old_vbyi not in text:
    sys.exit("VBYI pattern not found")
text = text.replace(old_vbyi, new_vbyi, 1)
p.write_text(text)
print("patched", p)
