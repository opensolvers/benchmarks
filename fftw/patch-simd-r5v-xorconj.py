#!/usr/bin/env python3
"""Patch simd-r5v.h: XOR-based VCONJ (like SSE2) + fused VBYI/VFMAI/VFNMSI."""
from pathlib import Path
import sys

p = Path(sys.argv[1])
text = p.read_text()

old_conj_byi = """#if 1
static inline V VCONJ(const V x) {
	Vint idx = TYPEINT(vid_v)(2*VL); // (0, 1, 2, 3, ...)
	//Vint vone = TYPEINT(vmv_v_x)(1, 2*VL);
	Vint hidx = TYPEINT(vand_vx)(idx, 1, 2*VL); // (0, 1, 0, 1, 0, 1)
	//return TYPEMASK(vfsgnjn_vv)(x, x, x, INT2MASK(hidx), 2*VL);
	return TYPEMERGEDMASK(vfsgnjn_vv)(INT2MASK(hidx), x, x, x, 2*VL);
}
#else
static inline V VCONJ(const V x)
{
	Vint partr = VPARTSPLIT; // (all 1, 0, all 1, 0, ...)
	V xl = TYPEINTERPRETU2F(vreinterpret_v)(TYPEINT(vand_vv)(TYPEINTERPRETF2U(vreinterpret_v)(x), partr, 2*VL)); // set odd elements to 0
	Vint parti = TYPEINT(vnot_v)(partr, 2*VL); // (0, all 1, 0, all 1, ...)
	V xh = TYPEINTERPRETU2F(vreinterpret_v)(TYPEINT(vand_vv)(TYPEINTERPRETF2U(vreinterpret_v)(x), parti, 2*VL)); // set even elements to 0
	return VADD(xl, VNEG(xh));
}
#endif

/* can probably be done better */
static inline V VBYI(V x)
{
  x = VCONJ(x);
  x = FLIP_RI(x);
  return x;
}
"""

new_conj_byi = """/* XOR sign-bit onto imag lanes — same idea as simd-sse2.h VCONJ.
 * Avoids masked vfsgnjn + mask reg pressure; one vsll+vxor. */
static inline V VCONJ(const V x)
{
	Vint idx = TYPEINT(vid_v)(2*VL);
	Vint odd = TYPEINT(vand_vx)(idx, 1, 2*VL); /* 0,1,0,1,... */
	Vint signs = TYPEINT(vsll_vx)(odd, DS(63, 31), 2*VL);
	Vint xi = TYPEINTERPRETF2U(vreinterpret_v)(x);
	return TYPEINTERPRETU2F(vreinterpret_v)(TYPEINT(vxor_vv)(xi, signs, 2*VL));
}

/* (re,im) -> (-im, re): flip then XOR-negate even lanes (one gather). */
static inline V VBYI(V x)
{
	x = FLIP_RI(x); /* (im, re, ...) */
	Vint idx = TYPEINT(vid_v)(2*VL);
	/* 1 on even lanes, 0 on odd: (vid&1)^1 */
	Vint even = TYPEINT(vxor_vx)(TYPEINT(vand_vx)(idx, 1, 2*VL), 1, 2*VL);
	Vint signs = TYPEINT(vsll_vx)(even, DS(63, 31), 2*VL);
	Vint xi = TYPEINTERPRETF2U(vreinterpret_v)(x);
	return TYPEINTERPRETU2F(vreinterpret_v)(TYPEINT(vxor_vv)(xi, signs, 2*VL));
}
"""

if old_conj_byi not in text:
    sys.exit("VCONJ/VBYI block not found")
text = text.replace(old_conj_byi, new_conj_byi, 1)

old_fma = """#define VFMAI(b, c) VADD(c, VBYI(b)) // fixme: improve
#define VFNMSI(b, c) VSUB(c, VBYI(b)) // fixme: improve
#define VFMACONJ(b,c)  VADD(VCONJ(b),c) // fixme: improve
#define VFMSCONJ(b,c)  VFMACONJ(b,VNEG(c)) // fixme: improve
#define VFNMSCONJ(b,c) VNEG(VFMSCONJ(b,c)) // fixme: improve
"""

# VFMAI macros must come AFTER VBYI/VCONJ are defined — currently they're BEFORE.
# So we remove the early #defines and add inline functions after VBYI.
if old_fma not in text:
    sys.exit("VFMAI defines not found")
text = text.replace(old_fma, """/* VFMAI/VFNMSI/VFMACONJ: defined as inlines after VBYI/VCONJ below */
""", 1)

# Insert fused inlines right after VBYI block (end of new_conj_byi)
marker = new_conj_byi.rstrip() + "\n"
insert = marker + """
static inline V VFMAI(V b, V c) { return VADD(c, VBYI(b)); }
static inline V VFNMSI(V b, V c) { return VSUB(c, VBYI(b)); }
static inline V VFMACONJ(V b, V c) { return VADD(VCONJ(b), c); }
static inline V VFMSCONJ(V b, V c) { return VSUB(VCONJ(b), c); }
static inline V VFNMSCONJ(V b, V c) { return VSUB(c, VCONJ(b)); }

"""
if marker not in text:
    sys.exit("marker for insert not found")
text = text.replace(marker, insert, 1)

p.write_text(text)
print("patched", p)
