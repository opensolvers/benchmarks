#!/usr/bin/env python3
"""Patch simd-r5v.h: FLIP_RI via vslide+vmerge (R5V_SIZE=256 double only).

t2bv_8 / BYTW2 are gather-bound on FLIP_RI. Try slide+merge instead of
vrgather for pairwise (re,im) swap. Leaves VDUPL/VDUPH on gather.
Requires xorconj base (VBYI calls FLIP_RI).
"""
from pathlib import Path
import sys

p = Path(sys.argv[1])
text = p.read_text()

old = """static inline V FLIP_RI(const V x) {
	Vint idx = TYPEINT(vid_v)(2*VL); // (0, 1, 2, 3, ...)
	//Vint vone = TYPEINT(vmv_v_x)(1, 2*VL);
	Vint hidx = TYPEINT(vxor_vx)(idx, 1, 2*VL); // (1, 0, 3, 2, ...)
	return TYPE(vrgather_vv)(x, hidx, 2*VL);
}
"""

new = """/* Pairwise swap (r,i)->(i,r). On X60, vrgather is costly; for native
 * VLEN=256 double (4 f64 lanes) try slide+merge. Other R5V_SIZE keep gather. */
#if (R5V_SIZE == 256) && !defined(FFTW_SINGLE)
static inline V FLIP_RI(const V x)
{
	/* down=[i0,r1,i1,?], up=[?,r0,i0,r1]; even<-down, odd<-up */
	V down = __riscv_vslidedown_vx_f64m1(x, 1, 4);
	V up = __riscv_vslideup_vx_f64m1(x, x, 1, 4);
	Vint idx = __riscv_vid_v_u64m1(4);
	Vint odd = __riscv_vand_vx_u64m1(idx, 1, 4);
	/* mask=1 on even lanes (idx&1==0) → take down; odd → take up */
	vbool64_t even = __riscv_vmseq_vx_u64m1_b64(odd, 0, 4);
	/* vmerge: mask ? op1 : op2  with __riscv_vmerge_vvm(op2, op1, mask, vl) */
	return __riscv_vmerge_vvm_f64m1(up, down, even, 4);
}
#else
static inline V FLIP_RI(const V x) {
	Vint idx = TYPEINT(vid_v)(2*VL); // (0, 1, 2, 3, ...)
	//Vint vone = TYPEINT(vmv_v_x)(1, 2*VL);
	Vint hidx = TYPEINT(vxor_vx)(idx, 1, 2*VL); // (1, 0, 3, 2, ...)
	return TYPE(vrgather_vv)(x, hidx, 2*VL);
}
#endif
"""

if old not in text:
    sys.exit("FLIP_RI block not found (wrong base or already patched)")
text = text.replace(old, new, 1)
p.write_text(text)
print("patched", p)
