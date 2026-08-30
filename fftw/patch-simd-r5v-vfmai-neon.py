#!/usr/bin/env python3
"""Patch simd-r5v.h: NEON-style VFMAI/VFNMSI/VZMUL via FLIP * (-1,1).

Requires XOR-conj patch already applied. Replaces VADD(c,VBYI(b)) with
VFMA(FLIP(b), mp, c) so the sign vector can be CSE'd / hoisted off the
gather-critical path (same as simd-neon.h).
"""
from pathlib import Path
import sys

p = Path(sys.argv[1])
text = p.read_text()

old = """static inline V VFMAI(V b, V c) { return VADD(c, VBYI(b)); }
static inline V VFNMSI(V b, V c) { return VSUB(c, VBYI(b)); }
static inline V VFMACONJ(V b, V c) { return VADD(VCONJ(b), c); }
static inline V VFMSCONJ(V b, V c) { return VSUB(VCONJ(b), c); }
static inline V VFNMSCONJ(V b, V c) { return VSUB(c, VCONJ(b)); }


#if 1
static inline V VZMUL(V tx, V sr)
{
    V tr = VDUPL(tx);
    V ti = VDUPH(tx);
    tr = VMUL(sr, tr);
    sr = VBYI(sr);
    return VFMA(ti, sr, tr);
}

static inline V VZMULJ(V tx, V sr)
{
    V tr = VDUPL(tx);
    V ti = VDUPH(tx);
    tr = VMUL(sr, tr);
    sr = VBYI(sr);
    return VFNMS(ti, sr, tr);
}

static inline V VZMULI(V tx, V sr)
{
    V tr = VDUPL(tx);
    V ti = VDUPH(tx);
    ti = VMUL(ti, sr);
    sr = VBYI(sr);
    return VFMS(tr, sr, ti);
}

static inline V VZMULIJ(V tx, V sr)
{
    V tr = VDUPL(tx);
    V ti = VDUPH(tx);
    ti = VMUL(ti, sr);
    sr = VBYI(sr);
    return VFMA(tr, sr, ti);
}
#else
"""

new = """/* Lane sign vectors: (-1,+1,...) and (+1,-1,...). Built like XOR-VCONJ. */
static inline V VSIGN_MP(void) /* (-1, +1, -1, +1, ...) */
{
	V one = VLIT1(DS(1., 1.f));
	Vint idx = TYPEINT(vid_v)(2*VL);
	Vint even = TYPEINT(vxor_vx)(TYPEINT(vand_vx)(idx, 1, 2*VL), 1, 2*VL);
	Vint signs = TYPEINT(vsll_vx)(even, DS(63, 31), 2*VL);
	Vint xi = TYPEINTERPRETF2U(vreinterpret_v)(one);
	return TYPEINTERPRETU2F(vreinterpret_v)(TYPEINT(vxor_vv)(xi, signs, 2*VL));
}
static inline V VSIGN_PM(void) /* (+1, -1, +1, -1, ...) */
{
	V one = VLIT1(DS(1., 1.f));
	Vint idx = TYPEINT(vid_v)(2*VL);
	Vint odd = TYPEINT(vand_vx)(idx, 1, 2*VL);
	Vint signs = TYPEINT(vsll_vx)(odd, DS(63, 31), 2*VL);
	Vint xi = TYPEINTERPRETF2U(vreinterpret_v)(one);
	return TYPEINTERPRETU2F(vreinterpret_v)(TYPEINT(vxor_vv)(xi, signs, 2*VL));
}

/* NEON-style: c ± i*b = c ± FLIP(b)*(-1,+1) — one gather, FMA, hoistable mp. */
static inline V VFMAI(V b, V c)
{
	return VFMA(FLIP_RI(b), VSIGN_MP(), c);
}
static inline V VFNMSI(V b, V c)
{
	return VFNMS(FLIP_RI(b), VSIGN_MP(), c);
}
static inline V VFMACONJ(V b, V c)
{
	return VFMA(b, VSIGN_PM(), c);
}
static inline V VFMSCONJ(V b, V c)
{
	return VFMS(b, VSIGN_PM(), c);
}
static inline V VFNMSCONJ(V b, V c)
{
	return VFNMS(b, VSIGN_PM(), c);
}


#if 1
/* VBYI(sr) ≡ FLIP(sr)*VSIGN_MP(); keep gathers at 3, drop post-FLIP XOR. */
static inline V VZMUL(V tx, V sr)
{
	V tr = VDUPL(tx);
	V ti = VDUPH(tx);
	tr = VMUL(sr, tr);
	sr = VMUL(FLIP_RI(sr), VSIGN_MP());
	return VFMA(ti, sr, tr);
}

static inline V VZMULJ(V tx, V sr)
{
	V tr = VDUPL(tx);
	V ti = VDUPH(tx);
	tr = VMUL(sr, tr);
	sr = VMUL(FLIP_RI(sr), VSIGN_MP());
	return VFNMS(ti, sr, tr);
}

static inline V VZMULI(V tx, V sr)
{
	V tr = VDUPL(tx);
	V ti = VDUPH(tx);
	ti = VMUL(ti, sr);
	sr = VMUL(FLIP_RI(sr), VSIGN_MP());
	return VFMS(tr, sr, ti);
}

static inline V VZMULIJ(V tx, V sr)
{
	V tr = VDUPL(tx);
	V ti = VDUPH(tx);
	ti = VMUL(ti, sr);
	sr = VMUL(FLIP_RI(sr), VSIGN_MP());
	return VFMA(tr, sr, ti);
}
#else
"""

if old not in text:
    sys.exit("VFMAI/VZMUL block not found (need xorconj base, or already patched)")
text = text.replace(old, new, 1)
p.write_text(text)
print("patched", p)
