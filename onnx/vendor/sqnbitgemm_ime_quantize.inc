// Vectorized M=1 A-row quantize for BlkLen=32 (from llama quantize_a_row_i8).
#pragma once

#define IME_QUANTIZE_A_ROW_BL32_ASM                                          \
    "addi         t3, zero, 32*4          \n\t"                               \
    "addi         t2, zero, 32            \n\t"                               \
    "addi         a1, %[SRC], 0           \n\t"                               \
    "addi         a2, %[SRC], 128         \n\t"                               \
    "addi         a3, %[SRC], 256         \n\t"                               \
    "addi         a4, %[SRC], 384         \n\t"                               \
    "addi         s1, %[DST], 0           \n\t"                               \
    "addi         s2, %[DST], 36          \n\t"                               \
    "addi         s3, %[DST], 72          \n\t"                               \
    "addi         s4, %[DST], 108         \n\t"                               \
    "blt          %[K], t3, IME_QK%=      \n\t"                               \
    "blt          %[K], t2, IME_QT%=        \n\t"                               \
    "IME_QM%=:                             \n\t"                               \
    "vsetvli      t1, zero, e32, m4       \n\t"                               \
    "addi         %[K], %[K], -128        \n\t"                               \
    "vle32.v      v0, (a1)                \n\t"                               \
    "addi         a1, a1, 512             \n\t"                               \
    "vle32.v      v4, (a2)                \n\t"                               \
    "addi         a2, a2, 512             \n\t"                               \
    "vle32.v      v8, (a3)                \n\t"                               \
    "addi         a3, a3, 512             \n\t"                               \
    "vle32.v      v12, (a4)               \n\t"                               \
    "addi         a4, a4, 512             \n\t"                               \
    "vfabs.v      v16, v0                 \n\t"                               \
    "vfabs.v      v20, v4                 \n\t"                               \
    "vfabs.v      v24, v8                 \n\t"                               \
    "vfabs.v      v28, v12                \n\t"                               \
    "vsetvli      t0, zero, e32, m2       \n\t"                               \
    "vfmax.vv     v16, v16, v18           \n\t"                               \
    "vfmax.vv     v20, v20, v22           \n\t"                               \
    "vfmax.vv     v24, v24, v26           \n\t"                               \
    "vfmax.vv     v28, v28, v30           \n\t"                               \
    "vsetvli      t0, zero, e32, m1       \n\t"                               \
    "vfmax.vv     v16, v16, v17           \n\t"                               \
    "vfmax.vv     v20, v20, v21           \n\t"                               \
    "vfmax.vv     v24, v24, v25           \n\t"                               \
    "vfmax.vv     v28, v28, v29           \n\t"                               \
    "vfredmax.vs  v17, v16, v17           \n\t"                               \
    "vfredmax.vs  v21, v20, v21           \n\t"                               \
    "vfredmax.vs  v25, v24, v25           \n\t"                               \
    "vfredmax.vs  v29, v28, v29           \n\t"                               \
    "vfmv.f.s     f10,  v17               \n\t"                               \
    "vfmv.f.s     f11,  v21               \n\t"                               \
    "vfmv.f.s     f12,  v25               \n\t"                               \
    "vfmv.f.s     f13,  v29               \n\t"                               \
    "fmul.s       f10, f10, %[RMAXREC]    \n\t"                               \
    "fmul.s       f11, f11, %[RMAXREC]    \n\t"                               \
    "fmul.s       f12, f12, %[RMAXREC]    \n\t"                               \
    "fmul.s       f13, f13, %[RMAXREC]    \n\t"                               \
    "fsw          f10, (s1)               \n\t"                               \
    "addi         s1, s1, 4               \n\t"                               \
    "fsw          f11, (s2)               \n\t"                               \
    "addi         s2, s2, 4               \n\t"                               \
    "fsw          f12, (s3)               \n\t"                               \
    "addi         s3, s3, 4               \n\t"                               \
    "fsw          f13, (s4)               \n\t"                               \
    "addi         s4, s4, 4               \n\t"                               \
    "fdiv.s       f10, %[FONE], f10       \n\t"                               \
    "fdiv.s       f11, %[FONE], f11       \n\t"                               \
    "fdiv.s       f12, %[FONE], f12       \n\t"                               \
    "fdiv.s       f13, %[FONE], f13       \n\t"                               \
    "vsetvli      t0, zero, e32, m4       \n\t"                               \
    "vfmul.vf     v16, v0, f10            \n\t"                               \
    "vfmul.vf     v20, v4, f11            \n\t"                               \
    "vfmul.vf     v24, v8, f12            \n\t"                               \
    "vfmul.vf     v28, v12, f13           \n\t"                               \
    "vfcvt.x.f.v  v16, v16                \n\t"                               \
    "vfcvt.x.f.v  v20, v20                \n\t"                               \
    "vfcvt.x.f.v  v24, v24                \n\t"                               \
    "vfcvt.x.f.v  v28, v28                \n\t"                               \
    "vsetvli      t0, zero, e16, m2       \n\t"                               \
    "vnclip.wx    v16, v16, zero          \n\t"                               \
    "vnclip.wx    v20, v20, zero          \n\t"                               \
    "vnclip.wx    v24, v24, zero          \n\t"                               \
    "vnclip.wx    v28, v28, zero          \n\t"                               \
    "vsetvli      t0, t1, e8, m1          \n\t"                               \
    "vnclip.wx    v16, v16, zero          \n\t"                               \
    "vnclip.wx    v20, v20, zero          \n\t"                               \
    "vnclip.wx    v24, v24, zero          \n\t"                               \
    "vnclip.wx    v28, v28, zero          \n\t"                               \
    "vse8.v       v16, (s1)               \n\t"                               \
    "addi         s1, s1, 140             \n\t"                               \
    "vse8.v       v20, (s2)               \n\t"                               \
    "addi         s2, s2, 140             \n\t"                               \
    "vse8.v       v24, (s3)               \n\t"                               \
    "addi         s3, s3, 140             \n\t"                               \
    "vse8.v       v28, (s4)               \n\t"                               \
    "addi         s4, s4, 140             \n\t"                               \
    "bge          %[K], t3, IME_QM%=      \n\t"                               \
    "blt          %[K], t2, IME_QT%=        \n\t"                               \
    "IME_QK%=:                            \n\t"                               \
    "vsetvli      t1, %[K], e32, m4       \n\t"                               \
    "vle32.v      v0, (a1)                \n\t"                               \
    "addi         a1, a1, 128             \n\t"                               \
    "sub          %[K], %[K], t1          \n\t"                               \
    "vfabs.v      v16, v0                 \n\t"                               \
    "vsetvli      t0, zero, e32, m2       \n\t"                               \
    "vfmax.vv     v16, v16, v18           \n\t"                               \
    "vsetvli      t0, zero, e32, m1       \n\t"                               \
    "vfmax.vv     v16, v16, v17           \n\t"                               \
    "vfredmax.vs  v17, v16, v17           \n\t"                               \
    "vfmv.f.s     f10,  v17               \n\t"                               \
    "fmul.s       f10, f10, %[RMAXREC]    \n\t"                               \
    "fsw          f10, (s1)               \n\t"                               \
    "addi         s1, s1, 4               \n\t"                               \
    "fdiv.s       f11, %[FONE], f10       \n\t"                               \
    "vsetvli      t0, zero, e32, m4       \n\t"                               \
    "vfmul.vf     v16, v0, f11            \n\t"                               \
    "vfcvt.x.f.v  v16, v16                \n\t"                               \
    "vsetvli      t0, zero, e16, m2       \n\t"                               \
    "vnclip.wx    v16, v16, zero          \n\t"                               \
    "vsetvli      t0, zero, e8, m1        \n\t"                               \
    "vnclip.wx    v16, v16, zero          \n\t"                               \
    "vse8.v       v16, (s1)               \n\t"                               \
    "addi         s1, s1, 32              \n\t"                               \
    "bge          %[K], t2, IME_QK%=      \n\t"                               \
    "IME_QT%=:                              \n\t"                               \
    "blez         %[K], IME_QE%=             \n\t"                               \
    "vsetvli      t0, t3, e32, m4         \n\t"                               \
    "vxor.vv      v0, v0, v0              \n\t"                               \
    "vxor.vv      v16, v16, v16           \n\t"                               \
    "jal          x0, IME_QK%=            \n\t"                               \
    "IME_QE%=:                             \n\t"
