// llama M1 ScaleFp16 no-ZP no-bias panel macros (BlkLen=32), smt.vmadot.
#pragma once

#define SQ4BIT_KERNEL_COMP_1x8x2_4X8X4          \
    "smt.vmadot   v16, v14, v0            \n\t"   \
    "smt.vmadot   v18, v14, v1            \n\t"   \
    "smt.vmadot   v20, v14, v2            \n\t"   \
    "smt.vmadot   v22, v14, v3            \n\t"   \
    "smt.vmadot   v16, v15, v4            \n\t"   \
    "smt.vmadot   v18, v15, v5            \n\t"   \
    "smt.vmadot   v20, v15, v6            \n\t"   \
    "smt.vmadot   v22, v15, v7            \n\t"

#define SQ4BIT_KERNEL_ACC_F16_1X4X4             \
    "vfcvt.f.x.v  v16,  v16               \n\t"   \
    "vfcvt.f.x.v  v18,  v18               \n\t"   \
    "vfcvt.f.x.v  v20,  v20               \n\t"   \
    "vfcvt.f.x.v  v22,  v22               \n\t"   \
    "addi         s2, s1, 8               \n\t"   \
    "addi         s3, s1, 16              \n\t"   \
    "addi         s4, s1, 24              \n\t"   \
    "addi         s6, s5, 12              \n\t"   \
    "vfmacc.vv    v28, v16, v24           \n\t"   \
    "vfmacc.vv    v29, v18, v25           \n\t"   \
    "vfmacc.vv    v30, v20, v26           \n\t"   \
    "vfmacc.vv    v31, v22, v27           \n\t"

#define SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4          \
    "vle8.v       v4, (s1)                \n\t"   \
    "addi         s1, s1, 128             \n\t"   \
    "vle8.v       v5, (s2)                \n\t"   \
    "addi         s2, s2, 128             \n\t"   \
    "vle8.v       v6, (s3)                \n\t"   \
    "addi         s3, s3, 128             \n\t"   \
    "vle8.v       v7, (s4)                \n\t"   \
    "addi         s4, s4, 128             \n\t"   \
    "vsetvli      t0, zero, e8, mf4       \n\t"   \
    "vle8.v       v14, (s5)               \n\t"   \
    "addi         s5, s5, 16              \n\t"   \
    "vle8.v       v15, (s6)               \n\t"   \
    "addi         s6, s6, 16              \n\t"   \
    "addi         t5, t5, -1              \n\t"   \
    "vsetvli      t0, zero, e8, m1        \n\t"   \
    "vand.vi      v0, v4, 15              \n\t"   \
    "vand.vi      v1, v5, 15              \n\t"   \
    "vand.vi      v2, v6, 15              \n\t"   \
    "vand.vi      v3, v7, 15              \n\t"   \
    "vsrl.vi      v4, v4, 4               \n\t"   \
    "vsrl.vi      v5, v5, 4               \n\t"   \
    "vsrl.vi      v6, v6, 4               \n\t"   \
    "vsrl.vi      v7, v7, 4               \n\t"

// Q8×16 M1: 8 contiguous 32B B tiles (slice-lo g0..3, slice-hi g0..3).
#define SQ8BIT_KERNEL_LOAD_B_16x8x2_I8           \
    "vsetvli      t0, zero, e8, m1        \n\t"   \
    "vle8.v       v0, (s1)                \n\t"   \
    "addi         s1, s1, 32              \n\t"   \
    "vle8.v       v1, (s1)                \n\t"   \
    "addi         s1, s1, 32              \n\t"   \
    "vle8.v       v2, (s1)                \n\t"   \
    "addi         s1, s1, 32              \n\t"   \
    "vle8.v       v3, (s1)                \n\t"   \
    "addi         s1, s1, 32              \n\t"   \
    "vle8.v       v4, (s1)                \n\t"   \
    "addi         s1, s1, 32              \n\t"   \
    "vle8.v       v5, (s1)                \n\t"   \
    "addi         s1, s1, 32              \n\t"   \
    "vle8.v       v6, (s1)                \n\t"   \
    "addi         s1, s1, 32              \n\t"   \
    "vle8.v       v7, (s1)                \n\t"   \
    "addi         s1, s1, 32              \n\t"

#define SQ8BIT_KERNEL_LOAD_A_1x8x2               \
    "vsetvli      t0, zero, e8, mf4       \n\t"   \
    "vle8.v       v14, (s5)               \n\t"   \
    "addi         s5, s5, 16              \n\t"   \
    "vle8.v       v15, (s6)               \n\t"   \
    "addi         s6, s6, 16              \n\t"   \
    "addi         t5, t5, -1              \n\t"   \
    "vsetvli      t0, zero, e8, m1        \n\t"

#define SQ8BIT_KERNEL_COMP_1x8x2_4X8X4           \
    "smt.vmadot   v16, v14, v0            \n\t"   \
    "smt.vmadot   v18, v14, v1            \n\t"   \
    "smt.vmadot   v20, v14, v2            \n\t"   \
    "smt.vmadot   v22, v14, v3            \n\t"   \
    "smt.vmadot   v16, v15, v4            \n\t"   \
    "smt.vmadot   v18, v15, v5            \n\t"   \
    "smt.vmadot   v20, v15, v6            \n\t"   \
    "smt.vmadot   v22, v15, v7            \n\t"
