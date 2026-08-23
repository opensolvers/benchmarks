/*
 * gemm_ime.c - SpaceMiT X60 IME (XsmtVdot v1.0) int8 GEMM microkernel.
 *
 * One smt.vmadot does a 4x4 int32 += (4x8 int8)*(4x8 int8)^T tile (vl=32,e8 at
 * VLEN=256). A/B are packed into contiguous 4x8 int8 tiles (shared with
 * gemm_packed_ref, so the layout is verified off-target). Two kernels:
 *   - ime_block_8x16: register-blocked main path. 8 accumulator pairs (v16..v31)
 *     hold an 8x16 output; each K-step loads 2 A + 4 B tiles and issues 8
 *     vmadots, so every load feeds several MACs (vs one in the edge kernel).
 *   - ime_tile: the plain 4x4 kernel, used only for edge blocks.
 *
 * smt.vmadot has no compiler intrinsic and (pre-binutils-2.46) no mnemonic, so
 * it is emitted as a raw instruction word via .insn - see SMT_VMADOT. This
 * builds on any RVV-capable binutils; qemu-user does not emulate it (X60 only).
 * SPDX-License-Identifier: MIT
 */
#include "gemm_s8s8s32.h"

#if defined(__riscv) && !defined(GEMM_NO_IME)

/* Emit `smt.vmadot vd, vs1, vs2` (i8) as a raw word, letting the assembler
 * compute the encoding: base 0xe200302b | (vd/2)<<8 | vs1<<15 | vs2<<20.
 * Cross-checked against LLVM's published encoding for v16,v0,v8 (0xe280382b). */
#define GEMM_STR2(x) #x
#define GEMM_STR1(x) GEMM_STR2(x)
#define SMT_VMADOT(vd, vs1, vs2)                                                \
    ".insn 4, 0xe200302b|((" GEMM_STR1(vd) "/2)<<8)|((" GEMM_STR1(vs1)          \
    ")<<15)|((" GEMM_STR1(vs2) ")<<20)\n\t"

/* One K-tile on operand bank v8–v13 (interleaved load/issue). */
#define IME_KSTEP_V813                                                       \
    "vle8.v      v8,  (%[a0])                 \n\t"                           \
    "addi        %[a0], %[a0], 32             \n\t"                           \
    "vle8.v      v9,  (%[a1])                 \n\t"                           \
    "addi        %[a1], %[a1], 32             \n\t"                           \
    "vle8.v      v10, (%[b0])                 \n\t"                           \
    "addi        %[b0], %[b0], 32             \n\t"                           \
    SMT_VMADOT(16, 8, 10)                                                    \
    "vle8.v      v11, (%[b1])                 \n\t"                           \
    "addi        %[b1], %[b1], 32             \n\t"                           \
    SMT_VMADOT(18, 8, 11)                                                    \
    "vle8.v      v12, (%[b2])                 \n\t"                           \
    "addi        %[b2], %[b2], 32             \n\t"                           \
    SMT_VMADOT(20, 8, 12)                                                    \
    "vle8.v      v13, (%[b3])                 \n\t"                           \
    "addi        %[b3], %[b3], 32             \n\t"                           \
    SMT_VMADOT(22, 8, 13)                                                    \
    SMT_VMADOT(24, 9, 10)                                                    \
    SMT_VMADOT(26, 9, 11)                                                    \
    SMT_VMADOT(28, 9, 12)                                                    \
    SMT_VMADOT(30, 9, 13)                                                    \

/* One K-tile on operand bank v0–v5. */
#define IME_KSTEP_V05                                                        \
    "vle8.v      v8,  (%[a0])                 \n\t"                           \
    "addi        %[a0], %[a0], 32             \n\t"                           \
    "vle8.v      v9,  (%[a1])                 \n\t"                           \
    "addi        %[a1], %[a1], 32             \n\t"                           \
    "vle8.v      v10, (%[b0])                 \n\t"                           \
    "addi        %[b0], %[b0], 32             \n\t"                           \
    SMT_VMADOT(16, 0, 2)                                                     \
    "vle8.v      v11, (%[b1])                 \n\t"                           \
    "addi        %[b1], %[b1], 32             \n\t"                           \
    SMT_VMADOT(18, 0, 3)                                                     \
    "vle8.v      v12, (%[b2])                 \n\t"                           \
    "addi        %[b2], %[b2], 32             \n\t"                           \
    SMT_VMADOT(20, 0, 4)                                                     \
    "vle8.v      v13, (%[b3])                 \n\t"                           \
    "addi        %[b3], %[b3], 32             \n\t"                           \
    SMT_VMADOT(22, 0, 5)                                                     \
    SMT_VMADOT(24, 1, 2)                                                     \
    SMT_VMADOT(26, 1, 3)                                                     \
    SMT_VMADOT(28, 1, 4)                                                     \
    SMT_VMADOT(30, 1, 5)                                                     \

/* Prefetch next K-tile into v0–v5 while operands for vmadot stay in v8–v13. */
#define IME_KSTEP_V813_PREF05                                                \
    "vle8.v      v0,  (%[a0])                 \n\t"                           \
    "addi        %[a0], %[a0], 32             \n\t"                           \
    "vle8.v      v1,  (%[a1])                 \n\t"                           \
    "addi        %[a1], %[a1], 32             \n\t"                           \
    SMT_VMADOT(16, 8, 10)                                                    \
    "vle8.v      v2,  (%[b0])                 \n\t"                           \
    "addi        %[b0], %[b0], 32             \n\t"                           \
    SMT_VMADOT(18, 8, 11)                                                    \
    "vle8.v      v3,  (%[b1])                 \n\t"                           \
    "addi        %[b1], %[b1], 32             \n\t"                           \
    SMT_VMADOT(20, 8, 12)                                                    \
    "vle8.v      v4,  (%[b2])                 \n\t"                           \
    "addi        %[b2], %[b2], 32             \n\t"                           \
    SMT_VMADOT(22, 8, 13)                                                    \
    "vle8.v      v5,  (%[b3])                 \n\t"                           \
    "addi        %[b3], %[b3], 32             \n\t"                           \
    SMT_VMADOT(24, 9, 10)                                                    \
    SMT_VMADOT(26, 9, 11)                                                    \
    SMT_VMADOT(28, 9, 12)                                                    \
    SMT_VMADOT(30, 9, 13)                                                    \

/* Prefetch next K-tile into v8–v13 while computing on v0–v5. */
#define IME_KSTEP_V05_PREF813                                                \
    "vle8.v      v8,  (%[a0])                 \n\t"                           \
    "addi        %[a0], %[a0], 32             \n\t"                           \
    "vle8.v      v9,  (%[a1])                 \n\t"                           \
    "addi        %[a1], %[a1], 32             \n\t"                           \
    SMT_VMADOT(16, 0, 2)                                                     \
    "vle8.v      v10, (%[b0])                 \n\t"                           \
    "addi        %[b0], %[b0], 32             \n\t"                           \
    SMT_VMADOT(18, 0, 3)                                                     \
    "vle8.v      v11, (%[b1])                 \n\t"                           \
    "addi        %[b1], %[b1], 32             \n\t"                           \
    SMT_VMADOT(20, 0, 4)                                                     \
    "vle8.v      v12, (%[b2])                 \n\t"                           \
    "addi        %[b2], %[b2], 32             \n\t"                           \
    SMT_VMADOT(22, 0, 5)                                                     \
    "vle8.v      v13, (%[b3])                 \n\t"                           \
    "addi        %[b3], %[b3], 32             \n\t"                           \
    SMT_VMADOT(24, 1, 2)                                                     \
    SMT_VMADOT(26, 1, 3)                                                     \
    SMT_VMADOT(28, 1, 4)                                                     \
    SMT_VMADOT(30, 1, 5)                                                     \

/* Compute one K-tile already resident in v0–v5 (no loads). */
#define IME_KCOMPUTE_V05                                                     \
    SMT_VMADOT(16, 0, 2)                                                     \
    SMT_VMADOT(18, 0, 3)                                                     \
    SMT_VMADOT(20, 0, 4)                                                     \
    SMT_VMADOT(22, 0, 5)                                                     \
    SMT_VMADOT(24, 1, 2)                                                     \
    SMT_VMADOT(26, 1, 3)                                                     \
    SMT_VMADOT(28, 1, 4)                                                     \
    SMT_VMADOT(30, 1, 5)                                                     \

/* Compute one K-tile already resident in v8–v13 (no loads). */
#define IME_KCOMPUTE_V813                                                    \
    SMT_VMADOT(16, 8, 10)                                                    \
    SMT_VMADOT(18, 8, 11)                                                    \
    SMT_VMADOT(20, 8, 12)                                                    \
    SMT_VMADOT(22, 8, 13)                                                    \
    SMT_VMADOT(24, 9, 10)                                                    \
    SMT_VMADOT(26, 9, 11)                                                    \
    SMT_VMADOT(28, 9, 12)                                                    \
    SMT_VMADOT(30, 9, 13)                                                    \

/* One K-tile: 4×32 panel (1 A tile reused across 8 B tiles). */
#define IME_KSTEP_4x32                                                       \
    "vle8.v      v8,  (%[a0])                 \n\t"                           \
    "addi        %[a0], %[a0], 32             \n\t"                           \
    "vle8.v      v9,  (%[b0])                 \n\t"                           \
    "addi        %[b0], %[b0], 32             \n\t"                           \
    SMT_VMADOT(16, 8, 9)                                                     \
    "vle8.v      v9,  (%[b1])                 \n\t"                           \
    "addi        %[b1], %[b1], 32             \n\t"                           \
    SMT_VMADOT(18, 8, 9)                                                     \
    "vle8.v      v9,  (%[b2])                 \n\t"                           \
    "addi        %[b2], %[b2], 32             \n\t"                           \
    SMT_VMADOT(20, 8, 9)                                                     \
    "vle8.v      v9,  (%[b3])                 \n\t"                           \
    "addi        %[b3], %[b3], 32             \n\t"                           \
    SMT_VMADOT(22, 8, 9)                                                     \
    "vle8.v      v9,  (%[b4])                 \n\t"                           \
    "addi        %[b4], %[b4], 32             \n\t"                           \
    SMT_VMADOT(24, 8, 9)                                                     \
    "vle8.v      v9,  (%[b5])                 \n\t"                           \
    "addi        %[b5], %[b5], 32             \n\t"                           \
    SMT_VMADOT(26, 8, 9)                                                     \
    "vle8.v      v9,  (%[b6])                 \n\t"                           \
    "addi        %[b6], %[b6], 32             \n\t"                           \
    SMT_VMADOT(28, 8, 9)                                                     \
    "vle8.v      v9,  (%[b7])                 \n\t"                           \
    "addi        %[b7], %[b7], 32             \n\t"                           \
    SMT_VMADOT(30, 8, 9)                                                     \

#define IME_ZERO_ACC_8x16                                                    \
    "vsetvli     t4, zero, e8, m8, ta, ma    \n\t"                           \
    "vmv.v.i     v16, 0                       \n\t"                           \
    "vmv.v.i     v24, 0                       \n\t"                           \
    "li          t1, 32                       \n\t"                           \
    "vsetvli     zero, t1, e8, m1, ta, ma     \n\t"
#define IME_KLOAD_V813                                                       \
    "vle8.v      v8,  (%[a0])                 \n\t"                           \
    "addi        %[a0], %[a0], 32             \n\t"                           \
    "vle8.v      v9,  (%[a1])                 \n\t"                           \
    "addi        %[a1], %[a1], 32             \n\t"                           \
    "vle8.v      v10, (%[b0])                 \n\t"                           \
    "addi        %[b0], %[b0], 32             \n\t"                           \
    "vle8.v      v11, (%[b1])                 \n\t"                           \
    "addi        %[b1], %[b1], 32             \n\t"                           \
    "vle8.v      v12, (%[b2])                 \n\t"                           \
    "addi        %[b2], %[b2], 32             \n\t"                           \
    "vle8.v      v13, (%[b3])                 \n\t"                           \
    "addi        %[b3], %[b3], 32             \n\t"

/* Edge kernel: one 4x4 output tile, acc(v16:v17) += sum over kb K-tiles of
 * a*b^T, stored row-major to c with row stride ldc (elements). */
static inline void ime_tile(const int8_t *a, const int8_t *b, long kb,
                            int32_t *c, long ldc)
{
    __asm__ volatile(
        "li          t3, 32                      \n\t"
        "vsetvli     zero, t3, e8, m1, ta, ma    \n\t"
        "vmv.v.i     v16, 0                       \n\t"
        "vmv.v.i     v17, 0                       \n\t"
        "mv          t0, %[kb]                    \n\t"
        "1:                                       \n\t"
        "vle8.v      v8, (%[a])                   \n\t"
        "vle8.v      v9, (%[b])                   \n\t"
        "addi        %[a], %[a], 32               \n\t"
        "addi        %[b], %[b], 32               \n\t"
        SMT_VMADOT(16, 8, 9)
        "addi        t0, t0, -1                   \n\t"
        "bnez        t0, 1b                       \n\t"
        "slli        t1, %[ldc], 2                \n\t"
        "vsetivli    zero, 4, e32, m1, ta, ma     \n\t"
        "vse32.v     v16, (%[c])                  \n\t"
        "add         t2, %[c], t1                 \n\t"
        "vslidedown.vi v24, v16, 4                \n\t"
        "vse32.v     v24, (t2)                    \n\t"
        "add         t2, t2, t1                   \n\t"
        "vse32.v     v17, (t2)                    \n\t"
        "add         t2, t2, t1                   \n\t"
        "vslidedown.vi v25, v17, 4                \n\t"
        "vse32.v     v25, (t2)                    \n\t"
        : [a] "+r"(a), [b] "+r"(b)
        : [kb] "r"(kb), [c] "r"(c), [ldc] "r"(ldc)
        : "t0", "t1", "t2", "t3", "v8", "v9", "v16", "v17", "v24", "v25", "memory");
}

/* Store one 8x16 output row: four 4-int32 segments to the four column pointers
 * t2..t5, then advance all four by one C row (ldcb bytes). DIR uses lanes 0-3
 * (block rows 0/2/4/6); SLIDE extracts lanes 4-7 via vslidedown (rows 1/3/5/7).
 * Keeping the store vectorized straight into C is what makes register-blocking
 * a win - a scalar scratch->C copy thrashes L2 on large N. */
#define IME_ST_ROW_DIR(r0, r1, r2, r3)                                         \
    "vse32.v " #r0 ", (t2)\n\t vse32.v " #r1 ", (t3)\n\t"                       \
    "vse32.v " #r2 ", (t4)\n\t vse32.v " #r3 ", (t5)\n\t"                       \
    "add t2,t2,t1\n\t add t3,t3,t1\n\t add t4,t4,t1\n\t add t5,t5,t1\n\t"
#define IME_ST_ROW_SLIDE(r0, r1, r2, r3)                                       \
    "vslidedown.vi v8, " #r0 ", 4\n\t  vse32.v v8,  (t2)\n\t"                   \
    "vslidedown.vi v9, " #r1 ", 4\n\t  vse32.v v9,  (t3)\n\t"                   \
    "vslidedown.vi v10, " #r2 ", 4\n\t vse32.v v10, (t4)\n\t"                   \
    "vslidedown.vi v11, " #r3 ", 4\n\t vse32.v v11, (t5)\n\t"                   \
    "add t2,t2,t1\n\t add t3,t3,t1\n\t add t4,t4,t1\n\t add t5,t5,t1\n\t"

/* Original: batch 6 loads then 8 vmadots. Kept for A/B. */
void ime_kloop_8x16_seq(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                        const int8_t *b1, const int8_t *b2, const int8_t *b3, long kb)
{
    __asm__ volatile(
        "li          t1, 32                       \n\t"
        "vsetvli     zero, t1, e8, m1, ta, ma     \n\t"
        "mv          t0, %[kb]                    \n\t"
        "1:                                       \n\t"
        "vle8.v      v8,  (%[a0])                 \n\t"
        "vle8.v      v9,  (%[a1])                 \n\t"
        "vle8.v      v10, (%[b0])                 \n\t"
        "vle8.v      v11, (%[b1])                 \n\t"
        "vle8.v      v12, (%[b2])                 \n\t"
        "vle8.v      v13, (%[b3])                 \n\t"
        "addi        %[a0], %[a0], 32             \n\t"
        "addi        %[a1], %[a1], 32             \n\t"
        "addi        %[b0], %[b0], 32             \n\t"
        "addi        %[b1], %[b1], 32             \n\t"
        "addi        %[b2], %[b2], 32             \n\t"
        "addi        %[b3], %[b3], 32             \n\t"
        SMT_VMADOT(16, 8, 10)
        SMT_VMADOT(18, 8, 11)
        SMT_VMADOT(20, 8, 12)
        SMT_VMADOT(22, 8, 13)
        SMT_VMADOT(24, 9, 10)
        SMT_VMADOT(26, 9, 11)
        SMT_VMADOT(28, 9, 12)
        SMT_VMADOT(30, 9, 13)
        "addi        t0, t0, -1                   \n\t"
        "bnez        t0, 1b                       \n\t"
        : [a0] "+r"(a0), [a1] "+r"(a1), [b0] "+r"(b0), [b1] "+r"(b1),
          [b2] "+r"(b2), [b3] "+r"(b3)
        : [kb] "r"(kb)
        : "t0", "t1", "v0", "v1", "v2", "v3", "v4", "v5", "v8", "v9", "v10", "v11", "v12",
          "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
          "v27", "v28", "v29", "v30", "v31", "memory");
}

/* Interleaved loads + vmadots on v8–v13. */
void ime_kloop_8x16_ilv(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                        const int8_t *b1, const int8_t *b2, const int8_t *b3, long kb)
{
    __asm__ volatile(
        "li          t1, 32                       \n\t"
        "vsetvli     zero, t1, e8, m1, ta, ma     \n\t"
        "mv          t0, %[kb]                    \n\t"
        "1:                                       \n\t"
        IME_KSTEP_V813
        "addi        t0, t0, -1                   \n\t"
        "bnez        t0, 1b                       \n\t"
        : [a0] "+r"(a0), [a1] "+r"(a1), [b0] "+r"(b0), [b1] "+r"(b1),
          [b2] "+r"(b2), [b3] "+r"(b3)
        : [kb] "r"(kb)
        : "t0", "t1", "v0", "v1", "v2", "v3", "v4", "v5", "v8", "v9", "v10", "v11", "v12",
          "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
          "v27", "v28", "v29", "v30", "v31", "memory");
}

/* Dual-bank software pipeline: compute v8–v13 while prefetching v0–v5 and vice versa. */
void ime_kloop_8x16_piped(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                          const int8_t *b1, const int8_t *b2, const int8_t *b3, long kb)
{
    if (kb <= 1) {
        if (kb == 1) {
            __asm__ volatile(
                "li          t1, 32                   \n\t"
                "vsetvli     zero, t1, e8, m1, ta, ma \n\t"
                IME_KLOAD_V813
                IME_KCOMPUTE_V813
                : [a0] "+r"(a0), [a1] "+r"(a1), [b0] "+r"(b0), [b1] "+r"(b1),
                  [b2] "+r"(b2), [b3] "+r"(b3)
                :
                : "t1", "v0", "v1", "v2", "v3", "v4", "v5", "v8", "v9", "v10", "v11", "v12",
                  "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                  "v26", "v27", "v28", "v29", "v30", "v31", "memory");
        }
        return;
    }
    __asm__ volatile(
        "li          t1, 32                       \n\t"
        "vsetvli     zero, t1, e8, m1, ta, ma     \n\t"
        IME_KLOAD_V813
        "mv          t0, %[kb]                    \n\t"
        "addi        t0, t0, -1                   \n\t"
        "1:                                       \n\t"
        IME_KSTEP_V813_PREF05
        "addi        t0, t0, -1                   \n\t"
        "beqz        t0, 2f                       \n\t"
        IME_KSTEP_V05_PREF813
        "addi        t0, t0, -1                   \n\t"
        "bnez        t0, 1b                       \n\t"
        "j           3f                           \n\t"
        "2:                                       \n\t"
        IME_KCOMPUTE_V05
        "j           4f                           \n\t"
        "3:                                       \n\t"
        IME_KCOMPUTE_V813
        "4:                                       \n\t"
        : [a0] "+r"(a0), [a1] "+r"(a1), [b0] "+r"(b0), [b1] "+r"(b1),
          [b2] "+r"(b2), [b3] "+r"(b3)
        : [kb] "r"(kb)
        : "t0", "t1", "v0", "v1", "v2", "v3", "v4", "v5", "v8", "v9", "v10", "v11", "v12",
          "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
          "v27", "v28", "v29", "v30", "v31", "memory");
}

/* Production inner loop (pipelined). */
void ime_kloop_8x16(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                    const int8_t *b1, const int8_t *b2, const int8_t *b3, long kb)
{
    ime_kloop_8x16_piped(a0, a1, b0, b1, b2, b3, kb);
}

/* 4×32 K-loop: one A row-group × eight B column-groups (SpacemiT A60 tile is
 * 4×8×4; this is 2× wider in N than 8×16 with the same 8 acc pairs). */
void ime_kloop_4x32_ilv(const int8_t *a0, const int8_t *b0, const int8_t *b1,
                        const int8_t *b2, const int8_t *b3, const int8_t *b4,
                        const int8_t *b5, const int8_t *b6, const int8_t *b7, long kb)
{
    __asm__ volatile(
        "li          t1, 32                       \n\t"
        "vsetvli     zero, t1, e8, m1, ta, ma     \n\t"
        "mv          t0, %[kb]                    \n\t"
        "1:                                       \n\t"
        IME_KSTEP_4x32
        "addi        t0, t0, -1                   \n\t"
        "bnez        t0, 1b                       \n\t"
        : [a0] "+r"(a0), [b0] "+r"(b0), [b1] "+r"(b1), [b2] "+r"(b2), [b3] "+r"(b3),
          [b4] "+r"(b4), [b5] "+r"(b5), [b6] "+r"(b6), [b7] "+r"(b7)
        : [kb] "r"(kb)
        : "t0", "t1", "v8", "v9", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory");
}

static void ime_store_8x16(int32_t *c, long ldc)
{
    __asm__ volatile(
        "slli        t1, %[ldc], 2                \n\t"
        "vsetivli    zero, 4, e32, m1, ta, ma     \n\t"
        "mv          t2, %[c]                      \n\t"
        "addi        t3, t2, 16                    \n\t"
        "addi        t4, t2, 32                    \n\t"
        "addi        t5, t2, 48                    \n\t"
        IME_ST_ROW_DIR(v16, v18, v20, v22)
        IME_ST_ROW_SLIDE(v16, v18, v20, v22)
        IME_ST_ROW_DIR(v17, v19, v21, v23)
        IME_ST_ROW_SLIDE(v17, v19, v21, v23)
        IME_ST_ROW_DIR(v24, v26, v28, v30)
        IME_ST_ROW_SLIDE(v24, v26, v28, v30)
        IME_ST_ROW_DIR(v25, v27, v29, v31)
        IME_ST_ROW_SLIDE(v25, v27, v29, v31)
        :
        : [c] "r"(c), [ldc] "r"(ldc)
        : "t1", "t2", "t3", "t4", "t5", "v8", "v9", "v10", "v11", "memory");
}

/* 8×32 on X60: two fused 8×16 panels (8×32 int8 is A100-class; 16 acc pairs
 * cannot fit in 32 vector regs on X60 per IME overlap rules). */
void ime_block_8x32(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                    const int8_t *b1, const int8_t *b2, const int8_t *b3,
                    const int8_t *b4, const int8_t *b5, const int8_t *b6,
                    const int8_t *b7, long kb, int32_t *c, long ldc)
{
    __asm__ volatile(IME_ZERO_ACC_8x16 : : : "t1", "t4", "v16", "v17", "v18", "v19",
                     "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                     "v29", "v30", "v31");
    ime_kloop_8x16_piped(a0, a1, b0, b1, b2, b3, kb);
    ime_store_8x16(c, ldc);
    __asm__ volatile(IME_ZERO_ACC_8x16 : : : "t1", "t4", "v16", "v17", "v18", "v19",
                     "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                     "v29", "v30", "v31");
    ime_kloop_8x16_piped(a0, a1, b4, b5, b6, b7, kb);
    ime_store_8x16(c + 16, ldc);
}

/* 4×32 block: four rows × 32 cols. */
void ime_block_4x32(const int8_t *a0, const int8_t *b0, const int8_t *b1,
                    const int8_t *b2, const int8_t *b3, const int8_t *b4,
                    const int8_t *b5, const int8_t *b6, const int8_t *b7, long kb,
                    int32_t *c, long ldc)
{
    __asm__ volatile(
        "vsetvli     t4, zero, e8, m8, ta, ma    \n\t"
        "vmv.v.i     v16, 0                       \n\t"
        "vmv.v.i     v24, 0                       \n\t"
        :
        :
        : "t4", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
          "v26", "v27", "v28", "v29", "v30", "v31");
    ime_kloop_4x32_ilv(a0, b0, b1, b2, b3, b4, b5, b6, b7, kb);
    /* 4 rows × 32 cols: store 4×4 tiles from v16..v31 */
    __asm__ volatile(
        "slli        t1, %[ldc], 2                \n\t"
        "vsetivli    zero, 4, e32, m1, ta, ma     \n\t"
        "mv          t2, %[c]                      \n\t"
        "vse32.v     v16, (t2)                     \n\t"
        "addi        t2, t2, 16                    \n\t"
        "vse32.v     v18, (t2)                     \n\t"
        "addi        t2, t2, 16                    \n\t"
        "vse32.v     v20, (t2)                     \n\t"
        "addi        t2, t2, 16                    \n\t"
        "vse32.v     v22, (t2)                     \n\t"
        "addi        t2, t2, 16                    \n\t"
        "vse32.v     v24, (t2)                     \n\t"
        "addi        t2, t2, 16                    \n\t"
        "vse32.v     v26, (t2)                     \n\t"
        "addi        t2, t2, 16                    \n\t"
        "vse32.v     v28, (t2)                     \n\t"
        "addi        t2, t2, 16                    \n\t"
        "vse32.v     v30, (t2)                     \n\t"
        "add         t2, %[c], t1                  \n\t"
        "vslidedown.vi v8, v16, 4                \n\t"
        "vse32.v     v8, (t2)                      \n\t"
        "addi        t3, t2, 16                    \n\t"
        "vslidedown.vi v9, v18, 4                \n\t"
        "vse32.v     v9, (t3)                      \n\t"
        "addi        t3, t3, 16                    \n\t"
        "vslidedown.vi v10, v20, 4               \n\t"
        "vse32.v     v10, (t3)                     \n\t"
        "addi        t3, t3, 16                    \n\t"
        "vslidedown.vi v11, v22, 4               \n\t"
        "vse32.v     v11, (t3)                     \n\t"
        "addi        t3, t3, 16                    \n\t"
        "vslidedown.vi v8, v24, 4                \n\t"
        "vse32.v     v8, (t3)                      \n\t"
        "addi        t3, t3, 16                    \n\t"
        "vslidedown.vi v9, v26, 4                \n\t"
        "vse32.v     v9, (t3)                      \n\t"
        "addi        t3, t3, 16                    \n\t"
        "vslidedown.vi v10, v28, 4               \n\t"
        "vse32.v     v10, (t3)                     \n\t"
        "addi        t3, t3, 16                    \n\t"
        "vslidedown.vi v11, v30, 4               \n\t"
        "vse32.v     v11, (t3)                     \n\t"
        :
        : [c] "r"(c), [ldc] "r"(ldc)
        : "t1", "t2", "t3", "v8", "v9", "v10", "v11", "memory");
}

/* Register-blocked 8×16 kernel. */
void ime_block_8x16(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                    const int8_t *b1, const int8_t *b2, const int8_t *b3, long kb,
                    int32_t *c, long ldc)
{
    __asm__ volatile(IME_ZERO_ACC_8x16 : : : "t1", "t4", "v16", "v17", "v18", "v19",
                     "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                     "v29", "v30", "v31");
    ime_kloop_8x16(a0, a1, b0, b1, b2, b3, kb);
    ime_store_8x16(c, ldc);
}

void gemm_ime(const int8_t *A, const int8_t *B, int32_t *C,
              int M, int N, int K, int8_t *Ap, int8_t *Bp, int ldc)
{
    if (ldc < N)
        ldc = N;
    const int KB = K / TK, MT = M / TM, NT = N / TN;
    /* L2 cache-blocking: process N in panels of `nc` tiles so the reused B-panel
 * (nc * K bytes) stays in the ~512 KB cluster L2 across the whole M sweep.
 * Target ~128 KB leaves room for the A column and the C write stream; this
 * is what stops the fast (latency-hidden) kernel from going DRAM-bound.
 * Packing is amortized to match the panel structure: pack_a_panel once on the
 * first N-panel (n0==0), pack_b_panel once per N-panel before the M sweep. */
    int nc = 32768 / (K > 0 ? K : 1);
    nc &= ~3;
    if (nc < 4) nc = 4;
    for (int n0 = 0; n0 < NT; n0 += nc) {
        int n1 = (n0 + nc < NT) ? n0 + nc : NT;
        pack_b_panel(B, Bp, n0, n1 - n0, K);
        for (int mb = 0; mb < MT; mb += 2) {
            if (n0 == 0) {
                int mlim = (mb + 2 <= MT) ? 2 : 1;
                pack_a_panel(A, Ap, mb, mlim, K);
            }
            for (int nb = n0; nb < n1; nb += 4) {
                if (mb + 2 <= MT && nb + 4 <= n1) {
                    ime_block_8x16(Ap + (size_t)mb * KB * TILE_BYTES,
                                   Ap + (size_t)(mb + 1) * KB * TILE_BYTES,
                                   Bp + (size_t)nb * KB * TILE_BYTES,
                                   Bp + (size_t)(nb + 1) * KB * TILE_BYTES,
                                   Bp + (size_t)(nb + 2) * KB * TILE_BYTES,
                                   Bp + (size_t)(nb + 3) * KB * TILE_BYTES, KB,
                                   C + (size_t)(mb * TM) * (size_t)ldc + nb * TN, ldc);
                } else {
                    int mlim = (mb + 2 <= MT) ? 2 : 1;
                    int nlim = (n1 - nb) < 4 ? (n1 - nb) : 4;
                    for (int dm = 0; dm < mlim; dm++)
                        for (int dn = 0; dn < nlim; dn++)
                            ime_tile(Ap + (size_t)(mb + dm) * KB * TILE_BYTES,
                                     Bp + (size_t)(nb + dn) * KB * TILE_BYTES, KB,
                                     C + (size_t)((mb + dm) * TM) * (size_t)ldc +
                                         (nb + dn) * TN,
                                     ldc);
                }
            }
        }
    }
}

#endif /* __riscv && !GEMM_NO_IME */
