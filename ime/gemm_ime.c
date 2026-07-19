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

/* Register-blocked 8x16 kernel: 8 accumulator pairs (v16..v31) = 2 M-tiles x 4
 * N-tiles. Each K-step loads A0,A1 (v8,v9) + B0..B3 (v10..v13) and issues 8
 * vmadots. The 8 result tiles are written straight to C (row stride ldc). */
static inline void ime_block_8x16(const int8_t *a0, const int8_t *a1,
                                  const int8_t *b0, const int8_t *b1,
                                  const int8_t *b2, const int8_t *b3,
                                  long kb, int32_t *c, long ldc)
{
    __asm__ volatile(
        "vsetvli     t4, zero, e8, m8, ta, ma    \n\t"
        "vmv.v.i     v16, 0                       \n\t"
        "vmv.v.i     v24, 0                       \n\t"
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
        "slli        t1, %[ldc], 2                \n\t"
        "vsetivli    zero, 4, e32, m1, ta, ma     \n\t"
        "mv          t2, %[c]                      \n\t"
        "addi        t3, t2, 16                    \n\t"
        "addi        t4, t2, 32                    \n\t"
        "addi        t5, t2, 48                    \n\t"
        IME_ST_ROW_DIR(v16, v18, v20, v22)   /* row 0 */
        IME_ST_ROW_SLIDE(v16, v18, v20, v22) /* row 1 */
        IME_ST_ROW_DIR(v17, v19, v21, v23)   /* row 2 */
        IME_ST_ROW_SLIDE(v17, v19, v21, v23) /* row 3 */
        IME_ST_ROW_DIR(v24, v26, v28, v30)   /* row 4 */
        IME_ST_ROW_SLIDE(v24, v26, v28, v30) /* row 5 */
        IME_ST_ROW_DIR(v25, v27, v29, v31)   /* row 6 */
        IME_ST_ROW_SLIDE(v25, v27, v29, v31) /* row 7 */
        : [a0] "+r"(a0), [a1] "+r"(a1), [b0] "+r"(b0), [b1] "+r"(b1),
          [b2] "+r"(b2), [b3] "+r"(b3)
        : [kb] "r"(kb), [c] "r"(c), [ldc] "r"(ldc)
        : "t0", "t1", "t2", "t3", "t4", "t5", "v8", "v9", "v10", "v11", "v12", "v13",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory");
}

void gemm_ime(const int8_t *A, const int8_t *B, int32_t *C,
              int M, int N, int K, int8_t *Ap, int8_t *Bp)
{
    const int KB = K / TK, MT = M / TM, NT = N / TN;
    /* L2 cache-blocking: process N in panels of `nc` tiles so the reused B-panel
     * (nc * K bytes) stays in the ~512 KB cluster L2 across the whole M sweep.
     * Target ~128 KB leaves room for the A column and the C write stream; this
     * is what stops the fast (latency-hidden) kernel from going DRAM-bound. */
    int nc = 32768 / (K > 0 ? K : 1);
    nc &= ~3;
    if (nc < 4) nc = 4;
    pack_a(A, Ap, M, K);
    pack_b(B, Bp, N, K);
    for (int n0 = 0; n0 < NT; n0 += nc) {
        int n1 = (n0 + nc < NT) ? n0 + nc : NT;
        for (int mb = 0; mb < MT; mb += 2) {
            for (int nb = n0; nb < n1; nb += 4) {
                if (mb + 2 <= MT && nb + 4 <= n1) {
                    ime_block_8x16(Ap + (size_t)mb * KB * TILE_BYTES,
                                   Ap + (size_t)(mb + 1) * KB * TILE_BYTES,
                                   Bp + (size_t)nb * KB * TILE_BYTES,
                                   Bp + (size_t)(nb + 1) * KB * TILE_BYTES,
                                   Bp + (size_t)(nb + 2) * KB * TILE_BYTES,
                                   Bp + (size_t)(nb + 3) * KB * TILE_BYTES, KB,
                                   C + (size_t)(mb * TM) * N + nb * TN, N);
                } else {
                    int mlim = (mb + 2 <= MT) ? 2 : 1;
                    int nlim = (n1 - nb) < 4 ? (n1 - nb) : 4;
                    for (int dm = 0; dm < mlim; dm++)
                        for (int dn = 0; dn < nlim; dn++)
                            ime_tile(Ap + (size_t)(mb + dm) * KB * TILE_BYTES,
                                     Bp + (size_t)(nb + dn) * KB * TILE_BYTES, KB,
                                     C + (size_t)((mb + dm) * TM) * N + (nb + dn) * TN, N);
                }
            }
        }
    }
}

#endif /* __riscv && !GEMM_NO_IME */
