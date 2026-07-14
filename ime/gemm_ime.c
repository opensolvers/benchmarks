/*
 * gemm_ime.c - SpaceMiT X60 IME (XsmtVdot v1.0) int8 GEMM microkernel.
 *
 * The reusable core: one smt.vmadot does a 4x4 int32 += (4x8 int8)*(4x8 int8)^T
 * tile (vl=32,e8 at VLEN=256). We pack A/B into contiguous 4x8 tiles (shared
 * with gemm_packed_ref, so the layout is verified off-target), then for each
 * 4x4 output block accumulate across K in the vd:vd+1 pair v16:v17 and store.
 *
 * Toolchain: the `.option arch, +xsmtvdot` directive enables the mnemonic
 * locally, so no -march=...xsmtvdot is needed - only a binutils that knows it
 * (>=2.46, or a SpaceMiT/Bianbu cross toolchain that ships xsmtvdot). The
 * GEMM_IME_INSN build below sidesteps this via a raw .insn encoding.
 * Runs ONLY on real X60 silicon: qemu-user does not emulate vmadot.
 * SPDX-License-Identifier: MIT
 */
#include "gemm_s8s8s32.h"

#if defined(__riscv)

/* Emit `smt.vmadot v16, v8, v9` (i8). With GEMM_IME_INSN we spell it as its raw
 * instruction word so a binutils without xsmtvdot (< 2.46) assembles it: base
 * 0xe200302b | (vd/2)<<8 | vs1<<15 | vs2<<20 = 0xe294382b. Cross-checked against
 * LLVM's published encoding for smt.vmadot v16,v0,v8 (0xe280382b). */
#if defined(GEMM_IME_INSN)
#define IME_ARCH ""
#define IME_VMADOT ".insn 0xe294382b                       \n\t"
#else
#define IME_ARCH ".option arch, +xsmtvdot                 \n\t"
#define IME_VMADOT "smt.vmadot  v16, v8, v9                 \n\t"
#endif

/* One 4x4 output tile: acc(v16:v17) = sum over `kb` K-tiles of a*b^T, then
 * store row-major to c with row stride ldc (elements). a,b are contiguous
 * 32-byte tiles (stride TILE_BYTES per K-step). */
static inline void ime_tile(const int8_t *a, const int8_t *b, long kb,
                            int32_t *c, long ldc)
{
    __asm__ volatile(
        IME_ARCH
        "li          t3, 32                      \n\t"
        "vsetvli     zero, t3, e8, m1, ta, ma    \n\t"
        "vmv.v.i     v16, 0                      \n\t" /* zero acc rows 0,1 */
        "vmv.v.i     v17, 0                      \n\t" /* zero acc rows 2,3 */
        "mv          t0, %[kb]                   \n\t"
        "1:                                      \n\t"
        "vle8.v      v8, (%[a])                  \n\t" /* A tile 4x8 */
        "vle8.v      v9, (%[b])                  \n\t" /* B tile 4x8 */
        "addi        %[a], %[a], 32              \n\t"
        "addi        %[b], %[b], 32              \n\t"
        IME_VMADOT /* v16:v17 += A * B^T */
        "addi        t0, t0, -1                  \n\t"
        "bnez        t0, 1b                      \n\t"
        /* store 4x4 int32: v16 = [row0(0..3) row1(4..7)], v17 = [row2 row3] */
        "slli        t1, %[ldc], 2               \n\t" /* row stride in bytes */
        "vsetivli    zero, 4, e32, m1, ta, ma    \n\t"
        "vse32.v     v16, (%[c])                 \n\t" /* row 0 */
        "add         t2, %[c], t1                \n\t"
        "vslidedown.vi v24, v16, 4               \n\t"
        "vse32.v     v24, (t2)                   \n\t" /* row 1 */
        "add         t2, t2, t1                  \n\t"
        "vse32.v     v17, (t2)                   \n\t" /* row 2 */
        "add         t2, t2, t1                  \n\t"
        "vslidedown.vi v25, v17, 4               \n\t"
        "vse32.v     v25, (t2)                   \n\t" /* row 3 */
        : [a] "+r"(a), [b] "+r"(b)
        : [kb] "r"(kb), [c] "r"(c), [ldc] "r"(ldc)
        : "t0", "t1", "t2", "t3", "v8", "v9", "v16", "v17", "v24", "v25", "memory");
}

void gemm_ime(const int8_t *A, const int8_t *B, int32_t *C,
              int M, int N, int K, int8_t *Ap, int8_t *Bp)
{
    const int KB = K / TK;
    pack_a(A, Ap, M, K);
    pack_b(B, Bp, N, K);
    for (int mb = 0; mb < M / TM; mb++) {
        const int8_t *ac = Ap + (size_t)mb * KB * TILE_BYTES;
        for (int nb = 0; nb < N / TN; nb++) {
            const int8_t *bc = Bp + (size_t)nb * KB * TILE_BYTES;
            ime_tile(ac, bc, KB, C + (size_t)(mb * TM) * N + nb * TN, N);
        }
    }
}

#endif /* __riscv */
