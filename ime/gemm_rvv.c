/*
 * gemm_rvv.c - RVV int8 baseline for s8s8s32 GEMM (the honest speed reference
 * the IME kernel must beat). Straightforward per-output widening dot product:
 * vwmul (int8xint8->int16) then a widening reduction into int32. Not maximally
 * cache-tuned (it reloads B rows), but correct and representative of "plain RVV
 * int8" - which is exactly the baseline a framework backend would start from.
 * SPDX-License-Identifier: MIT
 */
#include "gemm_s8s8s32.h"

#if defined(__riscv)
#include <riscv_vector.h>

void gemm_rvv(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        const int8_t *arow = A + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const int8_t *brow = B + (size_t)j * K;
            vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, 1);
            for (size_t k = 0, rem = K; rem > 0;) {
                size_t vl = __riscv_vsetvl_e8m1(rem);
                vint8m1_t va = __riscv_vle8_v_i8m1(arow + k, vl);
                vint8m1_t vb = __riscv_vle8_v_i8m1(brow + k, vl);
                vint16m2_t vp = __riscv_vwmul_vv_i16m2(va, vb, vl);
                acc = __riscv_vwredsum_vs_i16m2_i32m1(vp, acc, vl);
                k += vl;
                rem -= vl;
            }
            C[(size_t)i * N + j] = __riscv_vmv_x_s_i32m1_i32(acc);
        }
    }
}

#endif /* __riscv */
