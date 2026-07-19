/*
 * gemm_ref.c - portable reference + shared packing for the s8s8s32 IME core.
 * No RISC-V dependencies: builds and runs anywhere, so it validates the packing
 * and tile-index math (the error-prone part shared with the IME kernel) on any
 * host. Only the actual smt.vmadot / RVV inner loops are target-only.
 * SPDX-License-Identifier: MIT
 */
#include "gemm_s8s8s32.h"

void gemm_ref(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            const int8_t *a = A + (size_t)i * K;
            const int8_t *b = B + (size_t)j * K;
            for (int k = 0; k < K; k++)
                acc += (int32_t)a[k] * (int32_t)b[k];
            C[(size_t)i * N + j] = acc;
        }
    }
}

/* A[M][K] -> Ap laid out as [mb][kb][4][8], contiguous 32-byte tiles.
 * Tile (mb,kb) at Ap + (mb*(K/TK) + kb)*TILE_BYTES; element (r,c) at +(r*TK+c). */
void pack_a(const int8_t *A, int8_t *Ap, int M, int K)
{
    const int KB = K / TK;
    for (int mb = 0; mb < M / TM; mb++)
        for (int kb = 0; kb < KB; kb++) {
            int8_t *t = Ap + ((size_t)mb * KB + kb) * TILE_BYTES;
            for (int r = 0; r < TM; r++)
                for (int c = 0; c < TK; c++)
                    t[r * TK + c] = A[(size_t)(mb * TM + r) * K + (kb * TK + c)];
        }
}

/* B[N][K] -> Bp laid out as [nb][kb][4][8]; same tile shape as A. */
void pack_b(const int8_t *B, int8_t *Bp, int N, int K)
{
    const int KB = K / TK;
    for (int nb = 0; nb < N / TN; nb++)
        for (int kb = 0; kb < KB; kb++) {
            int8_t *t = Bp + ((size_t)nb * KB + kb) * TILE_BYTES;
            for (int r = 0; r < TN; r++)
                for (int c = 0; c < TK; c++)
                    t[r * TK + c] = B[(size_t)(nb * TN + r) * K + (kb * TK + c)];
        }
}

/* Scalar model of smt.vmadot: acc(4x4) += a(4x8) * b(4x8)^T. */
void tile_mul_ref(int32_t acc[TM * TN], const int8_t *a, const int8_t *b)
{
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) {
            int32_t s = 0;
            for (int c = 0; c < TK; c++)
                s += (int32_t)a[i * TK + c] * (int32_t)b[j * TK + c];
            acc[i * TN + j] += s;
        }
}

void gemm_packed_ref(const int8_t *A, const int8_t *B, int32_t *C,
                     int M, int N, int K, int8_t *Ap, int8_t *Bp)
{
    const int KB = K / TK;
    pack_a(A, Ap, M, K);
    pack_b(B, Bp, N, K);
    for (int mb = 0; mb < M / TM; mb++) {
        for (int nb = 0; nb < N / TN; nb++) {
            int32_t acc[TM * TN] = {0};
            const int8_t *ac = Ap + (size_t)mb * KB * TILE_BYTES;
            const int8_t *bc = Bp + (size_t)nb * KB * TILE_BYTES;
            for (int kb = 0; kb < KB; kb++)
                tile_mul_ref(acc, ac + (size_t)kb * TILE_BYTES,
                             bc + (size_t)kb * TILE_BYTES);
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    C[(size_t)(mb * TM + i) * N + (nb * TN + j)] = acc[i * TN + j];
        }
    }
}
