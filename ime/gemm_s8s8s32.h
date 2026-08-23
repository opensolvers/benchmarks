/*
 * gemm_s8s8s32.h - reusable int8xint8->int32 GEMM core for the SpaceMiT X60 IME.
 *
 * Computes C = A * B^T with C[i][j] = sum_k A[i][k] * B[j][k], where
 *   A is M x K  (row-major int8),
 *   B is N x K  (row-major int8; already in the "B^T" orientation vmadot wants),
 *   C is M x N  (row-major int32).
 *
 * This B-is-N-by-K layout is exactly the mmt4d / ggml-repack operand layout, so
 * the packing here is the same shape a framework backend (MLAS, XNNPACK, ruy...)
 * would feed to the IME kernel.
 *
 * The X60 IME (XsmtVdot v1.0) instruction `smt.vmadot vd, vs1, vs2` does one
 *   4x4 int32 += (4x8 int8) * (4x8 int8)^T
 * tile: M0=4, N0=4, K0=8 at VLEN=256 (vl=32,e8). vd is a vd:vd+1 pair, even.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef GEMM_S8S8S32_H
#define GEMM_S8S8S32_H

#include <stdint.h>
#include <stddef.h>

/* IME instruction tile (M0 x N0 x K0). Dims below must be multiples of these. */
#define TM 4
#define TN 4
#define TK 8
#define TILE_BYTES 32 /* 4 rows x 8 cols int8 = one packed vmadot operand tile */

/* Extra int32 columns on C row stride — breaks 32 KB L2 set aliasing on the C
 * write stream (see README "Cache-set aliasing"). Must stay 0 mod 4. */
#define GEMM_C_PAD 16

/* Byte slack before packed operand buffers — decouples Ap/Bp base from A/B/C. */
#define GEMM_BUF_PAD 2048

/* Portable reference: naive triple loop. Ground truth. */
void gemm_ref(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K);

/* Shared packing (the reusable core): rearrange row-major A/B into contiguous
 * 4x8 int8 tiles, tile-index order [block][kblock][4][8]. Ap needs M*K bytes,
 * Bp needs N*K bytes. M%4==N%4==0, K%8==0. */
void pack_a(const int8_t *A, int8_t *Ap, int M, int K);
void pack_b(const int8_t *B, int8_t *Bp, int N, int K);

/* Pack a row-tile panel: mb0 = starting M-block index, mb_cnt blocks (each TM rows). */
void pack_a_panel(const int8_t *A, int8_t *Ap, int mb0, int mb_cnt, int K);
/* Pack a column-tile panel: nb0 = starting N-block index, nb_cnt blocks (each TN cols). */
void pack_b_panel(const int8_t *B, int8_t *Bp, int nb0, int nb_cnt, int K);

/* Scalar model of one smt.vmadot: acc[i*4+j] += sum_c a[i*8+c]*b[j*8+c]. */
void tile_mul_ref(int32_t acc[TM * TN], const int8_t *a, const int8_t *b);

/* Portable packed path: pack + per-tile scalar multiply + unpack. Exercises the
 * exact packing/indexing the IME path uses, so it is verifiable off-target. */
void gemm_packed_ref(const int8_t *A, const int8_t *B, int32_t *C,
                     int M, int N, int K, int8_t *Ap, int8_t *Bp);

#if defined(__riscv)
/* RVV int8 baseline (vwmul + widening reduction). The honest speed reference. */
void gemm_rvv(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K);
#endif

/* IME path: pack + smt.vmadot microkernel + unpack. Requires X60 hardware and a
 * binutils that knows xsmtvdot. Define GEMM_NO_IME to build RVV-only (e.g. on a
 * board whose native binutils < 2.46). */
#if defined(__riscv) && !defined(GEMM_NO_IME)
void gemm_ime(const int8_t *A, const int8_t *B, int32_t *C,
              int M, int N, int K, int8_t *Ap, int8_t *Bp, int ldc);

/* Inner K-loop variants (8× vmadot per K-tile). Production uses piped. */
void ime_kloop_8x16(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                    const int8_t *b1, const int8_t *b2, const int8_t *b3, long kb);
void ime_kloop_8x16_seq(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                        const int8_t *b1, const int8_t *b2, const int8_t *b3, long kb);
void ime_kloop_8x16_ilv(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                        const int8_t *b1, const int8_t *b2, const int8_t *b3, long kb);
void ime_kloop_8x16_piped(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                          const int8_t *b1, const int8_t *b2, const int8_t *b3, long kb);

void ime_kloop_4x32_ilv(const int8_t *a0, const int8_t *b0, const int8_t *b1,
                        const int8_t *b2, const int8_t *b3, const int8_t *b4,
                        const int8_t *b5, const int8_t *b6, const int8_t *b7, long kb);

void ime_block_8x32(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                    const int8_t *b1, const int8_t *b2, const int8_t *b3,
                    const int8_t *b4, const int8_t *b5, const int8_t *b6,
                    const int8_t *b7, long kb, int32_t *c, long ldc);

void ime_block_4x32(const int8_t *a0, const int8_t *b0, const int8_t *b1,
                    const int8_t *b2, const int8_t *b3, const int8_t *b4,
                    const int8_t *b5, const int8_t *b6, const int8_t *b7, long kb,
                    int32_t *c, long ldc);

/* Full 8×16 micro-tile: zero acc, kloop, store to C (row stride ldc). */
void ime_block_8x16(const int8_t *a0, const int8_t *a1, const int8_t *b0,
                    const int8_t *b1, const int8_t *b2, const int8_t *b3, long kb,
                    int32_t *c, long ldc);
#endif

#endif /* GEMM_S8S8S32_H */
