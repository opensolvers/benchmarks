/*
 * Panel / memory tuning hooks for gemm_ime (step-2 experiments).
 * SPDX-License-Identifier: MIT
 */
#ifndef GEMM_PANEL_H
#define GEMM_PANEL_H

#include <stddef.h>

#if defined(__riscv) && !defined(GEMM_NO_IME)

/* N-panel width in TN-column tiles (0 = auto from K). Must be >= 4, multiple of 4. */
void gemm_ime_set_nc(int nc_tiles);
int gemm_ime_get_nc(int K);

/* 0 = plain panel loop; 1 = cache-touch next operand tiles before each block. */
void gemm_ime_set_megakernel(int on);

/* 1 = pack_a_panel(2 M-blocks) inside compute loop instead of monolithic pack_a. */
void gemm_ime_set_fused_pack_a(int on);

/* Compute with optional pack flags (Ap/Bp layout unchanged). */
void gemm_ime_compute_ex(const int8_t *A, const int8_t *B, int8_t *Ap, int8_t *Bp,
                         int32_t *C, int M, int N, int K, int ldc, int do_pack_a,
                         int do_pack_b);

/* B-panel TCM staging (see tcm.c). */
size_t gemm_ime_b_panel_bytes(int K);
size_t gemm_ime_packed_b_bytes(int N, int K);
void gemm_ime_compute_tcm_b(const int8_t *A, const int8_t *B, int8_t *Ap, int8_t *Bp_tcm,
                            int32_t *C, int M, int N, int K, int ldc, int do_pack_a);
/* B already packed in DRAM; memcpy each panel into |Bp_tcm| then compute. */
void gemm_ime_compute_tcm_staged(const int8_t *A, const int8_t *Bp_packed, int8_t *Ap,
                                 int8_t *Bp_tcm, int32_t *C, int M, int N, int K, int ldc,
                                 int do_pack_a);
void gemm_ime_tcm_b(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K,
                    int8_t *Ap, int8_t *Bp_tcm, int ldc);

/* Best path: B pre-packed in TCM, M-outer fused pack_a + compute (~+25% vs DRAM offline-B). */
void gemm_ime_compute_tcm_offline_b(const int8_t *A, int8_t *Ap, const int8_t *Bp_tcm,
                                    int32_t *C, int M, int N, int K, int ldc);

#endif

#endif /* GEMM_PANEL_H */
