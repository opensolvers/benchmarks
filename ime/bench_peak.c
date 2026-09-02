/*
 * bench_peak.c - cpufp-style isolated smt.vmadot throughput on X60.
 *
 *   ./ime-bench-peak [K] [reps] [variant]
 *     K        — inner dim (default 256; multiple of 8)
 *     reps     — timed iterations (default 200000)
 *     variant  — 0=seq 1=ilv 2=piped(prod) 3=block+store (default 2)
 *
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gemm_s8s8s32.h"

#if defined(__riscv) && !defined(GEMM_NO_IME)

#define PEAK_N 16

static double secs(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void fill(int8_t *p, size_t n, uint8_t v)
{
    memset(p, v, n);
}

static void zero_acc(void)
{
    __asm__ volatile(
        "vsetvli     t0, zero, e8, m8, ta, ma    \n\t"
        "vmv.v.i     v16, 0                       \n\t"
        "vmv.v.i     v24, 0                       \n\t"
        :
        :
        : "t0", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

typedef void (*kfn)(const int8_t *, const int8_t *, const int8_t *, const int8_t *,
                    const int8_t *, const int8_t *, long);

typedef void (*kfn4)(const int8_t *, const int8_t *, const int8_t *, const int8_t *,
                     const int8_t *, const int8_t *, const int8_t *, const int8_t *,
                     const int8_t *, long);

static void run_kloop(kfn fn, const int8_t *a0, const int8_t *a1, const int8_t *b0,
                      const int8_t *b1, const int8_t *b2, const int8_t *b3, long kb)
{
    zero_acc();
    fn(a0, a1, b0, b1, b2, b3, kb);
}

static void bench_variant(const char *tag, kfn kloop, int K, long reps, const int8_t *a0,
                          const int8_t *a1, const int8_t *b0, const int8_t *b1,
                          const int8_t *b2, const int8_t *b3, long KB, int32_t *C,
                          size_t apanel, size_t bpanel, int block_store)
{
    const double vmadot_per = 8.0 * (double)KB;

    for (int w = 0; w < 32; w++)
        if (block_store)
            ime_block_8x16(a0, a1, b0, b1, b2, b3, KB, C, PEAK_N + GEMM_C_PAD);
        else
            run_kloop(kloop, a0, a1, b0, b1, b2, b3, KB);

    double t0 = secs();
    for (long r = 0; r < reps; r++) {
        if (block_store)
            ime_block_8x16(a0, a1, b0, b1, b2, b3, KB, C, PEAK_N + GEMM_C_PAD);
        else
            run_kloop(kloop, a0, a1, b0, b1, b2, b3, KB);
    }
    double dt = secs() - t0;

    const double gops_insn = vmadot_per * (double)reps / dt * 256.0 / 1e9;
    const double cyc = 1.6e9 / (vmadot_per * (double)reps / dt);

    printf("%s K=%d reps=%ld  insn=%.2f GOP/s  %.3f c/vmadot  (Ap=%zu Bp=%zu)\n", tag, K,
           reps, gops_insn, cyc, apanel, bpanel);
}

typedef void (*blkfn)(const int8_t *, const int8_t *, const int8_t *, const int8_t *,
                      const int8_t *, const int8_t *, long, int32_t *, long);

static void bench_block(const char *tag, blkfn block, int K, long reps, const int8_t *a0,
                        const int8_t *a1, const int8_t *b0, const int8_t *b1,
                        const int8_t *b2, const int8_t *b3, long KB, int32_t *C,
                        size_t apanel, size_t bpanel)
{
    const double vmadot_per = 8.0 * (double)KB;
    const long ldc = PEAK_N + GEMM_C_PAD;

    for (int w = 0; w < 32; w++)
        block(a0, a1, b0, b1, b2, b3, KB, C, ldc);

    double t0 = secs();
    for (long r = 0; r < reps; r++)
        block(a0, a1, b0, b1, b2, b3, KB, C, ldc);
    double dt = secs() - t0;

    const double gops_insn = vmadot_per * (double)reps / dt * 256.0 / 1e9;
    const double cyc = 1.6e9 / (vmadot_per * (double)reps / dt);

    printf("%s K=%d reps=%ld  insn=%.2f GOP/s  %.3f c/vmadot  (Ap=%zu Bp=%zu)\n", tag, K,
           reps, gops_insn, cyc, apanel, bpanel);
}

int main(int argc, char **argv)
{
    const int K = argc > 1 ? atoi(argv[1]) : 256;
    long reps = argc > 2 ? atol(argv[2]) : 200000L;
    const int variant = argc > 3 ? atoi(argv[3]) : 99;
    if (K % TK) {
        printf("need K%%%d==0\n", TK);
        return 1;
    }
    if (reps < 1)
        reps = 1;

    const long KB = K / TK;
    const size_t apanel = (size_t)2 * KB * TILE_BYTES;
    const size_t bpanel = (size_t)8 * KB * TILE_BYTES;

    int8_t *A = aligned_alloc(64, (size_t)8 * K);
    int8_t *B = aligned_alloc(64, (size_t)32 * K);
    int8_t *Ap = aligned_alloc(64, apanel + GEMM_BUF_PAD);
    int8_t *Bp = aligned_alloc(64, bpanel + GEMM_BUF_PAD);
    int32_t *C = aligned_alloc(64, (size_t)8 * (PEAK_N + GEMM_C_PAD) * sizeof(int32_t));
    if (!A || !B || !Ap || !Bp || !C) {
        printf("alloc fail\n");
        return 1;
    }
    Ap += GEMM_BUF_PAD;
    Bp += GEMM_BUF_PAD;

    fill(A, (size_t)8 * K, 1);
    fill(B, (size_t)32 * K, 2);
    pack_a(A, Ap, 8, K);
    pack_b(B, Bp, 32, K);

    const int8_t *a0 = Ap;
    const int8_t *a1 = Ap + (size_t)KB * TILE_BYTES;
    const int8_t *b0 = Bp;
    const int8_t *b1 = Bp + (size_t)KB * TILE_BYTES;
    const int8_t *b2 = Bp + (size_t)2 * KB * TILE_BYTES;
    const int8_t *b3 = Bp + (size_t)3 * KB * TILE_BYTES;

    if (variant == 99) {
        printf("=== kloop/block variants (single core, L1) K=%d ===\n", K);
        bench_variant("8x16-piped", ime_kloop_8x16_piped, K, reps, a0, a1, b0, b1, b2,
                      b3, KB, C, apanel, bpanel, 0);
        bench_variant("8x16-opt", ime_kloop_8x16_opt, K, reps, a0, a1, b0, b1, b2, b3,
                      KB, C, apanel, bpanel, 0);
        bench_variant("8x16-burst", ime_kloop_8x16_burst, K, reps, a0, a1, b0, b1, b2,
                      b3, KB, C, apanel, bpanel, 0);
        bench_block("8x16-block-legacy", ime_block_8x16_legacy, K,
                    reps / 4 > 0 ? reps / 4 : 1, a0, a1, b0, b1, b2, b3, KB, C, apanel,
                    bpanel);
        bench_block("8x16-block-megablock", ime_block_8x16, K, reps / 4 > 0 ? reps / 4 : 1,
                    a0, a1, b0, b1, b2, b3, KB, C, apanel, bpanel);
        if (K == 256) {
            const int8_t *b4 = Bp + (size_t)4 * KB * TILE_BYTES;
            const int8_t *b5 = Bp + (size_t)5 * KB * TILE_BYTES;
            const int8_t *b6 = Bp + (size_t)6 * KB * TILE_BYTES;
            const int8_t *b7 = Bp + (size_t)7 * KB * TILE_BYTES;
            for (int w = 0; w < 32; w++) {
                zero_acc();
                ime_kloop_4x32_ilv(a0, b0, b1, b2, b3, b4, b5, b6, b7, KB);
            }
            double t0 = secs();
            for (long r = 0; r < reps; r++) {
                zero_acc();
                ime_kloop_4x32_ilv(a0, b0, b1, b2, b3, b4, b5, b6, b7, KB);
            }
            double dt = secs() - t0;
            const double vmadot_per = 8.0 * (double)KB;
            const double gops = vmadot_per * (double)reps / dt * 256.0 / 1e9;
            const double cyc = 1.6e9 / (vmadot_per * (double)reps / dt);
            printf("4x32-kloop K=%d reps=%ld  insn=%.2f GOP/s  %.3f c/vmadot  (4x32 panel)\n",
                   K, reps, gops, cyc);
            for (int w = 0; w < 32; w++)
                ime_block_8x32(a0, a1, b0, b1, b2, b3, b4, b5, b6, b7, KB, C,
                               PEAK_N + GEMM_C_PAD);
            t0 = secs();
            for (long r = 0; r < reps / 4; r++)
                ime_block_8x32(a0, a1, b0, b1, b2, b3, b4, b5, b6, b7, KB, C,
                               PEAK_N + GEMM_C_PAD);
            dt = secs() - t0;
            const double ops8x32 = 2.0 * 8.0 * 32.0 * (double)K * (double)(reps / 4);
            printf("8x32-block K=%d reps=%ld  GEMM=%.2f GOP/s  (fused 2x8x16)\n", K,
                   reps / 4, ops8x32 / dt / 1e9);
        } else {
            printf("(4x32/8x32 wide panels only benchmarked at K=256)\n");
        }
    } else if (variant == 3) {
        bench_variant("block+store", ime_kloop_8x16_piped, K, reps, a0, a1, b0, b1, b2, b3, KB,
                      C, apanel, bpanel, 1);
    } else {
        kfn f = ime_kloop_8x16_piped;
        const char *tag = "piped";
        if (variant == 0) {
            f = ime_kloop_8x16_seq;
            tag = "seq";
        } else if (variant == 1) {
            f = ime_kloop_8x16_ilv;
            tag = "ilv";
        }
        bench_variant(tag, f, K, reps, a0, a1, b0, b1, b2, b3, KB, C, apanel, bpanel, 0);
    }

    free(A);
    free(B);
    free(Ap - GEMM_BUF_PAD);
    free(Bp - GEMM_BUF_PAD);
    free(C);
    return 0;
}

#else

int main(void)
{
    printf("ime-bench-peak requires X60 IME build (make board-peak)\n");
    return 1;
}

#endif
