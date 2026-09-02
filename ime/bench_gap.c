/*
 * bench_gap.c - waterfall from silicon insn peak to full GEMM GOP/s.
 *
 *   ./ime-bench-gap [M] [N] [K] [reps] [anti_alias]
 *
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gemm_s8s8s32.h"

#if defined(__riscv) && !defined(GEMM_NO_IME)

static const double CPU_GHZ = 1.6;
static const double OPS_PER_VMADOT = 256.0; /* 128 MACs, 2 ops/MAC */
static const double CEIL_409 = CPU_GHZ * OPS_PER_VMADOT; /* 409.6 @ 1.6 GHz, 1/cycle */
static const double CEIL_512 = 512.0; /* vendor quote (see README gap section) */

static void *ime_buf(size_t nbytes, int anti_alias)
{
    size_t slack = anti_alias ? GEMM_BUF_PAD * 2 : 64;
    void *raw = NULL;
    if (posix_memalign(&raw, 64, nbytes + slack) != 0)
        return NULL;
    return anti_alias ? (char *)raw + GEMM_BUF_PAD : raw;
}

static void ime_buf_free(void *p, int anti_alias)
{
    if (!p)
        return;
    free(anti_alias ? (char *)p - GEMM_BUF_PAD : p);
}

static double secs(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void fill(int8_t *p, size_t n, uint32_t s)
{
    for (size_t i = 0; i < n; i++) {
        p[i] = (int8_t)(s >> 24);
        s = s * 1664525u + 1013904223u;
    }
}

/* vmadot count for the 8×16-blocked main path (+ 4×4 edges). */
static long count_vm_adots(int M, int N, int K)
{
    const int KB = K / TK, MT = M / TM, NT = N / TN;
    int nc = 32768 / (K > 0 ? K : 1);
    nc &= ~3;
    if (nc < 4)
        nc = 4;
    long v = 0;
    for (int n0 = 0; n0 < NT; n0 += nc) {
        int n1 = (n0 + nc < NT) ? n0 + nc : NT;
        for (int mb = 0; mb < MT; mb += 2) {
            for (int nb = n0; nb < n1; nb += 4) {
                if (mb + 2 <= MT && nb + 4 <= n1)
                    v += 8L * KB;
                else {
                    int mlim = (mb + 2 <= MT) ? 2 : 1;
                    int nlim = (n1 - nb) < 4 ? (n1 - nb) : 4;
                    v += (long)mlim * (long)nlim * KB;
                }
            }
        }
    }
    return v;
}

static double pct(double x, double ref)
{
    return ref > 0 ? 100.0 * x / ref : 0.0;
}

static void print_row(const char *label, double gops, double dt)
{
    printf("%-22s %8.2f GOP/s  %6.2f%%409  %6.2f%%512  (%.4f s)\n", label, gops,
           pct(gops, CEIL_409), pct(gops, CEIL_512), dt);
}

static double time_reps(int reps, void (*fn)(void *), void *ctx)
{
    for (int w = 0; w < 3; w++)
        fn(ctx);
    double t0 = secs();
    for (int r = 0; r < reps; r++)
        fn(ctx);
    return (secs() - t0) / (double)reps;
}

struct ctx_pack {
    const int8_t *A, *B;
    int8_t *Ap, *Bp;
    int M, N, K;
};

static void do_pack(void *p)
{
    struct ctx_pack *c = (struct ctx_pack *)p;
    gemm_ime_pack(c->A, c->B, c->Ap, c->Bp, c->M, c->N, c->K);
}

struct ctx_compute {
    const int8_t *Ap, *Bp;
    int32_t *C;
    int M, N, K, ldc;
};

static void do_compute(void *p)
{
    struct ctx_compute *c = (struct ctx_compute *)p;
    gemm_ime_compute(c->Ap, c->Bp, c->C, c->M, c->N, c->K, c->ldc);
}

struct ctx_full {
    const int8_t *A, *B;
    int8_t *Ap, *Bp;
    int32_t *C;
    int M, N, K, ldc;
};

static void do_full(void *p)
{
    struct ctx_full *c = (struct ctx_full *)p;
    gemm_ime(c->A, c->B, c->C, c->M, c->N, c->K, c->Ap, c->Bp, c->ldc);
}

struct ctx_kloop {
    const int8_t *a0, *a1, *b0, *b1, *b2, *b3;
    long KB;
};

static void do_kloop(void *p)
{
    struct ctx_kloop *c = (struct ctx_kloop *)p;
    ime_kloop_8x16_piped(c->a0, c->a1, c->b0, c->b1, c->b2, c->b3, c->KB);
}

struct ctx_block {
    const int8_t *a0, *a1, *b0, *b1, *b2, *b3;
    long KB;
    int32_t *C;
    int ldc;
};

static void do_block(void *p)
{
    struct ctx_block *c = (struct ctx_block *)p;
    ime_block_8x16(c->a0, c->a1, c->b0, c->b1, c->b2, c->b3, c->KB, c->C, c->ldc);
}

int main(int argc, char **argv)
{
    const int M = argc > 1 ? atoi(argv[1]) : 768;
    const int N = argc > 2 ? atoi(argv[2]) : 768;
    const int K = argc > 3 ? atoi(argv[3]) : 512;
    const int reps = argc > 4 ? atoi(argv[4]) : 25;
    const int anti_alias = argc > 5 ? atoi(argv[5]) : 1;
    if (M % TM || N % TN || K % TK) {
        printf("need M%%%d==0 N%%%d==0 K%%%d==0\n", TM, TN, TK);
        return 1;
    }
    if (reps < 1) {
        printf("reps must be >= 1\n");
        return 1;
    }

    const int ldc = N + (anti_alias ? GEMM_C_PAD : 0);
    const double gemm_ops = 2.0 * (double)M * (double)N * (double)K;
    const long vmadots = count_vm_adots(M, N, K);
    const long KB = K / TK;

    int8_t *A = ime_buf((size_t)M * K, anti_alias);
    int8_t *B = ime_buf((size_t)N * K, anti_alias);
    int8_t *Ap = ime_buf((size_t)M * K, anti_alias);
    int8_t *Bp = ime_buf((size_t)N * K, anti_alias);
    int32_t *C = ime_buf((size_t)M * (size_t)ldc * 4, anti_alias);
    if (!A || !B || !Ap || !Bp || !C) {
        printf("alloc fail\n");
        return 1;
    }
    fill(A, (size_t)M * K, 1u);
    fill(B, (size_t)N * K, 2u);

    /* One 8×16 micro-tile for isolated kloop / block (L1-resident). */
    const size_t apanel = (size_t)2 * KB * TILE_BYTES;
    const size_t bpanel = (size_t)4 * KB * TILE_BYTES;
    int8_t *At = aligned_alloc(64, apanel);
    int8_t *Bt = aligned_alloc(64, bpanel);
    int32_t *Ct = aligned_alloc(64, (size_t)8 * (N + GEMM_C_PAD) * sizeof(int32_t));
    if (!At || !Bt || !Ct) {
        printf("tile alloc fail\n");
        return 1;
    }
    pack_a_panel(A, At, 0, 2, K);
    pack_b_panel(B, Bt, 0, 4, K);

    const int8_t *a0 = At;
    const int8_t *a1 = At + (size_t)KB * TILE_BYTES;
    const int8_t *b0 = Bt;
    const int8_t *b1 = Bt + (size_t)KB * TILE_BYTES;
    const int8_t *b2 = Bt + (size_t)2 * KB * TILE_BYTES;
    const int8_t *b3 = Bt + (size_t)3 * KB * TILE_BYTES;

    struct ctx_pack cp = {A, B, Ap, Bp, M, N, K};
    struct ctx_compute cc = {Ap, Bp, C, M, N, K, ldc};
    struct ctx_full cf = {A, B, Ap, Bp, C, M, N, K, ldc};
    struct ctx_kloop ck = {a0, a1, b0, b1, b2, b3, KB};
    struct ctx_block cb = {a0, a1, b0, b1, b2, b3, KB, Ct, N + GEMM_C_PAD};

    const long kloop_reps = 200000L;
    const double dt_kloop = time_reps((int)kloop_reps, do_kloop, &ck);
    const double kloop_insn = 8.0 * (double)KB / dt_kloop * OPS_PER_VMADOT / 1e9;
    const double cyc_kloop = CPU_GHZ * 1e9 / (8.0 * (double)KB / dt_kloop);

    const int block_reps = (int)(kloop_reps / 4);
    const double dt_block = time_reps(block_reps > 0 ? block_reps : 1, do_block, &cb);
    const double block_gemm = 2.0 * 8.0 * 16.0 * (double)K / dt_block / 1e9;

    gemm_ime_pack(A, B, Ap, Bp, M, N, K);
    const double dt_pack = time_reps(reps, do_pack, &cp);
    const double dt_compute = time_reps(reps, do_compute, &cc);
    const double dt_full = time_reps(reps, do_full, &cf);

    const double gops_pack = gemm_ops / dt_pack / 1e9;
    const double gops_compute = gemm_ops / dt_compute / 1e9;
    const double gops_full = gemm_ops / dt_full / 1e9;
    const double pack_bytes = (double)((size_t)M * K + (size_t)N * K);
    const double pack_gbs = pack_bytes / dt_pack / 1e9;

    const double eff_c_vm_compute =
        CPU_GHZ * 1e9 * dt_compute / (double)vmadots;
    const double eff_c_vm_full = CPU_GHZ * 1e9 * dt_full / (double)vmadots;

    printf("=== IME theoretical vs measured gap (core 0, %.1f GHz) ===\n", CPU_GHZ);
    printf("M=%d N=%d K=%d ldc=%d anti_alias=%d  reps=%d\n", M, N, K, ldc,
           anti_alias, reps);
    printf("GEMM ops=%.3f G  vmadots=%ld  (%.1f per 8×16 block)\n", gemm_ops / 1e9,
           vmadots, (double)vmadots / ((double)M / 8.0 * (double)N / 16.0));
    printf("\nSilicon insn ceilings (1 issue/cycle, 256 ops/vmadot):\n");
    printf("  %.1f GOP/s @ %.1f GHz (1 issue/cycle, 256 ops/vmadot)\n", CEIL_409, CPU_GHZ);
    printf("  %.1f GOP/s cpufp peak on K1 (SpacemiT docs / pigirons cpufp @ ~2 GHz)\n",
           CEIL_512);
    printf("\nLayer                          GOP/s   %%409.6   %%512.0   wall time\n");
    print_row("kloop-only (L1 tile)", kloop_insn, dt_kloop);
    print_row("8×16 block+store (L1)", block_gemm, dt_block);
    printf("%-22s %8.2f GB/s   (gather A+B; not MAC throughput)\n", "pack (amortized)",
           pack_gbs);
    print_row("compute (pre-packed)", gops_compute, dt_compute);
    print_row("full GEMM (pack+compute)", gops_full, dt_full);

    printf("\nImplied cycles/vmadot (full problem):\n");
    printf("  compute-only: %.2f  (kloop microbench: %.2f)\n", eff_c_vm_compute,
           cyc_kloop);
    printf("  full GEMM:    %.2f\n", eff_c_vm_full);

    const double pack_frac = dt_pack / (dt_pack + dt_compute);
  printf("\nPack share of compute+pack: %.1f%%  (compute %.1f%%)\n", 100.0 * pack_frac,
         100.0 * (1.0 - pack_frac));

    const double gap_kloop_to_compute = kloop_insn / gops_compute;
    const double gap_compute_to_full = gops_compute / gops_full;
    const double vs_cpufp = gops_full / CEIL_512;
    printf("kloop→compute slowdown: %.2fx  (panel/L2/C-store + not L1-resident)\n",
           gap_kloop_to_compute);
    printf("compute→full overhead:  %.2fx  (re-pack each GEMM rep)\n",
           gap_compute_to_full);
    printf("full GEMM vs cpufp 511: %.1f%%  (%.2fx from silicon insn peak)\n",
           100.0 * vs_cpufp, CEIL_512 / gops_full);
    (void)gops_pack;

    free(At);
    free(Bt);
    free(Ct);
    ime_buf_free(A, anti_alias);
    ime_buf_free(B, anti_alias);
    ime_buf_free(Ap, anti_alias);
    ime_buf_free(Bp, anti_alias);
    ime_buf_free(C, anti_alias);
    return 0;
}

#else

int main(void)
{
    printf("ime-bench-gap requires X60 IME build (make board-gap)\n");
    return 1;
}

#endif
