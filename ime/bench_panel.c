/*
 * bench_panel.c - step-2 panel / memory tuning (nc sweep, megakernel, offline B).
 *
 *   ./ime-bench-panel [M] [N] [K] [reps] [anti_alias]
 *
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gemm_s8s8s32.h"
#include "gemm_panel.h"
#include "tcm.h"

#if defined(__riscv) && !defined(GEMM_NO_IME)

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

static double bench_tcm_b(int reps, const int8_t *A, const int8_t *B, int8_t *Ap,
                          int8_t *Bp_tcm, int32_t *C, int M, int N, int K, int ldc)
{
    gemm_ime_tcm_b(A, B, C, M, N, K, Ap, Bp_tcm, ldc);
    double t0 = secs();
    for (int r = 0; r < reps; r++)
        gemm_ime_compute_tcm_b(A, B, Ap, Bp_tcm, C, M, N, K, ldc, 1);
    return (secs() - t0) / (double)reps;
}

static double bench_tcm_offline_fused(int reps, const int8_t *A, int8_t *Ap,
                                      const int8_t *Bp_tcm, int32_t *C, int M, int N, int K,
                                      int ldc)
{
    gemm_ime_compute_tcm_offline_b(A, Ap, Bp_tcm, C, M, N, K, ldc);
    double t0 = secs();
    for (int r = 0; r < reps; r++)
        gemm_ime_compute_tcm_offline_b(A, Ap, Bp_tcm, C, M, N, K, ldc);
    return (secs() - t0) / (double)reps;
}

static double bench_offline_b_tcm(int reps, const int8_t *A, int8_t *Ap, int8_t *Bp_tcm,
                                  int32_t *C, int M, int N, int K, int ldc)
{
    gemm_ime_set_fused_pack_a(0);
    gemm_ime_compute_ex(A, NULL, Ap, Bp_tcm, C, M, N, K, ldc, 1, 0);
    double t0 = secs();
    for (int r = 0; r < reps; r++)
        gemm_ime_compute_ex(A, NULL, Ap, Bp_tcm, C, M, N, K, ldc, 1, 0);
    return (secs() - t0) / (double)reps;
}

static double bench_tcm_staged(int reps, const int8_t *A, const int8_t *Bp, int8_t *Ap,
                               int8_t *Bp_tcm, int32_t *C, int M, int N, int K, int ldc)
{
    gemm_ime_compute_tcm_staged(A, Bp, Ap, Bp_tcm, C, M, N, K, ldc, 1);
    double t0 = secs();
    for (int r = 0; r < reps; r++)
        gemm_ime_compute_tcm_staged(A, Bp, Ap, Bp_tcm, C, M, N, K, ldc, 1);
    return (secs() - t0) / (double)reps;
}

static int matrices_match(const int32_t *a, const int32_t *b, int M, int N, int ldc)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if (a[i * ldc + j] != b[i * ldc + j])
                return 0;
    return 1;
}

static double bench_full(int reps, const int8_t *A, const int8_t *B, int8_t *Ap,
                         int8_t *Bp, int32_t *C, int M, int N, int K, int ldc)
{
    gemm_ime(A, B, C, M, N, K, Ap, Bp, ldc);
    double t0 = secs();
    for (int r = 0; r < reps; r++)
        gemm_ime(A, B, C, M, N, K, Ap, Bp, ldc);
    return (secs() - t0) / (double)reps;
}

static double bench_offline_b(int reps, const int8_t *A, const int8_t *B, int8_t *Ap,
                              int8_t *Bp, int32_t *C, int M, int N, int K, int ldc)
{
    gemm_ime_pack(A, B, Ap, Bp, M, N, K);
    gemm_ime_compute_ex(A, B, Ap, Bp, C, M, N, K, ldc, 1, 0);
    double t0 = secs();
    for (int r = 0; r < reps; r++)
        gemm_ime_compute_ex(A, B, Ap, Bp, C, M, N, K, ldc, 1, 0);
    return (secs() - t0) / (double)reps;
}

static double bench_compute_only(int reps, int8_t *Ap, int8_t *Bp, int32_t *C, int M,
                                 int N, int K, int ldc)
{
    gemm_ime_compute(Ap, Bp, C, M, N, K, ldc);
    double t0 = secs();
    for (int r = 0; r < reps; r++)
        gemm_ime_compute(Ap, Bp, C, M, N, K, ldc);
    return (secs() - t0) / (double)reps;
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

    const int ldc = N + (anti_alias ? GEMM_C_PAD : 0);
    const double ops = 2.0 * (double)M * (double)N * (double)K;

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
    gemm_ime_pack(A, B, Ap, Bp, M, N, K);

    printf("=== panel / memory step-2 (core 0) M=%d N=%d K=%d ldc=%d reps=%d ===\n", M,
           N, K, ldc, reps);

    printf("\n--- nc sweep (auto megakernel=0) ---\n");
    printf("%-6s %10s %10s GOP/s\n", "nc", "B_panelKB", "full GEMM");
    const int nc_try[] = { 4, 8, 16, 32, 0, 64, 128 };
    double best_gops = 0;
    int best_nc = 0;
    for (size_t i = 0; i < sizeof(nc_try) / sizeof(nc_try[0]); i++) {
        gemm_ime_set_nc(nc_try[i]);
        gemm_ime_set_megakernel(0);
        const int nc = gemm_ime_get_nc(K);
        const int b_kb = (nc * TN * K) / 1024;
        const double dt = bench_full(reps, A, B, Ap, Bp, C, M, N, K, ldc);
        const double gops = ops / dt / 1e9;
        if (nc_try[i] == 0)
            printf("%-6s %10d %10.2f\n", "auto", b_kb, gops);
        else
            printf("%-6d %10d %10.2f\n", nc, b_kb, gops);
        if (gops > best_gops) {
            best_gops = gops;
            best_nc = nc;
        }
    }

    printf("\n--- megakernel prefetch touches (nc=best=%d) ---\n", best_nc);
    gemm_ime_set_nc(best_nc);
    gemm_ime_set_megakernel(0);
    const double dt_plain = bench_compute_only(reps, Ap, Bp, C, M, N, K, ldc);
    gemm_ime_set_megakernel(1);
    const double dt_mega = bench_compute_only(reps, Ap, Bp, C, M, N, K, ldc);
    printf("compute plain:    %.2f GOP/s\n", ops / dt_plain / 1e9);
    printf("compute megakern: %.2f GOP/s  (%.2f%%)\n", ops / dt_mega / 1e9,
           100.0 * (ops / dt_mega) / (ops / dt_plain));

    printf("\n--- offline B (pack B once, repack A each GEMM) ---\n");
    gemm_ime_set_megakernel(0);
    gemm_ime_set_fused_pack_a(0);
    const double dt_full = bench_full(reps, A, B, Ap, Bp, C, M, N, K, ldc);
    const double dt_offb = bench_offline_b(reps, A, B, Ap, Bp, C, M, N, K, ldc);
    printf("full pack+compute: %.2f GOP/s\n", ops / dt_full / 1e9);
    printf("offline-B:         %.2f GOP/s  (+%.1f%%)\n", ops / dt_offb / 1e9,
           100.0 * (ops / dt_offb - ops / dt_full) / (ops / dt_full));

    printf("\n--- pack_a tuning (RVV pack + fused M-panel) ---\n");
    gemm_ime_set_nc(best_nc);
    {
        double t0 = secs();
        for (int r = 0; r < reps; r++)
            pack_a(A, Ap, M, K);
        const double dt_pack_a = (secs() - t0) / (double)reps;
        printf("pack_a (RVV):      %.2f GB/s  (%.1f ms @ %zu KiB)\n",
               (double)((size_t)M * K) / dt_pack_a / 1e9, dt_pack_a * 1000.0,
               (size_t)M * K / 1024);

        gemm_ime_set_fused_pack_a(1);
        const double dt_fused = bench_offline_b(reps, A, B, Ap, Bp, C, M, N, K, ldc);
        gemm_ime_set_fused_pack_a(0);
        printf("offline-B:         %.2f GOP/s\n", ops / dt_offb / 1e9);
        printf("offline-B fused:   %.2f GOP/s  (%+.1f%% vs offline-B)\n",
               ops / dt_fused / 1e9, 100.0 * (ops / dt_fused - ops / dt_offb) / (ops / dt_offb));
    }

    printf("\n--- TCM optimizations (group tcm / taskset -c 0) ---\n");
    gemm_ime_set_nc(best_nc);
    gemm_ime_set_megakernel(0);
    const size_t panel_bytes = gemm_ime_b_panel_bytes(K);
    const size_t full_b_bytes = gemm_ime_packed_b_bytes(N, K);
    const int panel_kb = (int)(panel_bytes / 1024);
    const int full_b_kb = (int)(full_b_bytes / 1024);
    if (tcm_init(0) != 0) {
        printf("tcm_init failed (run setup-tcm-perms.sh or sudo)\n");
    } else {
        int32_t *Ctcm = ime_buf((size_t)M * (size_t)ldc * 4, anti_alias);
        int8_t *Bp_panel = tcm_malloc(panel_bytes);
        int8_t *Bp_full = tcm_malloc(full_b_bytes);
        if (!Ctcm || !Bp_panel) {
            printf("alloc fail (Ctcm=%p Bp_panel=%p)\n", (void *)Ctcm, (void *)Bp_panel);
        } else {
            gemm_ime(A, B, C, M, N, K, Ap, Bp, ldc);

            printf("panel slab: %d KiB  full Bp: %d KiB (TCM cap %zu KiB)\n", panel_kb,
                   full_b_kb, tcm_block_size() * tcm_blocks_total() / 1024);

            /* 1) repack raw B each panel → TCM (previous path) */
            gemm_ime_tcm_b(A, B, Ctcm, M, N, K, Ap, Bp_panel, ldc);
            int ok = matrices_match(C, Ctcm, M, N, ldc);
            printf("TCM repack-B:      correctness %s\n", ok ? "ok" : "MISMATCH");
            double dt_repack = ok ? bench_tcm_b(reps, A, B, Ap, Bp_panel, Ctcm, M, N, K, ldc)
                                  : 0;

            /* 2) offline B fully in TCM */
            double dt_offline_tcm = 0;
            double dt_offline_tcm_fused = 0;
            if (Bp_full) {
                pack_b(B, Bp_full, N, K);
                gemm_ime_compute_ex(A, NULL, Ap, Bp_full, Ctcm, M, N, K, ldc, 1, 0);
                ok = matrices_match(C, Ctcm, M, N, ldc);
                printf("TCM offline-B:     correctness %s  @ %p\n", ok ? "ok" : "MISMATCH",
                       (void *)Bp_full);
                if (ok)
                    dt_offline_tcm =
                        bench_offline_b_tcm(reps, A, Ap, Bp_full, Ctcm, M, N, K, ldc);
                gemm_ime_set_fused_pack_a(1);
                gemm_ime_compute_tcm_offline_b(A, Ap, Bp_full, Ctcm, M, N, K, ldc);
                ok = matrices_match(C, Ctcm, M, N, ldc);
                if (ok)
                    dt_offline_tcm_fused =
                        bench_tcm_offline_fused(reps, A, Ap, Bp_full, Ctcm, M, N, K, ldc);
                gemm_ime_set_fused_pack_a(0);
                printf("TCM offline fused: correctness %s\n", ok ? "ok" : "MISMATCH");
            } else {
                printf("TCM offline-B:     tcm_malloc(%zu) failed\n", full_b_bytes);
            }

            /* 3) DRAM offline Bp + memcpy panel → TCM */
            gemm_ime_compute_tcm_staged(A, Bp, Ap, Bp_panel, Ctcm, M, N, K, ldc, 1);
            ok = matrices_match(C, Ctcm, M, N, ldc);
            printf("TCM staged-mcpy:   correctness %s\n", ok ? "ok" : "MISMATCH");
            double dt_staged = ok ? bench_tcm_staged(reps, A, Bp, Ap, Bp_panel, Ctcm, M, N, K, ldc)
                                  : 0;

            printf("\n  vs full GEMM %.2f GOP/s | offline-B DRAM %.2f GOP/s\n",
                   ops / dt_full / 1e9, ops / dt_offb / 1e9);
            if (dt_repack > 0)
                printf("  TCM repack-B:    %.2f GOP/s  (%+.1f%% vs full)\n",
                       ops / dt_repack / 1e9,
                       100.0 * (ops / dt_repack - ops / dt_full) / (ops / dt_full));
            if (dt_offline_tcm > 0)
                printf("  TCM offline-B:   %.2f GOP/s  (%+.1f%% vs full, %+.1f%% vs DRAM off-B)\n",
                       ops / dt_offline_tcm / 1e9,
                       100.0 * (ops / dt_offline_tcm - ops / dt_full) / (ops / dt_full),
                       100.0 * (ops / dt_offline_tcm - ops / dt_offb) / (ops / dt_offb));
            if (dt_offline_tcm_fused > 0)
                printf("  TCM off-B fused: %.2f GOP/s  (%+.1f%% vs TCM off-B)\n",
                       ops / dt_offline_tcm_fused / 1e9,
                       100.0 * (ops / dt_offline_tcm_fused - ops / dt_offline_tcm) /
                           (ops / dt_offline_tcm));
            if (dt_staged > 0)
                printf("  TCM staged-mcpy: %.2f GOP/s  (%+.1f%% vs full)\n",
                       ops / dt_staged / 1e9,
                       100.0 * (ops / dt_staged - ops / dt_full) / (ops / dt_full));

            ime_buf_free(Ctcm, anti_alias);
        }
        if (Bp_full)
            tcm_free(Bp_full);
        if (Bp_panel)
            tcm_free(Bp_panel);
        tcm_shutdown();
    }

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
    printf("ime-bench-panel requires X60 IME build (make board-panel)\n");
    return 1;
}

#endif
