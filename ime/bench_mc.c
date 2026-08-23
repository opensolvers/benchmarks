/*
 * bench_mc.c - synthetic IME (s8s8s32 vmadot) throughput: 1 vs N OpenMP threads.
 *
 * Splits M across threads (each runs gemm_ime on an A/C row panel; B shared).
 * Pin the process to cluster-0 cores before running, e.g.:
 *   taskset -c 0   ./ime-bench-mc 768 768 512 30 1
 *   taskset -c 0-3 ./ime-bench-mc 768 768 512 30 4
 *
 * Reports per-thread median GOP/s, aggregate (sum of panel GOP/s), and
 * whole-GEMM wall GOP/s (2*M*N*K / wall_time).
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gemm_s8s8s32.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

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

static int check_panel(const int32_t *got, const int32_t *ref, int m0, int M, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if (got[(size_t)i * N + j] != ref[(size_t)(m0 + i) * N + j])
                return 1;
    return 0;
}

int main(int argc, char **argv)
{
#if !defined(__riscv) || defined(GEMM_NO_IME)
    (void)argc;
    (void)argv;
    printf("ime-bench-mc requires X60 IME build (make board-mc)\n");
    return 1;
#else
    int M = argc > 1 ? atoi(argv[1]) : 768;
    int N = argc > 2 ? atoi(argv[2]) : 768;
    int K = argc > 3 ? atoi(argv[3]) : 512;
    int reps = argc > 4 ? atoi(argv[4]) : 20;
    int nth = argc > 5 ? atoi(argv[5]) : 1;
    if (nth < 1) nth = 1;
    if (M % TM || N % TN || K % TK) {
        printf("need M%%%d==0 N%%%d==0 K%%%d==0\n", TM, TN, TK);
        return 1;
    }
    if (M % (nth * TM)) {
        printf("need M%%%d==0 for %d threads (M=%d)\n", nth * TM, nth, M);
        return 1;
    }
#ifndef _OPENMP
    if (nth > 1) {
        printf("OpenMP not enabled; rebuild with make board-mc\n");
        return 1;
    }
#endif

    const int m_panel = M / nth;
    int8_t *A = malloc((size_t)M * K);
    int8_t *B = malloc((size_t)N * K);
    int32_t *Cref = calloc((size_t)M * N, sizeof(int32_t));
    int32_t *Ctest = calloc((size_t)M * N, sizeof(int32_t));
    int8_t *Ap = malloc((size_t)nth * m_panel * K);
    int8_t *Bp = malloc((size_t)nth * N * K);
    if (!A || !B || !Cref || !Ctest || !Ap || !Bp) {
        printf("alloc fail\n");
        return 1;
    }
    fill(A, (size_t)M * K, 1u);
    fill(B, (size_t)N * K, 2u);

    gemm_ref(A, B, Cref, M, N, K);

    const double ops_total = 2.0 * (double)M * N * K;
    const double ops_panel = 2.0 * (double)m_panel * N * K;

#ifdef _OPENMP
    omp_set_num_threads(nth);
#endif

    /* correctness: one parallel GEMM */
    memset(Ctest, 0, (size_t)M * N * sizeof(int32_t));
#ifdef _OPENMP
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int m0 = tid * m_panel;
        int8_t *Ap_t = Ap + (size_t)tid * m_panel * K;
        int8_t *Bp_t = Bp + (size_t)tid * N * K;
        gemm_ime(A + (size_t)m0 * K, B, Ctest + (size_t)m0 * (size_t)N, m_panel, N, K,
                 Ap_t, Bp_t, N);
    }
#else
    gemm_ime(A, B, Ctest, M, N, K, Ap, Bp, N);
#endif
    int bad = 0;
    for (int t = 0; t < nth; t++) {
        int m0 = t * m_panel;
        bad |= check_panel(Ctest + (size_t)m0 * N, Cref, m0, m_panel, N);
    }

    double *panel_gops = calloc((size_t)nth, sizeof(double));
    double wall0 = secs();
    for (int r = 0; r < reps; r++) {
#ifdef _OPENMP
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int m0 = tid * m_panel;
            int8_t *Ap_t = Ap + (size_t)tid * m_panel * K;
            int8_t *Bp_t = Bp + (size_t)tid * N * K;
            double t0 = secs();
            gemm_ime(A + (size_t)m0 * K, B, Ctest + (size_t)m0 * (size_t)N, m_panel, N, K,
                 Ap_t, Bp_t, N);
            double g = ops_panel / (secs() - t0) / 1e9;
#pragma omp critical
            panel_gops[tid] += g;
        }
#else
        double t0 = secs();
        gemm_ime(A, B, Ctest, M, N, K, Ap, Bp, N);
        panel_gops[0] += ops_total / (secs() - t0) / 1e9;
#endif
    }
    double wall = secs() - wall0;

    double sum_gops = 0;
    printf("M=%d N=%d K=%d threads=%d reps=%d check=%s\n", M, N, K, nth, reps,
           bad ? "FAIL" : "ok");
    for (int t = 0; t < nth; t++) {
        double med = panel_gops[t] / (double)reps;
        printf("  core-panel %d: %.2f GOP/s (panel M=%d)\n", t, med, m_panel);
        sum_gops += med;
    }
    printf("  sum-panel:   %.2f GOP/s\n", sum_gops);
    printf("  wall-total:  %.2f GOP/s  (%.4f s for %d reps)\n", ops_total * reps / wall / 1e9,
           wall, reps);
    printf("  efficiency:  %.2f× vs single-panel (sum/wall if ideal)\n",
           (ops_total * reps / wall / 1e9) / (panel_gops[0] / reps + 1e-30));

    free(A);
    free(B);
    free(Cref);
    free(Ctest);
    free(Ap);
    free(Bp);
    free(panel_gops);
    return bad ? 2 : 0;
#endif
}
