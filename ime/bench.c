/*
 * bench.c - correctness + throughput harness for the s8s8s32 IME core.
 *
 * Fills A(MxK), B(NxK) int8 deterministically, computes C=A*B^T three ways and
 * cross-checks them bit-exactly against the scalar reference (integer math, so
 * equality is exact), reporting GOP/s for each:
 *   ref(scalar)  - ground truth (portable)
 *   packed-ref   - shared packing + scalar tile multiply (portable)
 *   rvv          - RVV int8 baseline            (X60)
 *   ime          - smt.vmadot microkernel       (X60)
 *
 * Build: make host   (portable: ref + packed-ref only)
 *        make board  (adds rvv + ime on the X60)
 * Run:   ./ime-bench [M] [N] [K]   (default 512 512 512; M%4=N%4=K%8=0)
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gemm_s8s8s32.h"

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

static int check(const char *name, const int32_t *got, const int32_t *ref, size_t n)
{
    for (size_t i = 0; i < n; i++)
        if (got[i] != ref[i]) {
            printf("  %-12s FAIL at %zu: got %d want %d\n", name, i, got[i], ref[i]);
            return 1;
        }
    return 0;
}

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 512;
    int N = argc > 2 ? atoi(argv[2]) : 512;
    int K = argc > 3 ? atoi(argv[3]) : 512;
    if (M % TM || N % TN || K % TK) {
        printf("need M%%%d==0 N%%%d==0 K%%%d==0\n", TM, TN, TK);
        return 1;
    }
    /* int32 accumulator overflows only if K*127*127 > 2^31 (K > ~133000). */
    int8_t *A = malloc((size_t)M * K), *B = malloc((size_t)N * K);
    int8_t *Ap = malloc((size_t)M * K), *Bp = malloc((size_t)N * K);
    int32_t *Cref = malloc((size_t)M * N * 4), *Ctest = malloc((size_t)M * N * 4);
    if (!A || !B || !Ap || !Bp || !Cref || !Ctest) {
        printf("alloc fail\n");
        return 1;
    }
    fill(A, (size_t)M * K, 1u);
    fill(B, (size_t)N * K, 2u);

    printf("M=%d N=%d K=%d  (%.1f MMAC)\n", M, N, K, (double)M * N * K / 1e6);
    const double ops = 2.0 * (double)M * N * K;

    double t = secs();
    gemm_ref(A, B, Cref, M, N, K);
    t = secs() - t;
    printf("%-12s %s %8.3f GOP/s  (%.4f s)\n", "ref(scalar)", "--  ", ops / t / 1e9, t);

#define RUN(LABEL, REPS, CALL)                                                   \
    do {                                                                         \
        CALL;                                                                    \
        double _t = secs();                                                      \
        for (int _r = 0; _r < (REPS); _r++)                                      \
            CALL;                                                                \
        _t = (secs() - _t) / (REPS);                                             \
        int _b = check(LABEL, Ctest, Cref, (size_t)M * N);                       \
        printf("%-12s %s %8.3f GOP/s  (%.4f s/rep)\n", LABEL, _b ? "FAIL" : "ok  ",\
               ops / _t / 1e9, _t);                                              \
    } while (0)

    RUN("packed-ref", 1, gemm_packed_ref(A, B, Ctest, M, N, K, Ap, Bp));
#if defined(__riscv)
    RUN("rvv", 5, gemm_rvv(A, B, Ctest, M, N, K));
#if !defined(GEMM_NO_IME)
    RUN("ime", 5, gemm_ime(A, B, Ctest, M, N, K, Ap, Bp));
#endif
#endif
    free(A); free(B); free(Ap); free(Bp); free(Cref); free(Ctest);
    return 0;
}
