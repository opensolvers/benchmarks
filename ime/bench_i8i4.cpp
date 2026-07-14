/*
 * bench_i8i4.cpp - raw-throughput microbenchmark for llama.cpp's SpaceMiT IME1
 * int8xint4 GEMM (spacemit_kernels::ime1::gemm_kernel_i8i4), linked straight out
 * of libggml-cpu.so, to compare the block-scaled Q4_0 kernel against ime-bench's
 * pure s8s8s32 ceiling (~42 GOP/s @768 on this X60) and probe the same 32 KB L2
 * cache-set aliasing.
 *
 * It drives the kernel through the *exact* call path forward_mul_mat uses on this
 * board (TCM unavailable -> the tiled no-TCM branch, ime.cpp:493): blk_len=32,
 * NB_COLS=16 col-groups, no zero-point, row_stride_a=(K/32)*36, weights as one
 * contiguous Q4_0 buffer (same byte size as the repacked block_q4_0x16 form).
 * Kernel throughput is data-independent, so a correctly-sized valid-Q4_0 buffer
 * gives the true GOP/s even though the x16 interleave isn't reproduced (values
 * are not bit-checked; the rate is cross-validated against llama-bench pp512).
 * SPDX-License-Identifier: MIT
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

namespace spacemit_kernels {
namespace ime1 {
size_t gemm_kernel_i8i4(size_t blk_len, const uint8_t *quant_a, const uint8_t *quant_b_data,
                        const uint8_t *quant_b_zp, float *c, size_t count_m, size_t count_n,
                        size_t k_blks, size_t ldc);
void quantize_a_row_i8(size_t blk_len, const float *a, size_t count_k, uint8_t *quant_a);
void quantize_a_4row_i8(size_t blk_len, const float *a, size_t count_k, uint8_t *quant_a);
}  // namespace ime1
}  // namespace spacemit_kernels

extern "C" size_t ggml_quantize_chunk(int type, const float *src, void *dst, int64_t start,
                                      int64_t nrows, int64_t n_per_row, const float *imatrix);
#define GGML_TYPE_Q4_0 2

enum { QK = 32, BLK = 32, NBCOLS = 16 };
static const long QA_BLK = 36; /* q8_blk_size(32) = sizeof(float)+32 */
static const long QB_BLK = 18; /* sizeof(block_q4_0) per k-block per column */

static double secs(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void fill_f32(float *p, size_t n, uint32_t s)
{
    for (size_t i = 0; i < n; i++) {
        p[i] = ((int32_t)(s >> 8) - 8388608) / 8388608.0f;
        s = s * 1664525u + 1013904223u;
    }
}

/* Replicate forward_mul_mat's single-thread no-TCM tiled loop (ime.cpp:493-540):
 * M tiled by m_stride, N walked in NB_COLS groups, kernel re-called per its
 * rows_handled return. ldc lets us pad the C leading dim for the aliasing test. */
static double run_gemm(const uint8_t *qa, const uint8_t *qb, float *C, int M, int N, int K, int ldc)
{
    const long rsa = (long)(K / QK) * QA_BLK;
    const long rsb = (long)(K / QK) * QB_BLK;
    const long kblk = K / QK;
    const long m_stride = ((long)N / (M > 0 ? M : 1) > 64) ? M : 16;
    double t0 = secs();
    for (int ms = 0; ms < M; ms += m_stride) {
        int mc = (int)std::min((long)M - ms, m_stride);
        int nblk = (mc == 1) ? N : NBCOLS;
        const uint8_t *bcol = qb;
        for (int ni = 0; ni < N;) {
            int nr = std::min(N - ni, nblk);
            const uint8_t *arow = qa + (long)ms * rsa;
            float *cblk = C + (long)ms * ldc + ni;
            int rem = mc;
            const uint8_t *bcur = bcol;
            while (rem > 0) {
                size_t rh = spacemit_kernels::ime1::gemm_kernel_i8i4((size_t)BLK, arow, bcur, nullptr,
                                                                     cblk, (size_t)rem, (size_t)nr,
                                                                     (size_t)kblk, (size_t)ldc);
                if (!rh) break;
                cblk += rh * ldc;
                arow += rh * rsa;
                rem -= (int)rh;
            }
            ni += nr;
            bcol += (long)nr * rsb;
        }
    }
    return secs() - t0;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 512;
    int N = argc > 2 ? atoi(argv[2]) : 512;
    int K = argc > 3 ? atoi(argv[3]) : 512;
    int reps = argc > 4 ? atoi(argv[4]) : 0;
    int pad = argc > 5 ? atoi(argv[5]) : 0; /* extra floats on the C row stride */
    if (N % NBCOLS || K % QK || M % 4) {
        printf("need M%%4==0 N%%%d==0 K%%%d==0\n", NBCOLS, QK);
        return 1;
    }
    const int ldc = N + pad;
    const long rsa = (long)(K / QK) * QA_BLK;
    std::vector<float> X((size_t)M * K);
    std::vector<uint8_t> qb((size_t)N * (K / QK) * QB_BLK);
    std::vector<uint8_t> qa((size_t)M * rsa);
    std::vector<float> C((size_t)M * ldc);
    fill_f32(X.data(), X.size(), 1u);

    std::vector<float> W((size_t)N * K, 0.5f); /* constant -> Q4_0 scale bytes stay valid
        small normals / zero even when read through the (unreproduced) x16 interleave, so
        the fp path sees no inf/denormal and the data-independent rate is trustworthy. */
    ggml_quantize_chunk(GGML_TYPE_Q4_0, W.data(), qb.data(), 0, N, K, nullptr);
    for (int m = 0; m < M; m += 4)
        spacemit_kernels::ime1::quantize_a_4row_i8((size_t)BLK, X.data() + (size_t)m * K, (size_t)K,
                                                   qa.data() + (size_t)m * rsa);

    printf("M=%d N=%d K=%d ldc=%d  (%.1f MMAC)\n", M, N, K, ldc, (double)M * N * K / 1e6);
    const double ops = 2.0 * (double)M * N * K;

    run_gemm(qa.data(), qb.data(), C.data(), M, N, K, ldc);
    double amax = 0;
    int finite = 1;
    for (size_t i = 0; i < (size_t)M * ldc; i++) {
        if (!std::isfinite(C[i])) finite = 0;
        amax = std::max(amax, (double)std::fabs(C[i]));
    }

    /* gemm[] = kernel only; gemc[] = kernel + copy padded scratch -> contiguous
     * M*N dst (the real cost of the pad fix, since llama.cpp's output is
     * contiguous). Compare baseline gemm@pad0 vs gemc@pad16 for the net gain. */
    std::vector<float> Cout((size_t)M * N);
    if (reps <= 0) reps = 20;
    std::vector<double> g(reps), gc(reps);
    for (int r = 0; r < reps; r++) {
        double t0 = secs();
        run_gemm(qa.data(), qb.data(), C.data(), M, N, K, ldc);
        double t1 = secs();
        for (int m = 0; m < M; m++)
            memcpy(Cout.data() + (size_t)m * N, C.data() + (size_t)m * ldc, (size_t)N * sizeof(float));
        double t2 = secs();
        g[r] = ops / (t1 - t0) / 1e9;
        gc[r] = ops / (t2 - t0) / 1e9;
    }
    qsort(g.data(), reps, sizeof(double), cmp_double);
    qsort(gc.data(), reps, sizeof(double), cmp_double);
    printf("i8i4 %s reps=%d ldc=%d  gemm=%.1f/%.1f/%.1f  gemm+copy=%.1f/%.1f/%.1f GOP/s  (|C|max=%.1f)\n",
           finite ? "finite" : "NONFIN", reps, ldc, g[0], g[reps / 2], g[reps - 1], gc[0],
           gc[reps / 2], gc[reps - 1], amax);
    return 0;
}
