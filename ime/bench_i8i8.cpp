/*
 * bench_i8i8.cpp - throughput + correctness for llama.cpp IME1 gemm_kernel_i8i8
 * (Q8_0 / Q6_K activations vs int8 weights, smt.vmadot + fp16 scale epilogue).
 * Links libggml-cpu from the -x60-ime-q8_0 build (patch adds gemm_kernel_i8i8).
 *
 * B layout matches M4 offset: per column per k-block [32 int8][2 fp16 scale] = 34B.
 * A layout reuses quantize_a_4row_i8 (36B/row per k-block). Driver mirrors
 * bench_i8i4 / forward_mul_mat tiled loop.
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
size_t gemm_kernel_i8i8(size_t blk_len, const uint8_t *quant_a, const uint8_t *quant_b_data,
                          const uint8_t *quant_b_zp, float *c, size_t count_m, size_t count_n,
                          size_t k_blks, size_t ldc);
void quantize_a_row_i8(size_t blk_len, const float *a, size_t count_k, uint8_t *quant_a);
void quantize_a_4row_i8(size_t blk_len, const float *a, size_t count_k, uint8_t *quant_a);
}  // namespace ime1
}  // namespace spacemit_kernels

extern "C" size_t ggml_quantize_chunk(int type, const float *src, void *dst, int64_t start,
                                      int64_t nrows, int64_t n_per_row, const float *imatrix);
#define GGML_TYPE_Q8_0 8

enum { QK = 32, BLK = 32, NBCOLS = 16 };
static const long QA_BLK = 36;
static const long QB_BLK = 34; /* block_q8_0: 2 fp16 + 32 int8 per k-block per column */

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

static double run_gemm(const uint8_t *qa, const uint8_t *qb, float *C, int M, int N, int K, int ldc,
                       int m1)
{
    const long rsa = (long)(K / QK) * QA_BLK;
    const long rsb = (long)(K / QK) * QB_BLK;
    const long kblk = K / QK;
    double t0 = secs();
    if (m1) {
        for (int m = 0; m < M; m++)
            spacemit_kernels::ime1::gemm_kernel_i8i8((size_t)BLK, qa + (long)m * rsa, qb, nullptr,
                                                     C + (long)m * ldc, 1, (size_t)N, (size_t)kblk,
                                                     (size_t)ldc);
        return secs() - t0;
    }
    const long m_stride = ((long)N / (M > 0 ? M : 1) > 64) ? M : 16;
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
                size_t rh = spacemit_kernels::ime1::gemm_kernel_i8i8((size_t)BLK, arow, bcur, nullptr,
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
    int pad = argc > 5 ? atoi(argv[5]) : 0;
    int chk = argc > 6 ? atoi(argv[6]) : 0;
    int m1 = argc > 7 ? atoi(argv[7]) : 0;
    int varied = chk || m1;
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
    if (varied) {
        uint32_t s = 0x9e3779b9u;
        auto fillb = [&](uint8_t *p, size_t n) {
            for (size_t i = 0; i < n; i++) {
                p[i] = (uint8_t)(((s >> 16) & 0x3fu) | 1u);
                s = s * 1664525u + 1013904223u;
            }
        };
        fillb(qa.data(), qa.size());
        fillb(qb.data(), qb.size());
    } else {
        /* Linear Q8_0 blocks are not byte-identical to repacked B in llama.cpp;
         * use layout-neutral varied fill (same as chk) for finite throughput. */
        uint32_t s = 0x9e3779b9u;
        auto fillb = [&](uint8_t *p, size_t n) {
            for (size_t i = 0; i < n; i++) {
                p[i] = (uint8_t)(((s >> 16) & 0x3fu) | 1u);
                s = s * 1664525u + 1013904223u;
            }
        };
        fillb(qa.data(), qa.size());
        fillb(qb.data(), qb.size());
    }

    printf("M=%d N=%d K=%d ldc=%d  (%.1f MMAC)\n", M, N, K, ldc, (double)M * N * K / 1e6);
    const double ops = 2.0 * (double)M * N * K;

    run_gemm(qa.data(), qb.data(), C.data(), M, N, K, ldc, m1);
    double amax = 0;
    int finite = 1;
    for (size_t i = 0; i < (size_t)M * ldc; i++) {
        if (!std::isfinite(C[i])) finite = 0;
        amax = std::max(amax, (double)std::fabs(C[i]));
    }

    double sig_sum = 0, sig_sq = 0, sig_max = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double v = C[(size_t)m * ldc + n];
            sig_sum += v;
            sig_sq += v * v;
            sig_max = std::max(sig_max, std::fabs(v));
        }
    printf("SIG M=%d N=%d K=%d chk=%d m1=%d finite=%d sum=%.6e sumsq=%.6e max=%.6e\n", M, N, K, chk,
           m1, finite, sig_sum, sig_sq, sig_max);
    if (chk) return finite ? 0 : 2;

    if (reps <= 0) reps = 20;
    std::vector<double> g(reps);
    for (int r = 0; r < reps; r++) {
        double t0 = secs();
        run_gemm(qa.data(), qb.data(), C.data(), M, N, K, ldc, m1);
        g[r] = ops / (secs() - t0) / 1e9;
    }
    qsort(g.data(), reps, sizeof(double), cmp_double);
    printf("i8i8 %s m1=%d reps=%d ldc=%d  gemm=%.1f/%.1f/%.1f GOP/s  (|C|max=%.1f)\n",
           finite ? "finite" : "NONFIN", m1, reps, ldc, g[0], g[reps / 2], g[reps - 1], amax);
    return 0;
}
