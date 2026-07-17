/*
 * bench_qnbit_mlas.cpp - isolated microbenchmark for the MLAS SQNBit int4
 * CompInt8 GEMM kernel on the SpaceMiT X60, linked straight out of ORT's
 * libonnxruntime_mlas.a. This is the exact kernel the MatMulNBits contrib op
 * dispatches to once the model carries accuracy_level=4 (SQNBIT_CompInt8); the
 * whole point of the accuracy_level fix in this directory is to land on THIS
 * kernel instead of the generic dequant+SGEMM fallback.
 *
 * It reproduces the contrib op's call path (matmul_nbits.cc): size the packed-B
 * buffer with MlasQNBitGemmPackQuantBDataSize, pack B with a single
 * MlasQNBitGemmPackQuantBData call, size the workspace with
 * MlasQNBitGemmBatchWorkspaceSize, and drive MlasQNBitGemmBatch<float> with
 * SQNBIT_CompInt8. thread_pool is nullptr so we measure a single core serially.
 *
 * NOTE: this RISC-V IME backend registers only the plain SQ4BitGemmPackQuantBData
 * (a memcpy) - not the x64 SQ4BitGemmPackQuantBDataAndBlkSum variant. Scales stay
 * external and are passed to the kernel at compute time via QuantBScale, so there
 * is a single data-only pack call. (The x64 two-call recipe, whose second call
 * passes QuantBData==nullptr to fold block sums, would memcpy from NULL here.)
 *
 * 4-bit, BlkLen=32, no zero point (symmetric, default zp=8). Kernel throughput
 * is data-independent, so a validly-sized random-nibble B + finite fp32 scales
 * give the true GOP/s; values are not bit-checked here (a finite/NaN guard on C
 * catches a broken layout). Shapes default to the FFN matmuls in the int4 model
 * (K=4096 N=11008 and K=11008 N=4096) at M=1 (token-at-a-time decode).
 * SPDX-License-Identifier: MIT
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "mlas.h"
#include "mlas_qnbit.h"

enum { BLKBITS = 4, BLKLEN = 32 };

static double secs(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    size_t M = argc > 1 ? (size_t)atol(argv[1]) : 1;
    size_t K = argc > 2 ? (size_t)atol(argv[2]) : 4096;
    size_t N = argc > 3 ? (size_t)atol(argv[3]) : 11008;
    int reps = argc > 4 ? atoi(argv[4]) : 50;

    if (K % BLKLEN) {
        printf("need K %% %d == 0\n", BLKLEN);
        return 1;
    }

    const MLAS_QNBIT_GEMM_COMPUTE_TYPE ct = SQNBIT_CompInt8;
    const bool has_zp = false;
    MLAS_BACKEND_KERNEL_SELECTOR_CONFIG cfg; /* POD defaults (use_kleidiai=true, unused on RISC-V) */

    if (!MlasIsQNBitGemmAvailable(BLKBITS, BLKLEN, ct)) {
        printf("MlasIsQNBitGemmAvailable(nbits=%d blklen=%d CompInt8) == false on this build\n",
               BLKBITS, BLKLEN);
        return 3;
    }

    const size_t k_blks = K / BLKLEN;
    /* Raw quantized B: 4-bit nibbles, N*K/2 bytes; scales: one fp32 per (col,block). */
    std::vector<uint8_t> qb((size_t)N * K / 2);
    std::vector<float> scales((size_t)N * k_blks);
    std::vector<float> A((size_t)M * K);
    std::vector<float> C((size_t)M * N);

    /* Deterministic finite fill. Random nibbles for B; small finite scales. */
    uint32_t s = 0x9e3779b9u;
    auto nextr = [&]() { s = s * 1664525u + 1013904223u; return s; };
    for (size_t i = 0; i < qb.size(); i++) qb[i] = (uint8_t)(nextr() >> 24);
    for (size_t i = 0; i < scales.size(); i++)
        scales[i] = 0.01f + (float)((nextr() >> 20) & 0x3ff) * (0.05f / 1024.0f);
    for (size_t i = 0; i < A.size(); i++)
        A[i] = ((int32_t)(nextr() >> 8) - 8388608) / 8388608.0f;

    /* --- Pack B exactly like MatMulNBits::PrePack (CompInt8, two calls). --- */
    size_t packed_size =
        MlasQNBitGemmPackQuantBDataSize(N, K, BLKBITS, BLKLEN, has_zp, ct, &cfg);
    if (packed_size == 0) {
        printf("MlasQNBitGemmPackQuantBDataSize == 0 (kernel refuses these params)\n");
        return 4;
    }
    std::vector<std::byte> packed(packed_size);
    std::memset(packed.data(), 0, packed_size);
    /* 1) pack the quantized data (thread_pool = nullptr -> serial) */
    MlasQNBitGemmPackQuantBData(N, K, BLKBITS, BLKLEN, ct, qb.data(), packed.data(),
                                /*QuantBScale*/ nullptr, has_zp,
                                /*QuantBZeroPoint*/ nullptr, nullptr, &cfg);
    /* NOTE: RISC-V IME CompInt8 keeps scales EXTERNAL (no BlkSum fold).
       The dispatch registers plain SQ4BitGemmPackQuantBData (a memcpy);
       there is no ...AndBlkSum variant, so a second finalize call with
       QuantBData=nullptr would memcpy from NULL and segfault. Scales are
       passed to the kernel at compute time via data.QuantBScale below. */

    const size_t workspace_size =
        MlasQNBitGemmBatchWorkspaceSize(M, N, K, /*batch*/ 1, BLKBITS, BLKLEN, has_zp, ct, &cfg);
    std::vector<std::byte> workspace(workspace_size ? workspace_size : 1);

    MLAS_QNBIT_GEMM_DATA_PARAMS<float> data;
    data.A = A.data();
    data.lda = K;
    data.QuantBDataWorkspace = packed.data();
    data.PackedQuantBData = packed.data();
    data.QuantBScale = scales.data();
    data.QuantBZeroPoint = nullptr;
    data.Bias = nullptr;
    data.C = C.data();
    data.ldc = N;

    auto call = [&]() {
        MlasQNBitGemmBatch<float>(M, N, K, /*batch*/ 1, BLKBITS, BLKLEN, ct, &data,
                                  workspace_size ? workspace.data() : nullptr,
                                  /*thread_pool*/ nullptr, &cfg);
    };

    call(); /* warm + finite check */
    int finite = 1;
    double amax = 0;
    for (size_t i = 0; i < C.size(); i++) {
        if (!std::isfinite(C[i])) finite = 0;
        amax = std::max(amax, (double)std::fabs(C[i]));
    }

    const double ops = 2.0 * (double)M * N * K;
    if (reps < 1) reps = 1;
    std::vector<double> g(reps);
    for (int r = 0; r < reps; r++) {
        double t0 = secs();
        call();
        double t1 = secs();
        g[r] = ops / (t1 - t0) / 1e9;
    }
    qsort(g.data(), reps, sizeof(double), cmp_double);
    printf("qnbit-mlas CompInt8 nbits=4 blklen=%d M=%zu K=%zu N=%zu reps=%d  %s  "
           "min/med/max = %.2f/%.2f/%.2f GOP/s  (packedB=%zu B, ws=%zu B, |C|max=%.3g)\n",
           BLKLEN, M, K, N, reps, finite ? "finite" : "NONFIN",
           g[0], g[reps / 2], g[reps - 1], packed_size, workspace_size, amax);
    return finite ? 0 : 2;
}
