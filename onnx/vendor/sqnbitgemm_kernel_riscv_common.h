/*++

Module Name:
    sqnbitgemm_kernel_riscv_common.h

Abstract:
    Shared plumbing for the RISC-V scalar and RVV SQNBit (block-quantized 4-bit)
    CompInt8 GEMM kernels: the fp32->int8 A quantizer, a "linear" B pack (raw
    4-bit -> int8 0..15, [n][kblk][BlkLen]) that both kernels consume, and the
    workspace sizing. Only the inner block dot-product differs between the two
    backends (scalar loop vs RVV intrinsics), so it lives in each kernel file.
    The IME backend keeps its own tiled pack (for smt.vmadot) in
    sqnbitgemm_kernel_ime.cpp.

--*/

#pragma once

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

#include <algorithm>
#include <cmath>

namespace sqnbitgemm_riscv_common
{

// fp32 activation row -> Q8 blocks, each [fp32 scale][BlkLen int8], symmetric.
inline void
QuantizeARow_CompInt8(size_t BlkLen, const float* A, size_t CountK, std::byte* QuantA)
{
    std::byte* blk = QuantA;
    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t klen = std::min(BlkLen, CountK - k);
        float amax = 0.0f;
        for (size_t i = 0; i < klen; ++i) {
            const float v = std::fabs(A[k + i]);
            if (v > amax) amax = v;
        }
        const float scale = amax / 127.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        Q8BlkScale(blk) = scale;
        int8_t* data = Q8BlkData(blk);
        for (size_t i = 0; i < BlkLen; ++i) {
            if (i < klen) {
                int q = static_cast<int>(std::lrintf(A[k + i] * inv));
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                data[i] = static_cast<int8_t>(q);
            } else {
                data[i] = 0;
            }
        }
        blk += Q8BlkSize(BlkLen);
    }
}

// Linear packed-B size: one int8 per weight, [n][kblk][BlkLen].
inline size_t
Q4BitGemmPackQuantBDataSize(
    size_t N, size_t K, size_t BlkLen, bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*cfg*/)
{
    if (ComputeType != SQNBIT_CompInt8) {
        return 0;
    }
    return N * MlasDivRoundup(K, BlkLen) * BlkLen;
}

// Unpack the raw 4-bit values (0..15) into int8 in [n][kblk][BlkLen] order.
inline void
SQ4BitGemmPackQuantBData(
    size_t N, size_t K, size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const std::byte* QuantBDataBegin, std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* /*ThreadPool*/,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*cfg*/)
{
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t blk_bytes = BlkLen / 2;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(QuantBDataBegin);
    int8_t* dst = reinterpret_cast<int8_t*>(PackedQuantBDataBegin);
    for (size_t n = 0; n < N; ++n) {
        for (size_t kb = 0; kb < BlockCountK; ++kb) {
            const uint8_t* blk = src + (n * BlockCountK + kb) * blk_bytes;
            for (size_t i = 0; i < BlkLen; ++i) {
                const int byte = blk[i >> 1];
                *dst++ = static_cast<int8_t>((i & 1) ? (byte >> 4) : (byte & 0x0F));
            }
        }
    }
}

inline size_t
QNBitGemmPerGemmWorkspaceSize(
    size_t M, size_t /*N*/, size_t K, size_t BlkLen, bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType, size_t /*BlkBitWidth*/,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*cfg*/)
{
    if (ComputeType != SQNBIT_CompInt8) {
        return 0;
    }
    return M * MlasDivRoundup(K, BlkLen) * Q8BlkSize(BlkLen);
}

inline size_t
QNBitGemmPerGemmWorkspaceAlignment(size_t /*BlkLen*/, MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/)
{
    return Q8BlkAlignment();
}

// Per-(column, block) 4-bit zero point (0..15), or 8 (symmetric) when absent.
inline int
BlockZeroPoint(const std::byte* QuantBZeroPoint, size_t n, size_t kb, size_t BlockCountK)
{
    if (QuantBZeroPoint == nullptr) {
        return 8;
    }
    const size_t zp_bytes = MlasDivRoundup(BlockCountK, size_t{2});
    const int byte = static_cast<int>(
        std::to_integer<uint8_t>(QuantBZeroPoint[n * zp_bytes + (kb >> 1)]));
    return (kb & 1) ? (byte >> 4) : (byte & 0x0F);
}

}  // namespace sqnbitgemm_riscv_common
