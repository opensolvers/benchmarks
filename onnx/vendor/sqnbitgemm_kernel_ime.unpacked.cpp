/*++
    IME SQNBit CompInt8 — A-hoist + pack-time B unpack (int8 [n][kblk][BlkLen]).
    Driver ldb comes from Q4BitGemmPackedBColumnStrideBytes (= BlockCountK * BlkLen).
--*/

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

namespace sqnbitgemm_ime
{

constexpr size_t TM = 4;
constexpr size_t TN = 4;
constexpr size_t TK = 8;
constexpr size_t TILE_BYTES = TM * TK;
constexpr size_t MAX_NSUB = 32;
constexpr size_t MAX_TILE_BYTES = MAX_NSUB * TILE_BYTES;

static inline void
ime_tile_4x4(const int8_t* a, const int8_t* b, long nk, int32_t* out)
{
    __asm__ volatile(
        ".option arch, +xsmtvdot                \n\t"
        "li          t3, 32                     \n\t"
        "vsetvli     zero, t3, e8, m1, ta, ma   \n\t"
        "vmv.v.i     v16, 0                     \n\t"
        "vmv.v.i     v17, 0                     \n\t"
        "mv          t0, %[nk]                  \n\t"
        "1:                                     \n\t"
        "vle8.v      v8, (%[a])                 \n\t"
        "vle8.v      v9, (%[b])                 \n\t"
        "addi        %[a], %[a], 32             \n\t"
        "addi        %[b], %[b], 32             \n\t"
        "smt.vmadot  v16, v8, v9                \n\t"
        "addi        t0, t0, -1                 \n\t"
        "bnez        t0, 1b                     \n\t"
        "vsetivli    zero, 8, e32, m1, ta, ma   \n\t"
        "vse32.v     v16, (%[o0])               \n\t"
        "vse32.v     v17, (%[o1])               \n\t"
        : [a] "+r"(a), [b] "+r"(b)
        : [nk] "r"(nk), [o0] "r"(out), [o1] "r"(out + 8)
        : "t0", "t3", "v8", "v9", "v16", "v17", "memory");
}

void
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

// Unpacked int8 per weight: [n][kblk][BlkLen] — driver ldb = BlockCountK * BlkLen.
size_t
Q4BitGemmPackQuantBDataSize(
    size_t N, size_t K, size_t BlkLen, bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*cfg*/)
{
    if (ComputeType != SQNBIT_CompInt8 || (BlkLen % TK) != 0) {
        return 0;
    }
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    return N * BlockCountK * BlkLen;
}

void
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

size_t
Q4BitGemmPackedBColumnStrideBytes(
    size_t BlkBitWidth, size_t BlkLen, size_t BlockCountK,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*cfg*/)
{
    if (BlkBitWidth != 4) {
        return 0;
    }
    return BlockCountK * BlkLen;
}

size_t
QNBitGemmPerGemmWorkspaceSize(
    size_t M, size_t /*N*/, size_t K, size_t BlkLen, bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType, size_t /*BlkBitWidth*/,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*cfg*/)
{
    if (ComputeType != SQNBIT_CompInt8) {
        return 0;
    }
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    return M * BlockCountK * Q8BlkSize(BlkLen);
}

size_t
QNBitGemmPerGemmWorkspaceAlignment(size_t /*BlkLen*/, MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/)
{
    return Q8BlkAlignment();
}

size_t
SQ4BitGemmKernel_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM, size_t CountN, size_t /*CountK*/,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias)
{
    assert((BlkLen % TK) == 0);
    assert((BlkLen / TK) <= MAX_NSUB);

    const size_t nsub = BlkLen / TK;
    const size_t qa_blk = Q8BlkSize(BlkLen);
    const size_t qa_row = BlockCountK * qa_blk;
    const size_t b_col_bytes = BlockCountK * BlkLen;  // unpacked int8 column stride
    const int8_t* Bp = reinterpret_cast<const int8_t*>(QuantBData);
    const size_t zp_blk_bytes = MlasDivRoundup(BlockCountK, size_t{2});

    alignas(32) int8_t Atile[MAX_TILE_BYTES];
    alignas(32) int8_t Btile[MAX_TILE_BYTES];
    int32_t tacc[TM * TN];

    for (size_t m = 0; m < CountM; m += TM) {
        const size_t mrows = std::min(TM, CountM - m);

        for (size_t r = 0; r < mrows; ++r) {
            float* crow = C + (m + r) * ldc;
            if (Bias) {
                std::memcpy(crow, Bias, CountN * sizeof(float));
            } else {
                std::memset(crow, 0, CountN * sizeof(float));
            }
        }

        for (size_t kb = 0; kb < BlockCountK; ++kb) {
            float as[TM] = {0, 0, 0, 0};
            int32_t asum[TM] = {0, 0, 0, 0};
            for (size_t r = 0; r < TM; ++r) {
                if (r >= mrows) {
                    for (size_t s = 0; s < nsub; ++s)
                        std::memset(&Atile[s * TILE_BYTES + r * TK], 0, TK);
                    continue;
                }
                const std::byte* ab = QuantA + (m + r) * qa_row + kb * qa_blk;
                as[r] = Q8BlkScale(ab);
                const int8_t* ad = Q8BlkData(ab);
                int32_t s_sum = 0;
                for (size_t s = 0; s < nsub; ++s)
                    for (size_t k = 0; k < TK; ++k) {
                        const int8_t v = ad[s * TK + k];
                        Atile[s * TILE_BYTES + r * TK + k] = v;
                        s_sum += v;
                    }
                asum[r] = s_sum;
            }

            for (size_t n = 0; n < CountN; n += TN) {
                const size_t ncols = std::min(TN, CountN - n);

                for (size_t c = 0; c < TN; ++c) {
                    if (c >= ncols) {
                        for (size_t s = 0; s < nsub; ++s)
                            std::memset(&Btile[s * TILE_BYTES + c * TK], 0, TK);
                        continue;
                    }
                    const int8_t* bblk =
                        Bp + (n + c) * b_col_bytes + kb * BlkLen;
                    for (size_t s = 0; s < nsub; ++s)
                        for (size_t k = 0; k < TK; ++k)
                            Btile[s * TILE_BYTES + c * TK + k] = bblk[s * TK + k];
                }

                ime_tile_4x4(Atile, Btile, static_cast<long>(nsub), tacc);

                for (size_t c = 0; c < ncols; ++c) {
                    const size_t nn = n + c;
                    int zp = 8;
                    if (QuantBZeroPoint) {
                        const std::byte* zpp =
                            QuantBZeroPoint + nn * zp_blk_bytes + (kb >> 1);
                        const int zb = static_cast<int>(std::to_integer<uint8_t>(*zpp));
                        zp = (kb & 1) ? (zb >> 4) : (zb & 0x0F);
                    }
                    const float bscale = QuantBScale[nn * BlockCountK + kb];
                    for (size_t r = 0; r < mrows; ++r) {
                        const int32_t centered = tacc[r * TN + c] - zp * asum[r];
                        C[(m + r) * ldc + nn] +=
                            as[r] * bscale * static_cast<float>(centered);
                    }
                }
            }
        }
    }
    return CountM;
}

}  // namespace sqnbitgemm_ime

const MLAS_QNBIT_GEMM_DISPATCH&
GetMlasQNBitGemmDispatchIme()
{
    static const MLAS_QNBIT_GEMM_DISPATCH dispatch = []() {
        MLAS_QNBIT_GEMM_DISPATCH d;
        d.Q4BitGemmPackQuantBDataSize = sqnbitgemm_ime::Q4BitGemmPackQuantBDataSize;
        d.SQ4BitGemmPackQuantBData = sqnbitgemm_ime::SQ4BitGemmPackQuantBData;
        d.Q4BitGemmPackedBColumnStrideBytes = sqnbitgemm_ime::Q4BitGemmPackedBColumnStrideBytes;
        d.QNBitGemmPerGemmWorkspaceSize = sqnbitgemm_ime::QNBitGemmPerGemmWorkspaceSize;
        d.QNBitGemmPerGemmWorkspaceAlignment = sqnbitgemm_ime::QNBitGemmPerGemmWorkspaceAlignment;
        d.SQ4BitGemmKernel_CompInt8 = sqnbitgemm_ime::SQ4BitGemmKernel_CompInt8;
        d.QuantizeARow_CompInt8 = sqnbitgemm_ime::QuantizeARow_CompInt8;
        return d;
    }();
    return dispatch;
}
