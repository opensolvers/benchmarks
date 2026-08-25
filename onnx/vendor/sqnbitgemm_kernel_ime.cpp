/*++

Module Name:
    sqnbitgemm_kernel_ime.cpp

Abstract:
    SQNBit (block-quantized n-bit) GEMM kernel for the SpaceMiT X60 RISC-V
    "IME" integer matrix extension, for MLAS.

    Computes  C (fp32) = A (int8, block-scaled) x B (4-bit, block-scaled)  for
    ComputeType == SQNBIT_CompInt8, BlkBitWidth == 4. This is the same operation
    as llama.cpp's spacemit `gemm_kernel_i8i4`; the microkernel is ported from the
    verified opensolvers/benchmarks `ime/` s8s8s32 core.

    The extension instruction is `smt.vmadot vd, vs1, vs2`, which performs a
    4x4 int32 tile update  acc += (4x8 int8) . (4x8 int8)^T  at vl=32, e8,
    VLEN=256. It has no compiler intrinsic; the assembler reaches it via
    `.option arch, +xsmtvdot`, which requires a binutils built with SpaceMiT
    xsmtvdot support (binutils-2.46.1 + the add-spacemit-xsmtvdot patch, provided
    through EESSI). qemu-user does not emulate it (X60 silicon only).

    Zero point is folded out of the hot loop using the block-sum identity
        sum_i a_i (b_i - zp) = (sum_i a_i b_i) - zp * (sum_i a_i)
    so the packed weights are the raw 0..15 nibbles and the same kernel serves
    both the symmetric (Q4_0, zp=8) and asymmetric (per-block zp) paths.

--*/

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

namespace sqnbitgemm_ime
{

// One smt.vmadot consumes a 4x8 int8 A-tile and a 4x8 int8 B-tile (B pre-
// transposed: b[col][k]) and updates a 4x4 int32 accumulator.
constexpr size_t TM = 4;   // rows per tile
constexpr size_t TN = 4;   // cols per tile
constexpr size_t TK = 8;   // K per tile
constexpr size_t TILE_BYTES = TM * TK;  // 32: one 4x8 int8 tile

// acc(4x4 int32) = sum over `nk` consecutive 4x8 tiles of  a . b^T.
// `a` and `b` are nk contiguous 32-byte tiles. `out` receives 16 int32,
// row-major: out[r*4 + c] = sum_k a[r][k] * b[c][k].
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
        // v16 = [r0c0..r0c3, r1c0..r1c3], v17 = [r2..., r3...]
        "vsetivli    zero, 8, e32, m1, ta, ma   \n\t"
        "vse32.v     v16, (%[o0])               \n\t"
        "vse32.v     v17, (%[o1])               \n\t"
        : [a] "+r"(a), [b] "+r"(b)
        : [nk] "r"(nk), [o0] "r"(out), [o1] "r"(out + 8)
        : "t0", "t3", "v8", "v9", "v16", "v17", "memory");
}

// ===========================================================================
// A quantization: fp32 row -> Q8 blocks, each [fp32 scale][BlkLen int8].
// Symmetric (zero-point-free) int8, matching MLAS's CompInt8 A format.
// ===========================================================================
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
                data[i] = 0;  // tail padding within the last block
            }
        }
        blk += Q8BlkSize(BlkLen);
    }
}

// B packing must stay column-major [n][kblk][BlkLen/2] with per-column byte
// stride == MLAS driver stride  ldb = k_blks * MlasQNBitBlkDataSizeInBytes(4,
// BlkLen). The driver slices B by  RangeStartN*ldb + n*ldb  when it partitions
// N, so any other layout (e.g. tile-major) reads the wrong column. Tiles for
// smt.vmadot are gathered on the fly in the kernel.
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
    return N * BlockCountK * (BlkLen / 2);
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
    std::memcpy(PackedQuantBDataBegin, QuantBDataBegin, N * BlockCountK * (BlkLen / 2));
}

// ===========================================================================
// Workspace: MLAS quantizes A into this scratch (Q8 blocks) before the kernel.
// ===========================================================================
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

// ===========================================================================
// The kernel.  C[m][n] = Bias[n] + sum_kb Ascale[m][kb]*Bscale[n][kb] *
//     ( sum_i Aq[m][kb][i] * (Bq[n][kb][i] - zp[n][kb]) ).
// Full 4x4 tiles use smt.vmadot; row/col remainders are handled by zero-padded
// tiles so no scalar fallback path is needed.
// ===========================================================================
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
    const size_t nsub = BlkLen / TK;
    const size_t qa_blk = Q8BlkSize(BlkLen);
    const size_t qa_row = BlockCountK * qa_blk;
    const size_t b_col_bytes = BlockCountK * (BlkLen / 2);  // packed 4-bit stride per column == MLAS ldb
    const uint8_t* Bp = reinterpret_cast<const uint8_t*>(QuantBData);
    const size_t zp_blk_bytes = MlasDivRoundup(BlockCountK, size_t{2});  // 4-bit zp, BlockCountK per col

    std::vector<int8_t> Atile(nsub * TILE_BYTES);
    std::vector<int8_t> Btile(nsub * TILE_BYTES);
    int32_t tacc[TM * TN];

    for (size_t m = 0; m < CountM; m += TM) {
        const size_t mrows = std::min(TM, CountM - m);
        for (size_t n = 0; n < CountN; n += TN) {
            const size_t ncols = std::min(TN, CountN - n);

            float acc[TM][TN];
            for (size_t r = 0; r < TM; ++r)
                for (size_t c = 0; c < TN; ++c)
                    acc[r][c] = (r < mrows && c < ncols && Bias) ? Bias[n + c] : 0.0f;

            for (size_t kb = 0; kb < BlockCountK; ++kb) {
                // Gather the 4xBlkLen A block into nsub 4x8 tiles; zero-pad
                // missing rows. Also accumulate the per-row int8 block sum.
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

                for (size_t c = 0; c < TN; ++c) {
                    if (c >= ncols) {
                        for (size_t s = 0; s < nsub; ++s)
                            std::memset(&Btile[s * TILE_BYTES + c * TK], 0, TK);
                        continue;
                    }
                    const uint8_t* bcol = Bp + (n + c) * b_col_bytes + kb * (BlkLen / 2);
                    for (size_t s = 0; s < nsub; ++s)
                        for (size_t k = 0; k < TK; ++k) {
                            const size_t i = s * TK + k;
                            const uint8_t byte = bcol[i >> 1];
                            const int8_t v = (i & 1) ? int8_t(byte >> 4) : int8_t(byte & 0x0F);
                            Btile[s * TILE_BYTES + c * TK + k] = v;
                        }
                }

                ime_tile_4x4(Atile.data(), Btile.data(),
                             static_cast<long>(nsub), tacc);

                for (size_t c = 0; c < ncols; ++c) {
                    const size_t nn = n + c;
                    int zp = 8;  // symmetric default (Q4_0)
                    if (QuantBZeroPoint) {
                        const std::byte* zpp =
                            QuantBZeroPoint + nn * zp_blk_bytes + (kb >> 1);
                        const int zb = static_cast<int>(std::to_integer<uint8_t>(*zpp));
                        zp = (kb & 1) ? (zb >> 4) : (zb & 0x0F);
                    }
                    const float bscale = QuantBScale[nn * BlockCountK + kb];
                    for (size_t r = 0; r < mrows; ++r) {
                        const int32_t centered = tacc[r * TN + c] - zp * asum[r];
                        acc[r][c] += as[r] * bscale * static_cast<float>(centered);
                    }
                }
            }

            for (size_t r = 0; r < mrows; ++r)
                for (size_t c = 0; c < ncols; ++c)
                    C[(m + r) * ldc + (n + c)] = acc[r][c];
        }
    }
    return CountM;
}

}  // namespace sqnbitgemm_ime

// ===========================================================================
// Dispatch object + platform accessor.
// ===========================================================================
const MLAS_QNBIT_GEMM_DISPATCH&
GetMlasQNBitGemmDispatchIme()
{
    static const MLAS_QNBIT_GEMM_DISPATCH dispatch = []() {
        MLAS_QNBIT_GEMM_DISPATCH d;
        d.Q4BitGemmPackQuantBDataSize = sqnbitgemm_ime::Q4BitGemmPackQuantBDataSize;
        d.SQ4BitGemmPackQuantBData = sqnbitgemm_ime::SQ4BitGemmPackQuantBData;
        d.QNBitGemmPerGemmWorkspaceSize = sqnbitgemm_ime::QNBitGemmPerGemmWorkspaceSize;
        d.QNBitGemmPerGemmWorkspaceAlignment = sqnbitgemm_ime::QNBitGemmPerGemmWorkspaceAlignment;
        d.SQ4BitGemmKernel_CompInt8 = sqnbitgemm_ime::SQ4BitGemmKernel_CompInt8;
        d.QuantizeARow_CompInt8 = sqnbitgemm_ime::QuantizeARow_CompInt8;
        return d;
    }();
    return dispatch;
}
