/*++

Module Name:
    sqnbitgemm_kernel_ime.cpp

Abstract:
    SQNBit (block-quantized n-bit) GEMM kernel for the SpaceMiT X60 RISC-V
    "IME" integer matrix extension, for MLAS.

    Computes  C (fp32) = A (int8, block-scaled) x B (4-bit or 8-bit, block-scaled)
    for ComputeType == SQNBIT_CompInt8. BlkBitWidth==4 matches llama.cpp spacemit
    `gemm_kernel_i8i4`; BlkBitWidth==8 is SQ8Bit CompInt8 (signed A × signed
    B'=B_u−128, zp folded via QuantBBlkSum). Microkernel from opensolvers
    `ime/` s8s8s32 / smt.vmadot.

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

    Perf notes (2026-08-24 / 08-25):
    - BlkLen=32: pack-time Q4_0×16 / Q8×16 panels; M=1 llama ScaleFp16; M≥4 gather from x16.
    - BlkLen=128 (Qwen AMD): standard column-major pack + RVV nibble gather + IME
      (same as pre-m1pack rvvgather). M1 driver only calls M1Kernel for BlkLen=32.

--*/

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#include "sqnbitgemm_ime_m1_panel.inc"
#include "sqnbitgemm_ime_quantize.inc"

namespace sqnbitgemm_ime
{

// Q4_0×16 panel layout (llama M1 / pack-time).
constexpr size_t kQ4x16NbCols = 16;
constexpr size_t kQ4x16KbBytes = 288;  // 32 scales + 256 weights
constexpr size_t kQ4x16ColBytes = 18;  // 288 / 16

// One smt.vmadot consumes a 4x8 int8 A-tile and a 4x8 int8 B-tile (B pre-
// transposed: b[col][k]) and updates a 4x4 int32 accumulator.
constexpr size_t TM = 4;   // rows per tile
constexpr size_t TN = 4;   // cols per tile
constexpr size_t TK = 8;   // K per tile
constexpr size_t TILE_BYTES = TM * TK;  // 32: one 4x8 int8 tile
// Largest BlkLen we stack-allocate for (MLAS QNBit: 16..256). 256/8 = 32 subs.
constexpr size_t MAX_NSUB = 32;
constexpr size_t MAX_TILE_BYTES = MAX_NSUB * TILE_BYTES;

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

// Gather one TN=4 B panel from pack-time Q4_0×16 weight tiles into Btile.
// Bp points at column 0 of the current N-chunk; scales are external (fp32).
static inline void
gather_b_panel_from_q4x16(
    const uint8_t* Bp,
    size_t n,
    size_t ncols,
    size_t kb,
    size_t BlockCountK,
    size_t nsub,
    int8_t* Btile)
{
    for (size_t c = 0; c < TN; ++c) {
        if (c >= ncols) {
            for (size_t s = 0; s < nsub; ++s) {
                std::memset(&Btile[s * TILE_BYTES + c * TK], 0, TK);
            }
            continue;
        }
        const size_t col = n + c;
        const size_t panel_col0 = col & ~(kQ4x16NbCols - 1);
        const size_t col_in_panel = col - panel_col0;
        const uint8_t* panel =
            Bp + panel_col0 * (BlockCountK * kQ4x16ColBytes);
        const uint8_t* weights = panel + kb * kQ4x16KbBytes + 32;
        const size_t g = col_in_panel / 4;
        const size_t cg = col_in_panel % 4;
        // two halves × (lo K0-7, hi K8-15) → four TK=8 Btile rows for BlkLen=32
        for (size_t half = 0; half < 2; ++half) {
            const uint8_t* src = weights + half * 128 + g * 32 + cg * 8;
            for (size_t k = 0; k < 8; ++k) {
                const uint8_t byte = src[k];
                Btile[(half * 2) * TILE_BYTES + c * TK + k] =
                    static_cast<int8_t>(byte & 0x0F);
                Btile[(half * 2 + 1) * TILE_BYTES + c * TK + k] =
                    static_cast<int8_t>(byte >> 4);
            }
        }
        (void)nsub;
    }
}

// Unpack packed 4-bit block → BlkLen int8 nibbles in [0..15] (BlkLen=128 path).
static inline void
unpack_nibbles_linear(const uint8_t* src, int8_t* dst, size_t BlkLen)
{
#if defined(__riscv_vector)
    const size_t packed = BlkLen / 2;
    size_t offset = 0;
    while (offset < packed) {
        size_t vl = __riscv_vsetvl_e8m1(packed - offset);
        vuint8m1_t bytes = __riscv_vle8_v_u8m1(src + offset, vl);
        vuint8m1_t lo = __riscv_vand_vx_u8m1(bytes, 0x0F, vl);
        vuint8m1_t hi = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(bytes, 4, vl), 0x0F, vl);
        __riscv_vsse8_v_i8m1(dst + offset * 2, 2, __riscv_vreinterpret_v_u8m1_i8m1(lo), vl);
        __riscv_vsse8_v_i8m1(dst + offset * 2 + 1, 2, __riscv_vreinterpret_v_u8m1_i8m1(hi), vl);
        offset += vl;
    }
#else
    for (size_t i = 0; i < BlkLen; ++i) {
        const uint8_t byte = src[i >> 1];
        dst[i] = static_cast<int8_t>((i & 1) ? (byte >> 4) : (byte & 0x0F));
    }
#endif
}

static inline void
gather_b_panel_column_major(
    const uint8_t* Bp,
    size_t n,
    size_t ncols,
    size_t kb,
    size_t BlkLen,
    size_t nsub,
    size_t b_col_bytes,
    int8_t* Btile)
{
    alignas(32) int8_t linear[256];
    for (size_t c = 0; c < TN; ++c) {
        if (c >= ncols) {
            for (size_t s = 0; s < nsub; ++s) {
                std::memset(&Btile[s * TILE_BYTES + c * TK], 0, TK);
            }
            continue;
        }
        unpack_nibbles_linear(Bp + (n + c) * b_col_bytes + kb * (BlkLen / 2), linear, BlkLen);
        for (size_t s = 0; s < nsub; ++s) {
            std::memcpy(&Btile[s * TILE_BYTES + c * TK], linear + s * TK, TK);
        }
    }
}

// ===========================================================================
// A quantization: fp32 row -> Q8 blocks, each [fp32 scale][BlkLen int8].
// Symmetric (zero-point-free) int8, matching MLAS's CompInt8 A format.
// ===========================================================================
void
QuantizeARow_CompInt8(size_t BlkLen, const float* A, size_t CountK, std::byte* QuantA)
{
    if (BlkLen == 32) {
        const float* src = A;
        std::byte* dst = QuantA;
        size_t k = CountK;
        constexpr float range_max_reciprocal = 1.0f / 127.0f;
        const float fone = 1.0f;
        __asm__ volatile(
            ".option push\n\t"
            ".option arch, +v\n\t"
            IME_QUANTIZE_A_ROW_BL32_ASM
            ".option pop\n\t"
            : [K] "+r"(k)
            : [SRC] "r"(src), [DST] "r"(dst), [FONE] "f"(fone),
              [RMAXREC] "f"(range_max_reciprocal)
            : "cc", "t3", "t2", "t1", "t0", "a1", "a2", "a3", "a4", "s1", "s2", "s3", "s4",
              "f10", "f11", "f12", "f13", "v0", "v4", "v8", "v12", "v16", "v17", "v18", "v20",
              "v21", "v22", "v24", "v25", "v26", "v28", "v29", "v30", "memory");
        return;
    }
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

// fp16 A-row → Q8 blocks (BlkLen=32 uses block-wise fp16→fp32 then RVV quantize).
void
QuantizeARow_CompInt8_FromFp16(size_t BlkLen, const MLAS_FP16* A, size_t CountK, std::byte* QuantA)
{
    if (BlkLen == 32) {
        alignas(32) float fbuf[32];
        std::byte* dst = QuantA;
        size_t k = CountK;
        while (k >= 32) {
            MlasConvertHalfToFloatBuffer(A, fbuf, 32);
            QuantizeARow_CompInt8(32, fbuf, 32, dst);
            A += 32;
            dst += Q8BlkSize(32);
            k -= 32;
        }
        if (k > 0) {
            // Zero-pad so RVV BlkLen=32 quantize never reads uninitialized tail.
            std::memset(fbuf, 0, sizeof(fbuf));
            MlasConvertHalfToFloatBuffer(A, fbuf, k);
            QuantizeARow_CompInt8(32, fbuf, 32, dst);
        }
        return;
    }
    std::byte* blk = QuantA;
    alignas(32) float fbuf[256];
    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t klen = std::min(BlkLen, CountK - k);
        MlasConvertHalfToFloatBuffer(A + k, fbuf, klen);
        float amax = 0.0f;
        for (size_t i = 0; i < klen; ++i) {
            const float v = std::fabs(fbuf[i]);
            if (v > amax) amax = v;
        }
        const float scale = amax / 127.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        Q8BlkScale(blk) = scale;
        int8_t* data = Q8BlkData(blk);
        for (size_t i = 0; i < BlkLen; ++i) {
            if (i < klen) {
                int q = static_cast<int>(std::lrintf(fbuf[i] * inv));
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                data[i] = static_cast<int8_t>(q);
            } else {
                data[i] = 0;
            }
        }
        blk += Q8BlkSize(BlkLen);
    }
}

// Pack column-major nibbles → x16 weight tiles (see kQ4x16* at top of namespace).
// Per-column byte stride ldb = BlockCountK * kQ4x16ColBytes.
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
    if (BlkLen == 32) {
        const size_t Npad = MlasDivRoundup(N, kQ4x16NbCols) * kQ4x16NbCols;
        return Npad / kQ4x16NbCols * BlockCountK * kQ4x16KbBytes;
    }
    // BlkLen=128 (Qwen) and other TK-aligned sizes: column-major nibbles.
    return N * BlockCountK * (BlkLen / 2);
}

size_t
Q4BitGemmPackedBColumnStrideBytes(
    size_t BlkBitWidth, size_t BlkLen, size_t BlockCountK,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*cfg*/)
{
    if (BlkBitWidth != 4 || (BlkLen % TK) != 0) {
        return 0;
    }
    if (BlkLen == 32) {
        return BlockCountK * kQ4x16ColBytes;
    }
    return BlockCountK * (BlkLen / 2);
}

// Pack column-major nibbles → x16 weight tiles inside each panel (scale slots
// left zero). Input: [n][kblk][BlkLen/2].
static void
pack_q4x16_weights(
    size_t N, size_t BlockCountK, const uint8_t* src, uint8_t* dst)
{
    const size_t Npad = MlasDivRoundup(N, kQ4x16NbCols) * kQ4x16NbCols;
    const size_t packed_per_col = 16;
    std::memset(dst, 0, Npad / kQ4x16NbCols * BlockCountK * kQ4x16KbBytes);

    for (size_t n0 = 0; n0 < N; n0 += kQ4x16NbCols) {
        const size_t ncols = std::min(kQ4x16NbCols, N - n0);
        uint8_t* panel = dst + (n0 / kQ4x16NbCols) * BlockCountK * kQ4x16KbBytes;
        for (size_t kb = 0; kb < BlockCountK; ++kb) {
            uint8_t* weights = panel + kb * kQ4x16KbBytes + 32;
            for (size_t g = 0; g < 4; ++g) {
                for (size_t half = 0; half < 2; ++half) {
                    uint8_t* tile = weights + half * 128 + g * 32;
                    for (size_t c = 0; c < 4; ++c) {
                        const size_t col = g * 4 + c;
                        if (col >= ncols) {
                            continue;
                        }
                        const uint8_t* scol =
                            src + ((n0 + col) * BlockCountK + kb) * packed_per_col + half * 8;
                        for (size_t k = 0; k < 8; ++k) {
                            const uint8_t b0 = scol[k >> 1];
                            const uint8_t b1 = scol[(k + 8) >> 1];
                            const int n0v = (k & 1) ? (b0 >> 4) : (b0 & 0x0F);
                            const int n1v = ((k + 8) & 1) ? (b1 >> 4) : (b1 & 0x0F);
                            tile[c * 8 + k] =
                                static_cast<uint8_t>((n0v & 0x0F) | ((n1v & 0x0F) << 4));
                        }
                    }
                }
            }
        }
    }
}

static void
pack_q4x16_scales(size_t N, size_t BlockCountK, const float* scales, uint8_t* dst)
{
    for (size_t n0 = 0; n0 < N; n0 += kQ4x16NbCols) {
        const size_t ncols = std::min(kQ4x16NbCols, N - n0);
        uint8_t* panel = dst + (n0 / kQ4x16NbCols) * BlockCountK * kQ4x16KbBytes;
        for (size_t kb = 0; kb < BlockCountK; ++kb) {
            auto* fp16 = reinterpret_cast<_Float16*>(panel + kb * kQ4x16KbBytes);
            for (size_t c = 0; c < ncols; ++c) {
                fp16[c] = static_cast<_Float16>(scales[(n0 + c) * BlockCountK + kb]);
            }
        }
    }
}

// AndBlkSum hook: receives QuantBData and/or QuantBScale (ORT PrePack may call
// twice). Writes Q4_0×16 into the start of the workspace. BlkSum unused.
void
SQ4BitGemmPackQuantBDataAndBlkSum(
    size_t N, size_t K, size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool /*HasZeroPoint*/,
    const std::byte* /*QuantBZPBegin*/,
    PackedQuantBDataStruct<float, 4>& PackedQuantB,
    MLAS_THREADPOOL* /*ThreadPool*/,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*cfg*/)
{
    assert((BlkLen % TK) == 0);
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    uint8_t* dst = reinterpret_cast<uint8_t*>(PackedQuantB.PackedQuantBData);
    if (BlkLen == 32) {
        if (QuantBDataBegin != nullptr) {
            pack_q4x16_weights(
                N, BlockCountK, reinterpret_cast<const uint8_t*>(QuantBDataBegin), dst);
        }
        if (QuantBScaleBegin != nullptr) {
            pack_q4x16_scales(N, BlockCountK, QuantBScaleBegin, dst);
        }
        return;
    }
    // Column-major: copy weights; scales stay external (QuantBScale).
    if (QuantBDataBegin != nullptr) {
        std::memcpy(dst, QuantBDataBegin, N * BlockCountK * (BlkLen / 2));
    }
}

void
SQ4BitGemmPackQuantBData(
    size_t N, size_t K, size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const std::byte* QuantBDataBegin, std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* /*ThreadPool*/,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*cfg*/)
{
    assert((BlkLen % TK) == 0);
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    if (BlkLen == 32) {
        pack_q4x16_weights(
            N, BlockCountK, reinterpret_cast<const uint8_t*>(QuantBDataBegin),
            reinterpret_cast<uint8_t*>(PackedQuantBDataBegin));
        return;
    }
    std::memcpy(PackedQuantBDataBegin, QuantBDataBegin, N * BlockCountK * (BlkLen / 2));
}

// ===========================================================================
// Workspace: MLAS quantizes A into this scratch (Q8 blocks) before the kernel.
// ===========================================================================
size_t
QNBitGemmPerGemmWorkspaceSize(
    size_t M, size_t /*N*/, size_t K, size_t BlkLen, bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType, size_t BlkBitWidth,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*cfg*/)
{
    if (ComputeType != SQNBIT_CompInt8) {
        return 0;
    }
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    if (BlkBitWidth == 8) {
        // W8: flat QuantData (M×BC×BlkLen) + QuantScale + BlockSum
        // (= Q8BlkSize + sizeof(float) per block — same as NEON).
        return M * BlockCountK * (Q8BlkSize(BlkLen) + sizeof(float));
    }
    // W4: interleaved Q8 blocks [scale|int8×BlkLen]
    return M * BlockCountK * Q8BlkSize(BlkLen);
}

size_t
QNBitGemmPerGemmWorkspaceAlignment(size_t /*BlkLen*/, MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/)
{
    return Q8BlkAlignment();
}

// ---------------------------------------------------------------------------
// llama M1 ScaleFp16 no-ZP microkernel (one N≤16 panel).
// ---------------------------------------------------------------------------
static void
gemm_m1_panel_llama(
    const std::byte* QuantA,
    const std::byte* QuantBDataPtr,
    float* CPtr,
    size_t BlockCountK,
    size_t nblks)
{
    const size_t INNER = 2;
    size_t cnt = BlockCountK;
    __asm__ volatile(
        ".option push\n\t"
        ".option arch, +xsmtvdot\n\t"
        "vsetvli      t0, zero, e32, m4       \n\t"
        "vxor.vv      v28, v28, v28           \n\t"
        "addi         s1, %[B], 0             \n\t"
        "addi         s2, %[B], 8             \n\t"
        "addi         s3, %[B], 16            \n\t"
        "addi         s4, %[B], 24            \n\t"
        "addi         s5, %[A], 0             \n\t"
        "addi         s6, %[A], 12            \n\t"
        "LOOP_K%=:                            \n\t"
        "vsetvli      t0, zero, e16, mf4      \n\t"
        "vle16.v      v4, (s1)                \n\t"
        "addi         s1, s1, 32              \n\t"
        "vle16.v      v5, (s2)                \n\t"
        "addi         s2, s2, 56              \n\t"
        "vle16.v      v6, (s3)                \n\t"
        "addi         s3, s3, 80              \n\t"
        "vle16.v      v7, (s4)                \n\t"
        "addi         s4, s4, 104             \n\t"
        "flw          f1, (s5)                \n\t"
        "addi         s5, s5, 4               \n\t"
        "vfwcvt.f.f.v v8, v4                  \n\t"
        "vfwcvt.f.f.v v9, v5                  \n\t"
        "vfwcvt.f.f.v v10, v6                 \n\t"
        "vfwcvt.f.f.v v11, v7                 \n\t"
        "vsetvli      t0, zero, e32, mf2      \n\t"
        "addi         t5, %[INNER], 0         \n\t"
        "vxor.vv      v16, v16, v16           \n\t"
        "vxor.vv      v18, v18, v18           \n\t"
        "vxor.vv      v20, v20, v20           \n\t"
        "vxor.vv      v22, v22, v22           \n\t"
        "vfmul.vf     v24, v8, f1             \n\t"
        "vfmul.vf     v25, v9, f1             \n\t"
        "vfmul.vf     v26, v10, f1            \n\t"
        "vfmul.vf     v27, v11, f1            \n\t"
        "addi         %[CNT], %[CNT], -1      \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "LOOP_INNER%=:                        \n\t"
        SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4
        "vadd.vi      v0, v0, -8              \n\t"
        "vadd.vi      v1, v1, -8              \n\t"
        "vadd.vi      v2, v2, -8              \n\t"
        "vadd.vi      v3, v3, -8              \n\t"
        "vadd.vi      v4, v4, -8              \n\t"
        "vadd.vi      v5, v5, -8              \n\t"
        "vadd.vi      v6, v6, -8              \n\t"
        "vadd.vi      v7, v7, -8              \n\t"
        SQ4BIT_KERNEL_COMP_1x8x2_4X8X4
        "bnez         t5, LOOP_INNER%=        \n\t"
        "vsetvli      t0, zero, e32, mf2      \n\t"
        SQ4BIT_KERNEL_ACC_F16_1X4X4
        "bnez         %[CNT], LOOP_K%=        \n\t"
        "addi         t3, zero, 16            \n\t"
        "addi         s1, %[C], 16            \n\t"
        "addi         s2, %[C], 32            \n\t"
        "addi         s3, %[C], 48            \n\t"
        "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        "vse32.v      v29, (s1)               \n\t"
        "vse32.v      v30, (s2)               \n\t"
        "vse32.v      v31, (s3)               \n\t"
        "jal          x0, END%=               \n\t"
        "ST_TAIL%=:                           \n\t"
        "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
        "sub          %[NBLKS], %[NBLKS], t0  \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
        "sub          %[NBLKS], %[NBLKS], t0  \n\t"
        "vse32.v      v29, (s1)               \n\t"
        "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
        "sub          %[NBLKS], %[NBLKS], t0  \n\t"
        "vse32.v      v30, (s2)               \n\t"
        "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
        "sub          %[NBLKS], %[NBLKS], t0  \n\t"
        "vse32.v      v31, (s3)               \n\t"
        "END%=:                               \n\t"
        ".option pop\n\t"
        : [CNT] "+r"(cnt), [NBLKS] "+r"(nblks)
        : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [C] "r"(CPtr)
        : "cc", "t0", "t5", "t3", "f1", "s1", "s2", "s3", "s4", "s5", "s6",
          "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
          "v0", "v1", "v2", "v3", "v14", "v15", "v16", "v18", "v20", "v22",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory");
}

void
SQ4BitGemmM1Kernel_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const std::byte* QuantBData,
    const float* /*QuantBScale*/,
    const std::byte* /*QuantBZeroPoint*/,
    float* C,
    size_t CountN,
    size_t /*CountK*/,
    size_t BlockCountK,
    size_t /*ldc*/,
    const float* Bias)
{
    assert(BlkLen == 32);
    MLAS_UNREFERENCED_PARAMETER(BlkLen);
    for (size_t n = 0; n < CountN; n += kQ4x16NbCols) {
        const size_t ncols = std::min(kQ4x16NbCols, CountN - n);
        const std::byte* panel = QuantBData + n * (BlockCountK * kQ4x16ColBytes);
        float* cptr = C + n;
        if (Bias) {
            alignas(16) float tmp[16] = {};
            gemm_m1_panel_llama(QuantA, panel, tmp, BlockCountK, ncols);
            for (size_t c = 0; c < ncols; ++c) {
                cptr[c] = Bias[n + c] + tmp[c];
            }
        } else {
            gemm_m1_panel_llama(QuantA, panel, cptr, BlockCountK, ncols);
        }
    }
}

// ===========================================================================
// The kernel.  C[m][n] = Bias[n] + sum_kb Ascale[m][kb]*Bscale[n][kb] *
//     ( sum_i Aq[m][kb][i] * (Bq[n][kb][i] - zp[n][kb]) ).
// Full 4x4 tiles use smt.vmadot; row/col remainders are handled by zero-padded
// tiles so no scalar fallback path is needed.
//
// Loop order: for each M-panel, init all N accs, then for each K-block gather
// A once and sweep N (was: gather A inside the N loop → O(N/TN) redundant work).
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
    assert((BlkLen % TK) == 0);
    assert((BlkLen / TK) <= MAX_NSUB);

    const size_t nsub = BlkLen / TK;
    const size_t qa_blk = Q8BlkSize(BlkLen);
    const size_t qa_row = BlockCountK * qa_blk;
    const uint8_t* Bp = reinterpret_cast<const uint8_t*>(QuantBData);
    const size_t zp_blk_bytes = MlasDivRoundup(BlockCountK, size_t{2});
    const size_t b_col_bytes = BlockCountK * (BlkLen / 2);
    const bool use_q4x16 = (BlkLen == 32);

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

                if (use_q4x16) {
                    gather_b_panel_from_q4x16(
                        Bp, n, ncols, kb, BlockCountK, nsub, Btile);
                } else {
                    gather_b_panel_column_major(
                        Bp, n, ncols, kb, BlkLen, nsub, b_col_bytes, Btile);
                }

                ime_tile_4x4(Atile, Btile, static_cast<long>(nsub), tacc);

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
                        C[(m + r) * ldc + nn] +=
                            as[r] * bscale * static_cast<float>(centered);
                    }
                }
            }
        }
    }
    return CountM;
}

// ===========================================================================
// SQ8Bit CompInt8: signed A × signed B'=(B_u−128).
// BlkLen=32: pack-time Q8×16 panels (llama i8i8 layout) + M1 ScaleFp16 asm;
//   M≥4 gathers from panels. BlkSum/scales follow the panel slab.
// Other BlkLen: column-major signed int8 + flat scales (gather path).
// ===========================================================================
constexpr size_t kQ8x16NbCols = 16;

size_t
Q8BitGemmPackQuantBDataSize(
    size_t N, size_t K, size_t BlkLen, bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*cfg*/)
{
    if (ComputeType != SQNBIT_CompInt8 || (BlkLen % TK) != 0) {
        return 0;
    }
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    size_t PackedQuantBDataSize = MlasSQ8PackedBDataBytes(N, BlockCountK, BlkLen);
    const size_t ScaleSize = N * BlockCountK * sizeof(float);
    size_t BlkSumSize = MlasDivRoundup(N, size_t{16}) * BlockCountK * 16 * sizeof(float);
    constexpr size_t PackedQuantBDataAlignment = 32;
    PackedQuantBDataSize += PackedQuantBDataAlignment - 1;
    constexpr size_t BlkSumAlignment = MlasQNBitQuantBBlkSumAlignment();
    BlkSumSize += BlkSumAlignment - 1;
    return PackedQuantBDataSize + ScaleSize + BlkSumSize;
}

static void
q8_resolve_packed(
    PackedQuantBDataStruct<float, 8>& PackedQuantB,
    size_t N, size_t BlockCountK, size_t BlkLen,
    std::byte*& data, float*& blksum, float*& scale)
{
    float* unused_corr = nullptr;
    MlasSQ8ResolvePackedQuantB(
        PackedQuantB.QuantBWorkspace_, N, BlockCountK, BlkLen,
        /*QuantAUnsigned=*/false, data, blksum, scale, unused_corr);
}

static void
q8_pack_weights_signed_column_major(
    size_t N, size_t BlockCountK, size_t BlkLen,
    const uint8_t* src, int8_t* dst)
{
    const size_t col_bytes = BlockCountK * BlkLen;
    for (size_t n = 0; n < N; ++n) {
        const uint8_t* scol = src + n * col_bytes;
        int8_t* dcol = dst + n * col_bytes;
        for (size_t i = 0; i < col_bytes; ++i) {
            dcol[i] = static_cast<int8_t>(static_cast<int>(scol[i]) - 128);
        }
    }
}

// Column-major uint8 [N][BC][BlkLen] → Q8×16 panels (signed = u−128).
// Per kb: [16×fp16 scale][SLICES×4 tiles of 4cols×8k as [col][k]].
static void
pack_q8x16_weights(
    size_t N, size_t BlockCountK, size_t BlkLen,
    const uint8_t* src, uint8_t* dst)
{
    const size_t Npad = MlasDivRoundup(N, kQ8x16NbCols) * kQ8x16NbCols;
    const size_t kb_bytes = MlasSQ8x16KbBytes(BlkLen);
    const size_t nsub = BlkLen / TK;
    std::memset(dst, 0, Npad / kQ8x16NbCols * BlockCountK * kb_bytes);

    for (size_t n0 = 0; n0 < N; n0 += kQ8x16NbCols) {
        const size_t ncols = std::min(kQ8x16NbCols, N - n0);
        uint8_t* panel = dst + (n0 / kQ8x16NbCols) * BlockCountK * kb_bytes;
        for (size_t kb = 0; kb < BlockCountK; ++kb) {
            int8_t* weights = reinterpret_cast<int8_t*>(panel + kb * kb_bytes + 32);
            for (size_t g = 0; g < 4; ++g) {
                for (size_t s = 0; s < nsub; ++s) {
                    int8_t* tile = weights + (s * 4 + g) * 32;
                    for (size_t c = 0; c < 4; ++c) {
                        const size_t col = g * 4 + c;
                        if (col >= ncols) {
                            continue;
                        }
                        const uint8_t* scol =
                            src + ((n0 + col) * BlockCountK + kb) * BlkLen + s * TK;
                        for (size_t k = 0; k < TK; ++k) {
                            tile[c * TK + k] = static_cast<int8_t>(
                                static_cast<int>(scol[k]) - 128);
                        }
                    }
                }
            }
        }
    }
}

static void
pack_q8x16_scales(
    size_t N, size_t BlockCountK, size_t BlkLen,
    const float* scales, uint8_t* dst)
{
    const size_t kb_bytes = MlasSQ8x16KbBytes(BlkLen);
    for (size_t n0 = 0; n0 < N; n0 += kQ8x16NbCols) {
        const size_t ncols = std::min(kQ8x16NbCols, N - n0);
        uint8_t* panel = dst + (n0 / kQ8x16NbCols) * BlockCountK * kb_bytes;
        for (size_t kb = 0; kb < BlockCountK; ++kb) {
            auto* fp16 = reinterpret_cast<_Float16*>(panel + kb * kb_bytes);
            for (size_t c = 0; c < ncols; ++c) {
                fp16[c] = static_cast<_Float16>(scales[(n0 + c) * BlockCountK + kb]);
            }
        }
    }
}

static void
q8_compute_pack_blksum_signed(
    size_t N, size_t K, size_t BlkLen,
    float* QuantBScaleBegin,
    const std::byte* QuantBZPBegin,
    float* BlockSumBegin)
{
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkSumFloats = MlasDivRoundup(N, size_t{16}) * BlockCountK * 16;
    std::memset(BlockSumBegin, 0, BlkSumFloats * sizeof(float));

    for (size_t n = 0; n < N; ++n) {
        for (size_t kb = 0; kb < BlockCountK; ++kb) {
            const size_t src_off = n * BlockCountK + kb;
            const float scale = QuantBScaleBegin[src_off];
            int zp = 128;
            if (QuantBZPBegin) {
                zp = static_cast<int>(
                    std::to_integer<uint8_t>(QuantBZPBegin[src_off]));
            }
            const size_t dst =
                ((n / 16) * BlockCountK + kb) * 16 + (n % 16);
            BlockSumBegin[dst] = -scale * static_cast<float>(zp - 128);
        }
    }
}

void
SQ8BitGemmPackQuantBDataAndBlkSum(
    size_t N, size_t K, size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const std::byte* QuantBDataBegin,
    const float* QuantBScaleBegin,
    bool HasZeroPoint,
    const std::byte* QuantBZPBegin,
    PackedQuantBDataStruct<float, 8>& PackedQuantB,
    MLAS_THREADPOOL* /*ThreadPool*/,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*cfg*/)
{
    assert((BlkLen % TK) == 0);
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    std::byte* data = nullptr;
    float* blksum = nullptr;
    float* scale = nullptr;
    q8_resolve_packed(PackedQuantB, N, BlockCountK, BlkLen, data, blksum, scale);

    if (QuantBDataBegin != nullptr) {
        if (BlkLen == 32) {
            pack_q8x16_weights(
                N, BlockCountK, BlkLen,
                reinterpret_cast<const uint8_t*>(QuantBDataBegin),
                reinterpret_cast<uint8_t*>(data));
        } else {
            q8_pack_weights_signed_column_major(
                N, BlockCountK, BlkLen,
                reinterpret_cast<const uint8_t*>(QuantBDataBegin),
                reinterpret_cast<int8_t*>(data));
        }
        return;
    }

    if (QuantBScaleBegin != nullptr) {
        std::memcpy(scale, QuantBScaleBegin, N * BlockCountK * sizeof(float));
        if (BlkLen == 32) {
            pack_q8x16_scales(
                N, BlockCountK, BlkLen, QuantBScaleBegin,
                reinterpret_cast<uint8_t*>(data));
        }
    }
    if ((QuantBScaleBegin != nullptr && !HasZeroPoint) || QuantBZPBegin != nullptr) {
        q8_compute_pack_blksum_signed(
            N, K, BlkLen, scale, QuantBZPBegin, blksum);
    }
}

void
QuantizeARowComputeBlkSum_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* AScaledGroupSum)
{
    int8_t* blob = reinterpret_cast<int8_t*>(QuantA);
    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t klen = std::min(BlkLen, CountK - k);
        float amax = 0.0f;
        for (size_t i = 0; i < klen; ++i) {
            const float v = std::fabs(A[k + i]);
            if (v > amax) amax = v;
        }
        const float scale = amax / 127.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        *QuantAScale++ = scale;

        int32_t qsum = 0;
        for (size_t i = 0; i < BlkLen; ++i) {
            if (i < klen) {
                int q = static_cast<int>(std::lrintf(A[k + i] * inv));
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                blob[k + i] = static_cast<int8_t>(q);
                qsum += q;
            } else {
                blob[k + i] = 0;
            }
        }
        *AScaledGroupSum++ = scale * static_cast<float>(qsum);
    }
}

static inline void
gather_b8_panel_from_q8x16(
    const int8_t* Bp,
    size_t n,
    size_t ncols,
    size_t kb,
    size_t BlockCountK,
    size_t BlkLen,
    size_t nsub,
    int8_t* Btile)
{
    const size_t kb_bytes = MlasSQ8x16KbBytes(BlkLen);
    for (size_t c = 0; c < TN; ++c) {
        if (c >= ncols) {
            for (size_t s = 0; s < nsub; ++s) {
                std::memset(&Btile[s * TILE_BYTES + c * TK], 0, TK);
            }
            continue;
        }
        const size_t col = n + c;
        const size_t panel_col0 = col & ~(kQ8x16NbCols - 1);
        const size_t col_in_panel = col - panel_col0;
        const int8_t* panel = reinterpret_cast<const int8_t*>(
            reinterpret_cast<const uint8_t*>(Bp) +
            panel_col0 * (BlockCountK * MlasSQ8x16ColBytes(BlkLen)));
        const int8_t* weights =
            reinterpret_cast<const int8_t*>(
                reinterpret_cast<const uint8_t*>(panel) + kb * kb_bytes + 32);
        const size_t g = col_in_panel / 4;
        const size_t cc = col_in_panel % 4;
        for (size_t s = 0; s < nsub; ++s) {
            const int8_t* tile = weights + (s * 4 + g) * 32;
            std::memcpy(&Btile[s * TILE_BYTES + c * TK], tile + cc * TK, TK);
        }
    }
}

static inline void
gather_b8_panel_column_major(
    const int8_t* Bp,
    size_t n,
    size_t ncols,
    size_t kb,
    size_t BlkLen,
    size_t nsub,
    size_t b_col_bytes,
    int8_t* Btile)
{
    for (size_t c = 0; c < TN; ++c) {
        if (c >= ncols) {
            for (size_t s = 0; s < nsub; ++s) {
                std::memset(&Btile[s * TILE_BYTES + c * TK], 0, TK);
            }
            continue;
        }
        const int8_t* src = Bp + (n + c) * b_col_bytes + kb * BlkLen;
        for (size_t s = 0; s < nsub; ++s) {
            std::memcpy(&Btile[s * TILE_BYTES + c * TK], src + s * TK, TK);
        }
    }
}

// M1 Q8×16 panel microkernel (BlkLen=32). A is interleaved Q8Blk [fp32|int8×32].
static void
gemm_m1_panel_q8x16(
    const std::byte* QuantA,
    const std::byte* QuantBDataPtr,
    float* CPtr,
    size_t BlockCountK,
    size_t nblks)
{
    const size_t INNER = 2;  // BlkLen/16
    size_t cnt = BlockCountK;
    __asm__ volatile(
        ".option push\n\t"
        ".option arch, +xsmtvdot\n\t"
        "vsetvli      t0, zero, e32, m4       \n\t"
        "vxor.vv      v28, v28, v28           \n\t"
        "addi         s1, %[B], 0             \n\t"
        "addi         s5, %[A], 0             \n\t"
        "addi         s6, %[A], 12            \n\t"
        "LOOP_K%=:                            \n\t"
        "vsetvli      t0, zero, e16, mf4      \n\t"
        "addi         s2, s1, 8               \n\t"
        "addi         s3, s1, 16              \n\t"
        "addi         s4, s1, 24              \n\t"
        "vle16.v      v4, (s1)                \n\t"
        "vle16.v      v5, (s2)                \n\t"
        "vle16.v      v6, (s3)                \n\t"
        "vle16.v      v7, (s4)                \n\t"
        "addi         s1, s1, 32              \n\t"
        "flw          f1, (s5)                \n\t"
        "addi         s5, s5, 4               \n\t"
        "vfwcvt.f.f.v v8, v4                  \n\t"
        "vfwcvt.f.f.v v9, v5                  \n\t"
        "vfwcvt.f.f.v v10, v6                 \n\t"
        "vfwcvt.f.f.v v11, v7                 \n\t"
        "vsetvli      t0, zero, e32, mf2      \n\t"
        "addi         t5, %[INNER], 0         \n\t"
        "vxor.vv      v16, v16, v16           \n\t"
        "vxor.vv      v18, v18, v18           \n\t"
        "vxor.vv      v20, v20, v20           \n\t"
        "vxor.vv      v22, v22, v22           \n\t"
        "vfmul.vf     v24, v8, f1             \n\t"
        "vfmul.vf     v25, v9, f1             \n\t"
        "vfmul.vf     v26, v10, f1            \n\t"
        "vfmul.vf     v27, v11, f1            \n\t"
        "addi         %[CNT], %[CNT], -1      \n\t"
        "LOOP_INNER%=:                        \n\t"
        SQ8BIT_KERNEL_LOAD_B_16x8x2_I8
        SQ8BIT_KERNEL_LOAD_A_1x8x2
        SQ8BIT_KERNEL_COMP_1x8x2_4X8X4
        "bnez         t5, LOOP_INNER%=        \n\t"
        "vsetvli      t0, zero, e32, mf2      \n\t"
        "vfcvt.f.x.v  v16, v16                \n\t"
        "vfcvt.f.x.v  v18, v18                \n\t"
        "vfcvt.f.x.v  v20, v20                \n\t"
        "vfcvt.f.x.v  v22, v22                \n\t"
        "vfmacc.vv    v28, v16, v24           \n\t"
        "vfmacc.vv    v29, v18, v25           \n\t"
        "vfmacc.vv    v30, v20, v26           \n\t"
        "vfmacc.vv    v31, v22, v27           \n\t"
        "addi         s6, s5, 12              \n\t"
        "bnez         %[CNT], LOOP_K%=        \n\t"
        "addi         t3, zero, 16            \n\t"
        "addi         s1, %[C], 16            \n\t"
        "addi         s2, %[C], 32            \n\t"
        "addi         s3, %[C], 48            \n\t"
        "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        "vse32.v      v29, (s1)               \n\t"
        "vse32.v      v30, (s2)               \n\t"
        "vse32.v      v31, (s3)               \n\t"
        "jal          x0, END%=               \n\t"
        "ST_TAIL%=:                           \n\t"
        "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
        "sub          %[NBLKS], %[NBLKS], t0  \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
        "sub          %[NBLKS], %[NBLKS], t0  \n\t"
        "vse32.v      v29, (s1)               \n\t"
        "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
        "sub          %[NBLKS], %[NBLKS], t0  \n\t"
        "vse32.v      v30, (s2)               \n\t"
        "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
        "sub          %[NBLKS], %[NBLKS], t0  \n\t"
        "vse32.v      v31, (s3)               \n\t"
        "END%=:                               \n\t"
        ".option pop\n\t"
        : [CNT] "+r"(cnt), [NBLKS] "+r"(nblks)
        : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [C] "r"(CPtr)
        : "cc", "t0", "t5", "t3", "f1", "s1", "s2", "s3", "s4", "s5", "s6",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v14", "v15",
          "v16", "v18", "v20", "v22", "v24", "v25", "v26", "v27",
          "v28", "v29", "v30", "v31", "memory");
}

static void
sq8_m1_add_blksum(
    float* C, size_t CountN, size_t BlockCountK,
    const float* ABlockSum, const float* QuantBBlkSum)
{
    for (size_t nn = 0; nn < CountN; ++nn) {
        float acc = 0.0f;
        for (size_t kb = 0; kb < BlockCountK; ++kb) {
            const size_t boff = ((nn / 16) * BlockCountK + kb) * 16 + (nn % 16);
            acc += ABlockSum[kb] * QuantBBlkSum[boff];
        }
        C[nn] += acc;
    }
}

static void
sq8_m1_kernel_q8x16(
    const std::byte* QuantAFlat,
    const float* QuantAScale,
    const std::byte* QuantBData,
    float* C,
    size_t CountN,
    size_t BlockCountK,
    const float* Bias,
    const float* ABlockSum,
    const float* QuantBBlkSum)
{
    // Interleave flat A → Q8Blk for the ScaleFp16 microkernel.
    alignas(32) std::byte a_int[256 * Q8BlkSize(32)];  // BC≤256
    assert(BlockCountK <= 256);
    for (size_t kb = 0; kb < BlockCountK; ++kb) {
        std::byte* blk = a_int + kb * Q8BlkSize(32);
        Q8BlkScale(blk) = QuantAScale[kb];
        std::memcpy(Q8BlkData(blk), QuantAFlat + kb * 32, 32);
    }

    const size_t col_stride = BlockCountK * MlasSQ8x16ColBytes(32);
    for (size_t n = 0; n < CountN; n += kQ8x16NbCols) {
        const size_t ncols = std::min(kQ8x16NbCols, CountN - n);
        const std::byte* panel = QuantBData + n * col_stride;
        float* cptr = C + n;
        if (Bias) {
            alignas(16) float tmp[16] = {};
            gemm_m1_panel_q8x16(a_int, panel, tmp, BlockCountK, ncols);
            for (size_t c = 0; c < ncols; ++c) {
                cptr[c] = Bias[n + c] + tmp[c];
            }
        } else {
            gemm_m1_panel_q8x16(a_int, panel, cptr, BlockCountK, ncols);
        }
    }
    if (ABlockSum && QuantBBlkSum) {
        sq8_m1_add_blksum(C, CountN, BlockCountK, ABlockSum, QuantBBlkSum);
    }
}

size_t
SQ8BitGemmKernel_BlkSum_CompInt8(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* /*QuantBZeroPoint*/,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t /*CountK*/,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum,
    const float* /*BlkUnsignedQuantAZeroPointCorrection*/)
{
    assert((BlkLen % TK) == 0);
    assert((BlkLen / TK) <= MAX_NSUB);

    if (CountM == 1 && BlkLen == 32) {
        sq8_m1_kernel_q8x16(
            QuantA, QuantAScale, QuantBData, C, CountN, BlockCountK,
            Bias, ABlockSum, QuantBBlkSum);
        return 1;
    }

    const size_t nsub = BlkLen / TK;
    const size_t qa_row = BlockCountK * BlkLen;
    const int8_t* Ap = reinterpret_cast<const int8_t*>(QuantA);
    const int8_t* Bp = reinterpret_cast<const int8_t*>(QuantBData);
    const size_t b_col_bytes =
        (BlkLen == 32) ? (BlockCountK * MlasSQ8x16ColBytes(BlkLen))
                       : (BlockCountK * BlkLen);
    const bool use_q8x16 = (BlkLen == 32);

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
            for (size_t r = 0; r < TM; ++r) {
                if (r >= mrows) {
                    for (size_t s = 0; s < nsub; ++s) {
                        std::memset(&Atile[s * TILE_BYTES + r * TK], 0, TK);
                    }
                    continue;
                }
                as[r] = QuantAScale[(m + r) * BlockCountK + kb];
                const int8_t* ad = Ap + (m + r) * qa_row + kb * BlkLen;
                for (size_t s = 0; s < nsub; ++s) {
                    std::memcpy(&Atile[s * TILE_BYTES + r * TK], ad + s * TK, TK);
                }
            }

            for (size_t n = 0; n < CountN; n += TN) {
                const size_t ncols = std::min(TN, CountN - n);
                if (use_q8x16) {
                    gather_b8_panel_from_q8x16(
                        Bp, n, ncols, kb, BlockCountK, BlkLen, nsub, Btile);
                } else {
                    gather_b8_panel_column_major(
                        Bp, n, ncols, kb, BlkLen, nsub, b_col_bytes, Btile);
                }
                ime_tile_4x4(Atile, Btile, static_cast<long>(nsub), tacc);

                for (size_t c = 0; c < ncols; ++c) {
                    const size_t nn = n + c;
                    const float bscale = QuantBScale[nn * BlockCountK + kb];
                    for (size_t r = 0; r < mrows; ++r) {
                        C[(m + r) * ldc + nn] +=
                            as[r] * bscale *
                            static_cast<float>(tacc[r * TN + c]);
                    }
                }
            }
        }

        for (size_t r = 0; r < mrows; ++r) {
            const float* a_sum = ABlockSum + (m + r) * BlockCountK;
            float* crow = C + (m + r) * ldc;
            for (size_t nn = 0; nn < CountN; ++nn) {
                float acc = 0.0f;
                for (size_t kb = 0; kb < BlockCountK; ++kb) {
                    const size_t boff =
                        ((nn / 16) * BlockCountK + kb) * 16 + (nn % 16);
                    acc += a_sum[kb] * QuantBBlkSum[boff];
                }
                crow[nn] += acc;
            }
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
        d.Q4BitGemmPackedBColumnStrideBytes = sqnbitgemm_ime::Q4BitGemmPackedBColumnStrideBytes;
        d.SQ4BitGemmPackQuantBData = sqnbitgemm_ime::SQ4BitGemmPackQuantBData;
        d.SQ4BitGemmPackQuantBDataAndBlkSum = sqnbitgemm_ime::SQ4BitGemmPackQuantBDataAndBlkSum;
        d.Q8BitGemmPackQuantBDataSize = sqnbitgemm_ime::Q8BitGemmPackQuantBDataSize;
        d.SQ8BitGemmPackQuantBDataAndBlkSum = sqnbitgemm_ime::SQ8BitGemmPackQuantBDataAndBlkSum;
        d.QNBitGemmPerGemmWorkspaceSize = sqnbitgemm_ime::QNBitGemmPerGemmWorkspaceSize;
        d.QNBitGemmPerGemmWorkspaceAlignment = sqnbitgemm_ime::QNBitGemmPerGemmWorkspaceAlignment;
        d.SQ4BitGemmKernel_CompInt8 = sqnbitgemm_ime::SQ4BitGemmKernel_CompInt8;
        d.SQ4BitGemmM1Kernel_CompInt8 = sqnbitgemm_ime::SQ4BitGemmM1Kernel_CompInt8;
        d.SQ8BitGemmKernel_BlkSum_CompInt8 = sqnbitgemm_ime::SQ8BitGemmKernel_BlkSum_CompInt8;
        d.QuantizeARow_CompInt8 = sqnbitgemm_ime::QuantizeARow_CompInt8;
        d.QuantizeARow_CompInt8_FromFp16 = sqnbitgemm_ime::QuantizeARow_CompInt8_FromFp16;
        d.QuantizeARowComputeBlkSum_CompInt8 = sqnbitgemm_ime::QuantizeARowComputeBlkSum_CompInt8;
        return d;
    }();
    return dispatch;
}
