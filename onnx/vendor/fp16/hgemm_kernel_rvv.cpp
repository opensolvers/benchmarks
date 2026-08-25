/*++
  RISC-V RVV Zvfh MLAS_HGEMM_DISPATCH for attention / fp16 MatMul.
--*/

#include "halfgemm.h"
#include "mlasi.h"

#if defined(MLAS_TARGET_RISCV64) && defined(MLAS_USE_RVV_ZVFH)

#include <riscv_vector.h>
#include <cstring>

namespace hgemm_rvv {

MLAS_FORCEINLINE const _mlas_fp16_*
AsRaw(const MLAS_FP16* p)
{
    return reinterpret_cast<const _mlas_fp16_*>(p);
}

MLAS_FORCEINLINE _mlas_fp16_*
AsRaw(MLAS_FP16* p)
{
    return reinterpret_cast<_mlas_fp16_*>(p);
}

MLAS_FORCEINLINE vfloat16m4_t
VLoad(const _mlas_fp16_* p, size_t vl)
{
    return __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vle16_v_u16m4(p, vl));
}

MLAS_FORCEINLINE void
VStore(_mlas_fp16_* p, vfloat16m4_t v, size_t vl)
{
    __riscv_vse16_v_u16m4(p, __riscv_vreinterpret_v_f16m4_u16m4(v), vl);
}

void
ScaleC(_mlas_fp16_* C, size_t CountM, size_t CountN, size_t ldc, float scale)
{
    if (scale == 1.f) {
        return;
    }
    const _Float16 sf = static_cast<_Float16>(scale);
    for (size_t m = 0; m < CountM; ++m) {
        _mlas_fp16_* row = C + m * ldc;
        size_t n = 0;
        while (n < CountN) {
            size_t vl = __riscv_vsetvl_e16m4(CountN - n);
            vfloat16m4_t v = VLoad(row + n, vl);
            v = __riscv_vfmul_vf_f16m4(v, sf, vl);
            VStore(row + n, v, vl);
            n += vl;
        }
    }
}

void
HGemm_TransposedB_Kernel(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    MLAS_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    _mlas_fp16_ alpha,
    _mlas_fp16_ beta)
{
    const auto* a = AsRaw(A);
    const auto* b = AsRaw(B);
    auto* c = AsRaw(C);
    const float alpha_f = MLAS_Half2Float(alpha);
    const float beta_f = MLAS_Half2Float(beta);
    const _Float16 alpha_h = static_cast<_Float16>(alpha_f);

    for (size_t m = 0; m < CountM; ++m) {
        const _mlas_fp16_* a_row = a + m * lda;
        _mlas_fp16_* c_row = c + m * ldc;
        size_t n = 0;
        while (n < CountN) {
            size_t vl = __riscv_vsetvl_e16m4(CountN - n);
            vfloat16m4_t acc = beta_f == 0.f
                                   ? __riscv_vfmv_v_f_f16m4(static_cast<_Float16>(0.f), vl)
                                   : __riscv_vfmul_vf_f16m4(VLoad(c_row + n, vl), static_cast<_Float16>(beta_f), vl);
            for (size_t k = 0; k < CountK; ++k) {
                const float a_val = MLAS_Half2Float(a_row[k]);
                vfloat16m4_t bv = VLoad(b + k + (n)*ldb, vl);
                acc = __riscv_vfmacc_vf_f16m4(acc, static_cast<_Float16>(alpha_f * a_val), bv, vl);
            }
            VStore(c_row + n, acc, vl);
            n += vl;
        }
    }
    (void)alpha_h;
}

void
HGemm_B_Kernel(
    const MLAS_FP16* A,
    const MLAS_FP16* B,
    MLAS_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    _mlas_fp16_ alpha,
    _mlas_fp16_ beta)
{
    const auto* a = AsRaw(A);
    const auto* b = AsRaw(B);
    auto* c = AsRaw(C);
    const float alpha_f = MLAS_Half2Float(alpha);
    const float beta_f = MLAS_Half2Float(beta);
    const bool zero = beta_f == 0.f;

    for (size_t m = 0; m < CountM; ++m) {
        const _mlas_fp16_* a_row = a + m * lda;
        _mlas_fp16_* c_row = c + m * ldc;
        size_t n = 0;
        while (n < CountN) {
            size_t vl = __riscv_vsetvl_e16m4(CountN - n);
            vfloat16m4_t acc = zero
                                   ? __riscv_vfmv_v_f_f16m4(static_cast<_Float16>(0.f), vl)
                                   : __riscv_vfmul_vf_f16m4(VLoad(c_row + n, vl), static_cast<_Float16>(beta_f), vl);
            for (size_t k = 0; k < CountK; ++k) {
                const _Float16 a_val = static_cast<_Float16>(MLAS_Half2Float(a_row[k]) * alpha_f);
                vfloat16m4_t bv = VLoad(b + k * ldb + n, vl);
                acc = __riscv_vfmacc_vf_f16m4(acc, a_val, bv, vl);
            }
            VStore(c_row + n, acc, vl);
            n += vl;
        }
    }
}

void
HPackB_TransposedB_Kernel(
    const MLAS_FP16* B,
    MLAS_FP16* PackedB,
    size_t CountN,
    size_t CountK,
    size_t ldb)
{
    for (size_t n = 0; n < CountN; ++n) {
        for (size_t k = 0; k < CountK; ++k) {
            AsRaw(PackedB)[k + n * CountK] = AsRaw(B)[k + n * ldb];
        }
    }
}

void
HPackB_B_Kernel(
    const MLAS_FP16* B,
    MLAS_FP16* PackedB,
    size_t CountN,
    size_t CountK,
    size_t ldb)
{
    for (size_t k = 0; k < CountK; ++k) {
        memcpy(AsRaw(PackedB) + k * CountN, AsRaw(B) + k * ldb, CountN * sizeof(_mlas_fp16_));
    }
}

void
HGemm_PackedB_Kernel(
    const MLAS_FP16* A,
    const MLAS_FP16* PackedB,
    MLAS_FP16* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldc,
    _mlas_fp16_ alpha,
    _mlas_fp16_ beta)
{
    HGemm_B_Kernel(A, PackedB, C, CountM, CountN, CountK, lda, CountN, ldc, alpha, beta);
}

}  // namespace hgemm_rvv

const MLAS_HGEMM_DISPATCH MlasHGemmDispatchRvv = []() {
    MLAS_HGEMM_DISPATCH d{};
    d.HPackBKernel_TransposedB = hgemm_rvv::HPackB_TransposedB_Kernel;
    d.HPackBKernel_B = hgemm_rvv::HPackB_B_Kernel;
    d.HGemmKernel_TransposedB = hgemm_rvv::HGemm_TransposedB_Kernel;
    d.HGemmKernel_B = hgemm_rvv::HGemm_B_Kernel;
    d.HGemmKernel_PackedB = hgemm_rvv::HGemm_PackedB_Kernel;
    return d;
}();

#endif
