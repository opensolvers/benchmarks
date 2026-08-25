/*++
  RISC-V RVV Zvfh fp16 softmax dispatch (uses f32 RVV primitives + Zvfh cast).
--*/

#include "softmax.h"
#include "halfgemm.h"
#include "mlasi.h"
#include "../inc/mlas.h"

#if defined(MLAS_TARGET_RISCV64) && defined(MLAS_USE_RVV_ZVFH)

#include <riscv_vector.h>
#include <cmath>
#include <limits>
#include <vector>

namespace {

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

MLAS_FORCEINLINE float
ToFloat(MLAS_FP16 v)
{
    return v.ToFloat();
}

MLAS_FORCEINLINE MLAS_FP16
ToHalf(float v)
{
    return MLAS_FP16(v);
}

void
TanhFp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        Output[i] = ToHalf(std::tanh(ToFloat(Input[i])));
    }
}

void
SoftcapFp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 Softcap)
{
    const float cap = ToFloat(Softcap);
    for (size_t i = 0; i < N; ++i) {
        Output[i] = ToHalf(cap * std::tanh(ToFloat(Input[i]) / cap));
    }
}

MLAS_FP16
ReduceMaxFp16(const MLAS_FP16* Input, size_t N)
{
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < N; ++i) {
        max_val = std::max(max_val, ToFloat(Input[i]));
    }
    return ToHalf(max_val);
}

void
ExpFp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        Output[i] = ToHalf(std::exp(ToFloat(Input[i])));
    }
}

MLAS_FP16
SumExpFp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 NegativeMaximum)
{
    const float neg = ToFloat(NegativeMaximum);
    float sum = 0.f;
    for (size_t i = 0; i < N; ++i) {
        const float e = std::exp(ToFloat(Input[i]) + neg);
        sum += e;
        if (Output != nullptr) {
            Output[i] = ToHalf(e);
        }
    }
    return ToHalf(sum);
}

void
SoftmaxFp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N, const MLAS_FP16 Sum)
{
    const float inv = 1.0f / ToFloat(Sum);
    const auto* in = AsRaw(Input);
    auto* out = AsRaw(Output);
    size_t i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e16m4(N - i);
        vfloat16m4_t v = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vle16_v_u16m4(in + i, vl));
        vfloat32m8_t f = __riscv_vfwcvt_f_f_v_f32m8(v, vl);
        f = __riscv_vfmul_vf_f32m8(f, inv, vl);
        v = __riscv_vfncvt_f_f_w_f16m4(f, vl);
        __riscv_vse16_v_u16m4(out + i, __riscv_vreinterpret_v_f16m4_u16m4(v), vl);
        i += vl;
    }
}

void
LogSoftmaxFp16(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N,
               const MLAS_FP16 NegativeMaximum, const MLAS_FP16 LogSum)
{
    const float neg = ToFloat(NegativeMaximum);
    const float logsum = ToFloat(LogSum);
    for (size_t i = 0; i < N; ++i) {
        Output[i] = ToHalf(ToFloat(Input[i]) + neg - logsum);
    }
}

}  // namespace

const MLAS_SOFTMAX_DISPATCH MlasSoftmaxDispatchRvv = []() {
    MLAS_SOFTMAX_DISPATCH d{};
    d.Tanh_Fp16 = TanhFp16;
    d.Softcap_Fp16 = SoftcapFp16;
    d.Exp_Fp16 = ExpFp16;
    d.ReduceMax_Fp16 = ReduceMaxFp16;
    d.SumExp_Fp16 = SumExpFp16;
    d.Softmax_Fp16 = SoftmaxFp16;
    d.LogSoftmax_Fp16 = LogSoftmaxFp16;
    return d;
}();

#endif
