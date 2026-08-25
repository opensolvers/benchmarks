/*++
  RISC-V RVV Zvfh fp16 elementwise dispatch.
--*/

#include "eltwise.h"
#include "halfgemm.h"
#include "mlasi.h"

#if defined(MLAS_TARGET_RISCV64) && defined(MLAS_USE_RVV_ZVFH)

#include <riscv_vector.h>

namespace {

void
AddKernelFp16(const MLAS_FP16* left, const MLAS_FP16* right, MLAS_FP16* output, size_t N)
{
    const auto* l = reinterpret_cast<const _mlas_fp16_*>(left);
    const auto* r = reinterpret_cast<const _mlas_fp16_*>(right);
    auto* o = reinterpret_cast<_mlas_fp16_*>(output);
    size_t i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e16m4(N - i);
        vfloat16m4_t a = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vle16_v_u16m4(l + i, vl));
        vfloat16m4_t b = __riscv_vreinterpret_v_u16m4_f16m4(__riscv_vle16_v_u16m4(r + i, vl));
        vfloat16m4_t out = __riscv_vfadd_vv_f16m4(a, b, vl);
        __riscv_vse16_v_u16m4(o + i, __riscv_vreinterpret_v_f16m4_u16m4(out), vl);
        i += vl;
    }
}

}  // namespace

const MLAS_ELTWISE_DISPATCH MlasEltwiseDispatchRvv = []() {
    MLAS_ELTWISE_DISPATCH d{};
    d.Add_Fp16 = AddKernelFp16;
    return d;
}();

#endif
