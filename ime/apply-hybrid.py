#!/usr/bin/env python3
"""Apply SPACEMIT hybrid prefill/decode split to llama-pipe sources on the board."""
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "llama-pipe/src")

# --- ime_env.h ---
h = root / "ggml/src/ggml-cpu/spacemit/ime_env.h"
ht = h.read_text()
if "ime_min_m" not in ht:
    ht = ht.replace(
        "    bool                         use_ime1{ false };\n",
        "    bool                         use_ime1{ false };\n"
        "    /* Prefill (M>=ime_min_m) uses IME tiles; decode (M<ime_min_m) uses\n"
        "     * native weights via ggml RVV mul_mat. Needs SPACEMIT_HYBRID=1 (2x weight RAM). */\n"
        "    bool                         hybrid{ false };\n"
        "    int                          ime_min_m{ 4 };\n",
    )
    h.write_text(ht)
    print("patched ime_env.h")
else:
    print("ime_env.h already patched")

# --- ime_env.cpp ---
c = root / "ggml/src/ggml-cpu/spacemit/ime_env.cpp"
ct = c.read_text()
if "SPACEMIT_HYBRID" not in ct:
    needle = "    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;\n"
    insert = needle + """
    {
        char * hy = getenv("SPACEMIT_HYBRID");
        hybrid = hy != nullptr && strcmp(hy, "0") != 0;
        char * mm = getenv("SPACEMIT_IME_MIN_M");
        if (mm != nullptr) {
            ime_min_m = std::max(1, atoi(mm));
        }
        if (hybrid) {
            GGML_LOG_INFO(
                "CPU_RISCV64_SPACEMIT: hybrid=1 ime_min_m=%d (IME prefill M>=%d, RVV decode M<%d; ~2x weight RAM)\\n",
                ime_min_m, ime_min_m, ime_min_m);
        }
    }

"""
    if needle not in ct:
        raise SystemExit("ime_env.cpp needle missing")
    ct = ct.replace(needle, insert, 1)
    if "#include <cstdlib>" not in ct and "#include <stdlib.h>" not in ct:
        ct = ct.replace("#include <cstring>", "#include <cstring>\n#include <cstdlib>\n#include <algorithm>")
    c.write_text(ct)
    print("patched ime_env.cpp")
else:
    print("ime_env.cpp already patched")

# --- ime.cpp ---
p = root / "ggml/src/ggml-cpu/spacemit/ime.cpp"
t = p.read_text()

if '#include "../ops.h"' not in t and '#include "ops.h"' not in t:
    t = t.replace('#include "repack.h"\n', '#include "repack.h"\n#include "../ops.h"\n')

if "hybrid_plain_plus_tiled" not in t:
    marker = """        default:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            break;
    }

    return nbytes;
}
"""
    if marker not in t:
        raise SystemExit("nbytes marker missing")
    new = """        default:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            break;
    }

    // hybrid: keep native weights for RVV decode + IME tiles for prefill
    if (ggml::cpu::riscv64_spacemit::global_spine_env_info.hybrid) {
        nbytes = plain_nbytes() + nbytes;  // hybrid_plain_plus_tiled
    }
    return nbytes;
}
"""
    t = t.replace(marker, new, 1)
    print("patched nbytes")

if "hybrid_set_tensor" not in t:
    old_set = """static void ggml_backend_riscv64_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor *         tensor,
                                                            const void *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    if (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}
"""
    new_set = """static void ggml_backend_riscv64_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor *         tensor,
                                                            const void *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    if (tensor_traits) {
        if (ggml::cpu::riscv64_spacemit::global_spine_env_info.hybrid) {
            // hybrid_set_tensor: [native | ime-tiled]
            memcpy(tensor->data, data, size);
            ggml_tensor tiled = *tensor;
            tiled.data = (char *) tensor->data + size;
            auto OK = tensor_traits->repack(&tiled, data, size);
            GGML_ASSERT(OK == 0);
        } else {
            auto OK = tensor_traits->repack(tensor, data, size);
            GGML_ASSERT(OK == 0);
        }
    }

    GGML_UNUSED(buffer);
}
"""
    if old_set not in t:
        raise SystemExit("set_tensor block missing")
    t = t.replace(old_set, new_set, 1)
    print("patched set_tensor")

if "hybrid_decode_rvv" not in t:
    old_fwd = """        void *        w_data  = (void *) src0->data;
        const float * feature = (const float *) src1->data;
        float *       output  = (float *) dst->data;

        const int64_t gemm_m = ne11 * ne12 * ne13;
        const int64_t gemm_k = ne10;
        const int64_t gemm_n = ne01;
"""
    new_fwd = """        const float * feature = (const float *) src1->data;
        float *       output  = (float *) dst->data;

        const int64_t gemm_m = ne11 * ne12 * ne13;
        const int64_t gemm_k = ne10;
        const int64_t gemm_n = ne01;

        // hybrid_decode_rvv: decode/small-M uses native weights + stock ggml RVV mul_mat
        if (global_spine_env_info.hybrid && gemm_m < global_spine_env_info.ime_min_m) {
            ggml_tensor src0n = *src0;
            src0n.extra       = nullptr;
            ggml_tensor dstn  = *dst;
            dstn.src[0]       = &src0n;
            dstn.src[1]       = const_cast<ggml_tensor *>(src1);
            ggml_compute_forward_mul_mat(params, &dstn);
            return;
        }

        void * w_data = (void *) src0->data;
        if (global_spine_env_info.hybrid) {
            w_data = (char *) src0->data + ggml_nbytes(src0);
        }
"""
    if old_fwd not in t:
        raise SystemExit("forward_mul_mat header missing")
    t = t.replace(old_fwd, new_fwd, 1)
    print("patched forward_mul_mat")

# need string.h for memcpy in set_tensor - ime.cpp likely has cstring via other headers
if "#include <cstring>" not in t and '#include "string.h"' not in t:
    t = t.replace('#include "../ops.h"\n', '#include "../ops.h"\n#include <cstring>\n')

p.write_text(t)
print("done", p)
