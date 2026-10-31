# llama.cpp model suite on BPI-SM10 / K3 (A100 IME2)

2026-10-31. The RV2 study sets, run on the K3 only: the **10 Q4_0** models
from [`models.md`](models.md) and the **17 Q4_K_M** models from
[`results_17model_m1gemv_vs_rvv.md`](results_17model_m1gemv_vs_rvv.md).
No RVV arm. This does not replace the r=3 Qwen2.5-0.5B thread sweep in
[`results_sm10_q4_0.md`](results_sm10_q4_0.md).

| | |
| --- | --- |
| Board | Banana Pi BPI-SM10, SpacemiT K3, Bianbu 4.0.6 |
| Fork | `opensolvers/llama.cpp` branch `x60-ime-rvv` @ `107ead2f2` |
| Bench | `llama-bench -p 128 -n 32 -t 8 -r 1` |
| Coherence | `llama-cli -n 32 -t 8 -c 1024 --temp 0 --single-turn`. Prompt: “Explain what RISC-V vector extensions are, in two short sentences.” |
| Launch | `echo $$ > /proc/set_ai_thread` before `exec`, then `taskset -c 8-15`. `SPACEMIT_DISABLE_TCM=1` |
| Runtime log | `use_ime2: 1`, `use_ime1: 0` on all 27 runs |
| Clock | A100 userspace 1.800 GHz |

Tokens per second are one repeat. A row is coherent when the 32 tokens are
English and not a repeated token. Raw rows:
[`results_sm10_model_suite.tsv`](results_sm10_model_suite.tsv).

## Q4_0 (10 models)

Nine of ten are coherent. **Qwen2.5-3B-Instruct repeats the character `G`**
for the whole completion. The bench still printed a speed; that row is not
a valid result. The same model’s Q4_K_M file, below, is coherent.

The 0.5B row is one repeat of the file in the r=3 sweep. Prefill **540.21**
sits next to that sweep’s 8-thread mean (**525.96**). Decode **57.78** sits
next to the 4-thread peak of that sweep (**58.67**), not its 8-thread mean
(**38.78**).

| Model | MiB | Coherent | pp128 | tg32 |
| --- | ---: | --- | ---: | ---: |
| Qwen2.5-0.5B-Instruct | 408 | yes | 540.21 | 57.78 |
| TinyLlama-1.1B-Chat | 608 | yes | 279.67 | 35.26 |
| Llama-3.2-1B-Instruct | 737 | yes | 291.49 | 25.86 |
| Qwen2.5-1.5B-Instruct | 1016 | yes | 204.11 | 22.83 |
| Gemma-2-2B-it | 1554 | yes | 114.81 | 10.53 |
| Qwen2.5-3B-Instruct | 1905 | repeat | 103.75 | 12.32 |
| Llama-3.2-3B-Instruct | 1832 | yes | 108.70 | 11.72 |
| Phi-3.5-mini-instruct | 2081 | yes | 78.99 | 10.48 |
| Mistral-7B-Instruct-v0.3 | 3922 | yes | 45.57 | 5.75 |
| Qwen2.5-7B-Instruct | 3798 | yes | 48.81 | 5.70 |

Qwen2.5-7B is the two-shard Q4_0 GGUF. Qwen2.5-3B Q4_0 stores `output.weight`
as Q6_K with shape 2048×151936. Qwen2.5-1.5B Q4_0 also has a Q6_K head and
was coherent, and Qwen2.5-7B Q4_0 was coherent.

## Q4_K_M (17 models)

All 17 are coherent. On this build Q4_K dispatches to the IME2 Q4 kernel
(`gemm_kernel_i8i4`). That is a different path from the RV2 `m1gemv` study,
where Q4_K_M did not engage `smt.vmadot`.

`stablelm-2-1.6b` is `second-state/stablelm-2-zephyr-1.6b-GGUF`, file
`stablelm-2-zephyr-1_6b-Q4_K_M.gguf`. DeepSeek-Coder answered in English and
described RISC-V as an ARM ISA. The DeepSeek-R1 distill opened with a
thinking trace. Both are the model, not a repeated token.

| Model | MiB | pp128 | tg32 |
| --- | ---: | ---: | ---: |
| qwen2.5-0.5b | 468 | 433.34 | 51.44 |
| llama-3.2-1b | 770 | 219.68 | 24.49 |
| gemma-3-1b | 768 | 191.69 | 22.63 |
| deepseek-coder-1.3b | 833 | 172.65 | 25.31 |
| qwen2.5-1.5b | 1065 | 147.25 | 20.09 |
| stablelm-2-1.6b | 983 | 167.67 | 21.19 |
| falcon3-1b | 1008 | 179.10 | 21.78 |
| smollm2-1.7b | 1006 | 132.45 | 18.90 |
| deepseek-r1-qwen-1.5b | 1065 | 147.74 | 20.06 |
| granite-3.1-2b | 1473 | 86.95 | 12.98 |
| gemma-2-2b | 1629 | 87.09 | 9.07 |
| qwen2.5-3b | 2007 | 73.31 | 10.69 |
| falcon3-3b | 1912 | 82.71 | 9.94 |
| llama-3.2-3b | 1925 | 77.84 | 10.37 |
| phi-3.5-mini | 2282 | 50.79 | 9.25 |
| gemma-3-4b | 2374 | 58.17 | 7.48 |
| phi-4-mini | 2376 | 60.76 | 8.39 |

The BXM-4-64 GPU is still unmeasured.
