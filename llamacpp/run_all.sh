#!/usr/bin/env bash
set -u
V=~/validate_model.sh
run(){ "$V" "$1" "$2" "$3" "$4"; }

run "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf" "tinyllama-1.1b.q4_0.gguf" "tinyllama-1.1b" "1.1B"
run "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_0.gguf" "qwen2.5-1.5b.q4_0.gguf" "qwen2.5-1.5b" "1.5B"
run "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf" "llama-3.2-1b.q4_0.gguf" "llama-3.2-1b" "1.2B"
run "https://huggingface.co/second-state/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_0.gguf" "gemma-2-2b-it.q4_0.gguf" "gemma-2-2b-it" "2.6B"
run "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_0.gguf" "qwen2.5-3b.q4_0.gguf" "qwen2.5-3b" "3.1B"
run "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf" "llama-3.2-3b.q4_0.gguf" "llama-3.2-3b" "3.2B"
run "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_0.gguf" "phi-3.5-mini.q4_0.gguf" "phi-3.5-mini" "3.8B"
run "https://huggingface.co/second-state/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_0.gguf" "mistral-7b-v0.3.q4_0.gguf" "mistral-7b-v0.3" "7.2B"

# Qwen2.5-7B is sharded (2 files); fetch both then validate shard-1 (llama auto-joins)
MODELS=~/llama-x60/models
B="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main"
if wget -q -O "$MODELS/qwen2.5-7b.q4_0-00001-of-00002.gguf" "$B/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf" && \
   wget -q -O "$MODELS/qwen2.5-7b.q4_0-00002-of-00002.gguf" "$B/qwen2.5-7b-instruct-q4_0-00002-of-00002.gguf"; then
  run "PRELOADED" "qwen2.5-7b.q4_0-00001-of-00002.gguf" "qwen2.5-7b" "7.6B"
  rm -f "$MODELS"/qwen2.5-7b.q4_0-0000*.gguf
else
  echo -e "qwen2.5-7b\t7.6B\tNA\tDL_FAIL\tNA\tNA\tNA\tNA\tNA" >> ~/model_validation.tsv
  rm -f "$MODELS"/qwen2.5-7b.q4_0-0000*.gguf
fi
echo ALL_MODELS_DONE
