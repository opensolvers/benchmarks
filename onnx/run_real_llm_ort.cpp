// Real int4 ONNX LLM decode via ORT C++ (no onnxruntime-genai).
// Prefill + greedy decode with past_key_values KV cache.
// Optional ORT chrome-trace profiling: pass profile_prefix as argv[5].
//
// KV / prompt config via env (defaults = Qwen2.5-0.5B):
//   LLM_KV_HEADS, LLM_HEAD_DIM, LLM_LAYERS, LLM_MAX_SEQ
//   LLM_PROMPT_TOKENS  comma-separated int64 token ids
//   LLM_EOS_IDS        comma-separated stop token ids
// KV dtype is auto-detected from the model (float32 or float16).
#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void die(const char* msg) {
  std::fprintf(stderr, "FATAL: %s\n", msg);
  std::exit(1);
}

static int env_int(const char* key, int def) {
  const char* v = std::getenv(key);
  if (!v || !*v) return def;
  return std::atoi(v);
}

static std::vector<int64_t> parse_ids(const char* csv, const std::vector<int64_t>& def) {
  if (!csv || !*csv) return def;
  std::vector<int64_t> out;
  const char* p = csv;
  while (*p) {
    char* end = nullptr;
    long long v = std::strtoll(p, &end, 10);
    if (end == p) break;
    out.push_back((int64_t)v);
    p = end;
    if (*p == ',') ++p;
  }
  return out.empty() ? def : out;
}

static int64_t argmax_f32(const float* p, int64_t n) {
  int64_t best = 0;
  float v = p[0];
  for (int64_t i = 1; i < n; ++i) {
    if (p[i] > v) {
      v = p[i];
      best = i;
    }
  }
  return best;
}

static int64_t argmax_f16(const Ort::Float16_t* p, int64_t n) {
  int64_t best = 0;
  float v = static_cast<float>(p[0]);
  for (int64_t i = 1; i < n; ++i) {
    float x = static_cast<float>(p[i]);
    if (x > v) {
      v = x;
      best = i;
    }
  }
  return best;
}

static bool finite_f32(const float* p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(p[i])) return false;
  }
  return true;
}

static bool finite_f16(const Ort::Float16_t* p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(static_cast<float>(p[i]))) return false;
  }
  return true;
}

struct KvCache {
  int64_t batch = 1;
  int64_t heads = 2;
  int64_t head_dim = 64;
  int layers = 24;
  int64_t max_seq = 512;
  bool fp16 = false;

  int64_t seq = 0;
  std::vector<std::vector<float>> key_f32, value_f32;
  std::vector<std::vector<Ort::Float16_t>> key_f16, value_f16;

  void init(int64_t h, int64_t hd, int L, int64_t ms, bool use_fp16) {
    heads = h;
    head_dim = hd;
    layers = L;
    max_seq = ms;
    fp16 = use_fp16;
    seq = 0;
    const size_t cap =
        (size_t)batch * (size_t)heads * (size_t)max_seq * (size_t)head_dim;
    if (fp16) {
      key_f16.assign((size_t)layers, {});
      value_f16.assign((size_t)layers, {});
      for (int i = 0; i < layers; ++i) {
        key_f16[i].assign(cap, Ort::Float16_t(0.f));
        value_f16[i].assign(cap, Ort::Float16_t(0.f));
      }
    } else {
      key_f32.assign((size_t)layers, {});
      value_f32.assign((size_t)layers, {});
      for (int i = 0; i < layers; ++i) {
        key_f32[i].assign(cap, 0.f);
        value_f32[i].assign(cap, 0.f);
      }
    }
  }

  size_t elems() const {
    return (size_t)batch * (size_t)heads * (size_t)seq * (size_t)head_dim;
  }

  void adopt_key(int layer, const void* data, int64_t new_seq) {
    if (layer < 0 || layer >= layers) die("KV layer OOB");
    seq = new_seq;
    const size_t n = elems();
    if (fp16) {
      if (n > key_f16[layer].size()) die("KV key overflow");
      std::memcpy(key_f16[layer].data(), data, n * sizeof(Ort::Float16_t));
    } else {
      if (n > key_f32[layer].size()) die("KV key overflow");
      std::memcpy(key_f32[layer].data(), data, n * sizeof(float));
    }
  }

  void adopt_value(int layer, const void* data, int64_t new_seq) {
    if (layer < 0 || layer >= layers) die("KV layer OOB");
    seq = new_seq;
    const size_t n = elems();
    if (fp16) {
      if (n > value_f16[layer].size()) die("KV value overflow");
      std::memcpy(value_f16[layer].data(), data, n * sizeof(Ort::Float16_t));
    } else {
      if (n > value_f32[layer].size()) die("KV value overflow");
      std::memcpy(value_f32[layer].data(), data, n * sizeof(float));
    }
  }
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <model.onnx> [new_tokens=12] [threads=1] [profile_prefix]\n",
                 argv[0]);
    return 2;
  }
  const char* model_path = argv[1];
  const int new_tokens = argc > 2 ? std::atoi(argv[2]) : 12;
  const int threads = argc > 3 ? std::atoi(argv[3]) : 1;
  const char* profile_prefix = argc > 4 ? argv[4] : nullptr;

  const std::vector<int64_t> default_prompt = {
      151644, 8948, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644,
      77091,  198};
  const std::vector<int64_t> default_eos = {151645, 151643};

  std::vector<int64_t> tokens =
      parse_ids(std::getenv("LLM_PROMPT_TOKENS"), default_prompt);
  std::vector<int64_t> eos_ids =
      parse_ids(std::getenv("LLM_EOS_IDS"), default_eos);

  const int64_t kv_heads = env_int("LLM_KV_HEADS", 2);
  const int64_t head_dim = env_int("LLM_HEAD_DIM", 64);
  const int layers = env_int("LLM_LAYERS", 24);
  const int64_t max_seq = env_int("LLM_MAX_SEQ", 512);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "real-llm");
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(threads);
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  so.EnableCpuMemArena();
  so.EnableMemPattern();
  if (profile_prefix) {
    so.EnableProfiling(profile_prefix);
    std::printf("profiling enabled prefix=%s\n", profile_prefix);
  }
  Ort::Session session(env, model_path, so);
  Ort::AllocatorWithDefaultOptions alloc;
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const size_t n_in = session.GetInputCount();
  const size_t n_out = session.GetOutputCount();
  std::vector<std::string> in_s, out_s;
  std::vector<const char*> in_c, out_c;
  for (size_t i = 0; i < n_in; ++i) {
    in_s.emplace_back(session.GetInputNameAllocated(i, alloc).get());
  }
  for (size_t i = 0; i < n_out; ++i) {
    out_s.emplace_back(session.GetOutputNameAllocated(i, alloc).get());
  }
  for (auto& s : in_s) in_c.push_back(s.c_str());
  for (auto& s : out_s) out_c.push_back(s.c_str());

  bool kv_fp16 = false;
  for (size_t i = 0; i < n_in; ++i) {
    if (in_s[i].rfind("past_key_values.", 0) == 0) {
      auto et = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
      if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) kv_fp16 = true;
      else if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) die("unsupported KV dtype");
      break;
    }
  }

  std::printf(
      "loaded %s  inputs=%zu outputs=%zu threads=%d  kv_heads=%lld "
      "head_dim=%lld layers=%d kv=%s prompt_len=%zu\n",
      model_path, n_in, n_out, threads, (long long)kv_heads,
      (long long)head_dim, layers, kv_fp16 ? "fp16" : "fp32", tokens.size());

  KvCache kv;
  kv.init(kv_heads, head_dim, layers, max_seq, kv_fp16);

  std::vector<int64_t> attn_mask;
  std::vector<int64_t> ids_shape;
  std::vector<int64_t> mask_shape;
  std::vector<int64_t> past_shape;
  float dummy_f32 = 0.f;
  Ort::Float16_t dummy_f16(0.f);

  auto is_eos = [&](int64_t id) {
    for (auto e : eos_ids)
      if (e == id) return true;
    return false;
  };

  auto run_step = [&](const std::vector<int64_t>& step_toks,
                      const char* run_tag) -> int64_t {
    const int64_t cur = (int64_t)step_toks.size();
    const int64_t total = kv.seq + cur;

    ids_shape = {1, cur};
    attn_mask.assign((size_t)total, 1);
    mask_shape = {1, total};
    past_shape = {1, kv.heads, kv.seq, kv.head_dim};

    std::vector<Ort::Value> inputs;
    inputs.reserve(n_in);
    for (size_t i = 0; i < n_in; ++i) {
      const std::string& n = in_s[i];
      if (n == "input_ids") {
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            mem, const_cast<int64_t*>(step_toks.data()), step_toks.size(),
            ids_shape.data(), ids_shape.size()));
      } else if (n == "attention_mask") {
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            mem, attn_mask.data(), attn_mask.size(), mask_shape.data(),
            mask_shape.size()));
      } else if (n.rfind("past_key_values.", 0) == 0 &&
                 n.find(".key") != std::string::npos) {
        int layer = std::atoi(n.c_str() + std::strlen("past_key_values."));
        if (kv.fp16) {
          Ort::Float16_t* ptr =
              kv.seq == 0 ? &dummy_f16 : kv.key_f16[layer].data();
          size_t ne = kv.seq == 0 ? 0 : kv.elems();
          inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
              mem, ptr, ne, past_shape.data(), past_shape.size()));
        } else {
          float* ptr = kv.seq == 0 ? &dummy_f32 : kv.key_f32[layer].data();
          size_t ne = kv.seq == 0 ? 0 : kv.elems();
          inputs.push_back(Ort::Value::CreateTensor<float>(
              mem, ptr, ne, past_shape.data(), past_shape.size()));
        }
      } else if (n.rfind("past_key_values.", 0) == 0 &&
                 n.find(".value") != std::string::npos) {
        int layer = std::atoi(n.c_str() + std::strlen("past_key_values."));
        if (kv.fp16) {
          Ort::Float16_t* ptr =
              kv.seq == 0 ? &dummy_f16 : kv.value_f16[layer].data();
          size_t ne = kv.seq == 0 ? 0 : kv.elems();
          inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
              mem, ptr, ne, past_shape.data(), past_shape.size()));
        } else {
          float* ptr = kv.seq == 0 ? &dummy_f32 : kv.value_f32[layer].data();
          size_t ne = kv.seq == 0 ? 0 : kv.elems();
          inputs.push_back(Ort::Value::CreateTensor<float>(
              mem, ptr, ne, past_shape.data(), past_shape.size()));
        }
      } else {
        std::fprintf(stderr, "unknown input %s\n", n.c_str());
        die("unknown input");
      }
    }

    Ort::RunOptions run_opts;
    if (run_tag) run_opts.SetRunTag(run_tag);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs;
    try {
      outputs = session.Run(run_opts, in_c.data(), inputs.data(), inputs.size(),
                            out_c.data(), out_c.size());
    } catch (const Ort::Exception& e) {
      std::fprintf(stderr, "ORT Run failed (%s): %s\n",
                   run_tag ? run_tag : "-", e.what());
      std::exit(1);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int64_t next = -1;
    int64_t seq_l = 0, vocab = 0;
    bool got_logits = false;

    for (size_t i = 0; i < out_s.size(); ++i) {
      const std::string& n = out_s[i];
      auto info = outputs[i].GetTensorTypeAndShapeInfo();
      auto sh = info.GetShape();
      auto et = info.GetElementType();
      if (n == "logits") {
        seq_l = sh[1];
        vocab = sh[2];
        size_t logits_n = info.GetElementCount();
        if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          const float* logits = outputs[i].GetTensorData<float>();
          if (!finite_f32(logits, logits_n)) die("non-finite logits");
          const float* last = logits + (size_t)(seq_l - 1) * (size_t)vocab;
          next = argmax_f32(last, vocab);
        } else if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
          const Ort::Float16_t* logits =
              outputs[i].GetTensorData<Ort::Float16_t>();
          if (!finite_f16(logits, logits_n)) die("non-finite logits");
          const Ort::Float16_t* last =
              logits + (size_t)(seq_l - 1) * (size_t)vocab;
          next = argmax_f16(last, vocab);
        } else {
          die("unsupported logits dtype");
        }
        got_logits = true;
      } else if (n.rfind("present.", 0) == 0 &&
                 n.find(".key") != std::string::npos) {
        int layer = std::atoi(n.c_str() + std::strlen("present."));
        const void* data = kv.fp16
                               ? static_cast<const void*>(
                                     outputs[i].GetTensorData<Ort::Float16_t>())
                               : static_cast<const void*>(
                                     outputs[i].GetTensorData<float>());
        kv.adopt_key(layer, data, sh[2]);
      } else if (n.rfind("present.", 0) == 0 &&
                 n.find(".value") != std::string::npos) {
        int layer = std::atoi(n.c_str() + std::strlen("present."));
        const void* data = kv.fp16
                               ? static_cast<const void*>(
                                     outputs[i].GetTensorData<Ort::Float16_t>())
                               : static_cast<const void*>(
                                     outputs[i].GetTensorData<float>());
        kv.adopt_value(layer, data, sh[2]);
      }
    }
    if (!got_logits) die("missing logits");

    std::printf("  cur=%lld total=%lld  %6.1f ms  next=%lld  tag=%s\n",
                (long long)cur, (long long)total, ms, (long long)next,
                run_tag ? run_tag : "-");
    return next;
  };

  std::printf("=== prefill %zu tokens ===\n", tokens.size());
  int64_t next = run_step(tokens, profile_prefix ? "prefill" : nullptr);
  tokens.push_back(next);

  std::printf("=== decode %d tokens ===\n", new_tokens);
  for (int i = 0; i < new_tokens - 1; ++i) {
    if (is_eos(tokens.back())) break;
    std::vector<int64_t> one = {tokens.back()};
    const char* tag = (profile_prefix && i == 0) ? "decode" : nullptr;
    next = run_step(one, tag);
    tokens.push_back(next);
  }

  if (profile_prefix) {
    auto profile = session.EndProfilingAllocated(alloc);
    std::printf("profile_json=%s\n", profile.get());
  }

  std::printf("token_ids:");
  for (auto t : tokens) std::printf(" %lld", (long long)t);
  std::printf("\nOK\n");
  return 0;
}
