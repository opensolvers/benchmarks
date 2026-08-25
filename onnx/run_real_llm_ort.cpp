// Real int4 ONNX LLM decode via ORT C++ (no onnxruntime-genai).
// Prefill + greedy decode with past_key_values KV cache.
// Optional ORT chrome-trace profiling: pass profile_prefix as argv[5].
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

static int64_t argmax(const float* p, int64_t n) {
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

static bool finite_buf(const float* p, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(p[i])) return false;
  }
  return true;
}

struct KvCache {
  static constexpr int64_t kBatch = 1;
  static constexpr int64_t kHeads = 2;
  static constexpr int64_t kHeadDim = 64;
  static constexpr int kLayers = 24;
  static constexpr int64_t kMaxSeq = 512;

  int64_t seq = 0;
  std::vector<float> key_data[kLayers];
  std::vector<float> value_data[kLayers];

  void init() {
    seq = 0;
    const size_t cap =
        (size_t)kBatch * (size_t)kHeads * (size_t)kMaxSeq * (size_t)kHeadDim;
    for (int i = 0; i < kLayers; ++i) {
      key_data[i].assign(cap, 0.f);
      value_data[i].assign(cap, 0.f);
    }
  }

  size_t elems() const {
    return (size_t)kBatch * (size_t)kHeads * (size_t)seq * (size_t)kHeadDim;
  }

  void adopt_present(int layer, const float* data, int64_t new_seq) {
    seq = new_seq;
    const size_t n = elems();
    if (n > key_data[layer].size()) die("KV key overflow");
    std::memcpy(key_data[layer].data(), data, n * sizeof(float));
  }

  void adopt_present_value(int layer, const float* data, int64_t new_seq) {
    seq = new_seq;
    const size_t n = elems();
    if (n > value_data[layer].size()) die("KV value overflow");
    std::memcpy(value_data[layer].data(), data, n * sizeof(float));
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

  std::vector<int64_t> tokens = {151644, 8948, 198, 3838, 374, 220, 17, 10, 17,
                                 30,     151645, 198, 151644, 77091, 198};

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
  std::printf("loaded %s  inputs=%zu outputs=%zu threads=%d\n", model_path,
              n_in, n_out, threads);

  KvCache kv;
  kv.init();

  std::vector<int64_t> attn_mask;
  std::vector<int64_t> ids_shape;
  std::vector<int64_t> mask_shape;
  std::vector<int64_t> past_shape;
  float dummy = 0.f;

  auto run_step = [&](const std::vector<int64_t>& step_toks,
                      const char* run_tag) -> int64_t {
    const int64_t cur = (int64_t)step_toks.size();
    const int64_t total = kv.seq + cur;

    ids_shape = {1, cur};
    attn_mask.assign((size_t)total, 1);
    mask_shape = {1, total};
    past_shape = {1, KvCache::kHeads, kv.seq, KvCache::kHeadDim};

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
        float* ptr = kv.seq == 0 ? &dummy : kv.key_data[layer].data();
        size_t ne = kv.seq == 0 ? 0 : kv.elems();
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem, ptr, ne, past_shape.data(), past_shape.size()));
      } else if (n.rfind("past_key_values.", 0) == 0 &&
                 n.find(".value") != std::string::npos) {
        int layer = std::atoi(n.c_str() + std::strlen("past_key_values."));
        float* ptr = kv.seq == 0 ? &dummy : kv.value_data[layer].data();
        size_t ne = kv.seq == 0 ? 0 : kv.elems();
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem, ptr, ne, past_shape.data(), past_shape.size()));
      } else {
        std::fprintf(stderr, "unknown input %s\n", n.c_str());
        die("unknown input");
      }
    }

    Ort::RunOptions run_opts;
    if (run_tag) run_opts.SetRunTag(run_tag);

    auto t0 = std::chrono::steady_clock::now();
    auto outputs = session.Run(run_opts, in_c.data(), inputs.data(), inputs.size(),
                               out_c.data(), out_c.size());
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const float* logits = nullptr;
    int64_t seq_l = 0, vocab = 0;
    size_t logits_n = 0;

    for (size_t i = 0; i < out_s.size(); ++i) {
      const std::string& n = out_s[i];
      auto info = outputs[i].GetTensorTypeAndShapeInfo();
      auto sh = info.GetShape();
      if (n == "logits") {
        logits = outputs[i].GetTensorData<float>();
        logits_n = info.GetElementCount();
        seq_l = sh[1];
        vocab = sh[2];
      } else if (n.rfind("present.", 0) == 0 &&
                 n.find(".key") != std::string::npos) {
        int layer = std::atoi(n.c_str() + std::strlen("present."));
        int64_t new_seq = sh[2];
        kv.adopt_present(layer, outputs[i].GetTensorData<float>(), new_seq);
      } else if (n.rfind("present.", 0) == 0 &&
                 n.find(".value") != std::string::npos) {
        int layer = std::atoi(n.c_str() + std::strlen("present."));
        int64_t new_seq = sh[2];
        kv.adopt_present_value(layer, outputs[i].GetTensorData<float>(),
                               new_seq);
      }
    }
    if (!logits) die("missing logits");
    if (!finite_buf(logits, logits_n)) die("non-finite logits");

    const float* last = logits + (size_t)(seq_l - 1) * (size_t)vocab;
    int64_t next = argmax(last, vocab);
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
    std::vector<int64_t> one = {tokens.back()};
    const char* tag = (profile_prefix && i == 0) ? "decode" : nullptr;
    next = run_step(one, tag);
    tokens.push_back(next);
    if (next == 151645 || next == 151643) break;
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
