#include "llm_trace/synthetic.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

namespace llm_trace {

namespace {

// splitmix64: portable and identical on every platform (unlike <random> distributions).
class Prng {
 public:
  explicit Prng(std::uint64_t seed) : state_(seed) {}
  std::uint64_t next() {
    std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  // Uniform in [0, 1).
  double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

 private:
  std::uint64_t state_;
};

TensorMetadata tensor(const std::string& name, std::vector<std::int64_t> shape) {
  TensorMetadata t;
  t.name = name;
  t.dtype = "float16";
  t.element_count = element_count_of(shape);
  t.byte_count = *t.element_count * 2;
  t.shape = std::move(shape);
  return t;
}

std::string layer_name(int layer) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "decoder.layer.%02d", layer);
  return buffer;
}

struct StepBuilder {
  const SyntheticConfig& cfg;
  Prng& rng;
  std::vector<TraceEvent>& out;
  std::uint64_t& clock_ns;
  std::int64_t token_index;
  std::int64_t q_len;
  std::int64_t kv_len;

  void op(const std::string& name, Category category, const std::string& op_type,
          std::optional<int> layer, double base_ns, std::vector<TensorMetadata> inputs,
          std::vector<TensorMetadata> outputs) {
    TraceEvent e;
    e.type = EventType::Operation;
    e.name = name;
    e.category = category;
    e.op_type = op_type;
    e.timestamp_ns = clock_ns;
    const double jitter = 0.85 + 0.3 * rng.unit();
    const auto duration = static_cast<std::uint64_t>(std::max(200.0, base_ns * jitter));
    e.duration_ns = duration;
    if (layer) {
      e.layer_index = *layer;
      e.parent_id = layer_name(*layer);
    }
    e.token_index = token_index;
    e.device = "synthetic-cpu";
    e.inputs = std::move(inputs);
    e.outputs = std::move(outputs);
    out.push_back(std::move(e));
    clock_ns += duration + 150;  // fixed launch gap between operations
  }
};

void emit_step(const SyntheticConfig& cfg, Prng& rng, std::vector<TraceEvent>& out,
               std::uint64_t& clock_ns, std::int64_t token_index, std::int64_t q_len,
               std::int64_t kv_len) {
  StepBuilder b{cfg, rng, out, clock_ns, token_index, q_len, kv_len};
  const std::int64_t h = cfg.hidden_size;
  const std::int64_t heads = cfg.heads;
  const std::int64_t head_dim = std::max<std::int64_t>(1, h / std::max<std::int64_t>(1, heads));
  const std::int64_t ffn = 4 * h;
  const double q = static_cast<double>(q_len);
  const double kv = static_cast<double>(kv_len);
  const double hd = static_cast<double>(h);
  // Cost model: ~1 ns per 64 multiply-accumulates plus fixed overheads. Synthetic only.
  auto matmul_ns = [](double m, double k, double n) { return 2'000.0 + m * k * n / 64.0; };

  const auto hidden = std::vector<std::int64_t>{1, q_len, h};
  b.op("embedding.tokens", Category::Embedding, "Gather", std::nullopt, 3'000.0 + 40.0 * q,
       {tensor("input_ids", {1, q_len})}, {tensor("hidden_states", hidden)});

  for (int layer = 0; layer < cfg.layers; ++layer) {
    const std::string prefix = layer_name(layer);
    b.op(prefix + ".input_layernorm", Category::Normalization, "LayerNormalization", layer,
         1'500.0 + q * hd / 32.0, {tensor("hidden_states", hidden)},
         {tensor("normed", hidden)});
    b.op(prefix + ".attention.qkv_proj", Category::Attention, "MatMul", layer,
         matmul_ns(q, hd, 3 * hd), {tensor("normed", hidden)},
         {tensor("qkv", {1, q_len, 3 * h})});
    b.op(prefix + ".attention.scores", Category::Attention, "MatMul", layer,
         matmul_ns(q * static_cast<double>(heads), static_cast<double>(head_dim), kv),
         {tensor("query", {1, heads, q_len, head_dim}), tensor("key", {1, heads, kv_len, head_dim})},
         {tensor("scores", {1, heads, q_len, kv_len})});
    b.op(prefix + ".attention.softmax", Category::Attention, "Softmax", layer,
         1'000.0 + q * kv * static_cast<double>(heads) / 16.0,
         {tensor("scores", {1, heads, q_len, kv_len})},
         {tensor("probs", {1, heads, q_len, kv_len})});
    b.op(prefix + ".attention.context", Category::Attention, "MatMul", layer,
         matmul_ns(q * static_cast<double>(heads), kv, static_cast<double>(head_dim)),
         {tensor("probs", {1, heads, q_len, kv_len}), tensor("value", {1, heads, kv_len, head_dim})},
         {tensor("context", {1, heads, q_len, head_dim})});
    b.op(prefix + ".attention.out_proj", Category::Attention, "MatMul", layer,
         matmul_ns(q, hd, hd), {tensor("context", hidden)}, {tensor("attn_out", hidden)});
    b.op(prefix + ".post_attention_layernorm", Category::Normalization, "LayerNormalization",
         layer, 1'500.0 + q * hd / 32.0, {tensor("attn_out", hidden)}, {tensor("normed", hidden)});
    b.op(prefix + ".mlp.up_proj", Category::Mlp, "MatMul", layer,
         matmul_ns(q, hd, static_cast<double>(ffn)), {tensor("normed", hidden)},
         {tensor("up", {1, q_len, ffn})});
    b.op(prefix + ".mlp.activation", Category::Mlp, "Gelu", layer,
         800.0 + q * static_cast<double>(ffn) / 24.0, {tensor("up", {1, q_len, ffn})},
         {tensor("act", {1, q_len, ffn})});
    b.op(prefix + ".mlp.down_proj", Category::Mlp, "MatMul", layer,
         matmul_ns(q, static_cast<double>(ffn), hd), {tensor("act", {1, q_len, ffn})},
         {tensor("mlp_out", hidden)});
  }

  b.op("final_layernorm", Category::Normalization, "LayerNormalization", std::nullopt,
       1'500.0 + q * hd / 32.0, {tensor("hidden_states", hidden)}, {tensor("normed", hidden)});
  b.op("lm_head", Category::Output, "MatMul", std::nullopt,
       matmul_ns(1.0, hd, static_cast<double>(cfg.vocab_size)),
       {tensor("normed", {1, 1, h})}, {tensor("logits", {1, 1, cfg.vocab_size})});
}

}  // namespace

int synthetic_events_per_step(const SyntheticConfig& config) { return 3 + config.layers * 10; }

TraceMetadata synthetic_metadata(const SyntheticConfig& config) {
  TraceMetadata m;
  m.run_id = "demo-seed-" + std::to_string(config.seed);
  m.source = DataSource::Synthetic;
  m.origin = DataSource::Synthetic;
  m.adapter = "synthetic";
  m.runtime = "synthetic cost model";
  m.runtime_version = LLM_TRACE_VERSION;
  m.model = "synthetic-decoder-" + std::to_string(config.layers) + "L-" +
            std::to_string(config.hidden_size) + "d";
  m.capabilities.operation_timing = true;
  m.capabilities.tensor_metadata = true;
  m.capabilities.layer_hierarchy = true;
  m.capabilities.token_index = true;
  m.notes.push_back("Synthetic demo data: timings come from a cost model, not a real model.");
  return m;
}

AdapterCapabilities SyntheticAdapter::capabilities() const {
  return synthetic_metadata(config_).capabilities;
}

std::vector<TraceEvent> generate_synthetic(const SyntheticConfig& config) {
  SyntheticConfig cfg = config;
  cfg.layers = std::max(1, cfg.layers);
  cfg.prompt_tokens = std::max(1, cfg.prompt_tokens);
  cfg.decode_tokens = std::max(0, cfg.decode_tokens);
  cfg.hidden_size = std::max(8, cfg.hidden_size);
  cfg.heads = std::max(1, cfg.heads);

  Prng rng(cfg.seed);
  std::vector<TraceEvent> out;
  out.reserve(static_cast<std::size_t>(synthetic_events_per_step(cfg)) *
                  static_cast<std::size_t>(cfg.decode_tokens + 1) +
              2);
  std::uint64_t clock = 0;

  TraceEvent start;
  start.type = EventType::RunStart;
  start.name = "run_start";
  start.category = Category::Session;
  start.timestamp_ns = clock;
  out.push_back(start);

  emit_step(cfg, rng, out, clock, 0, cfg.prompt_tokens, cfg.prompt_tokens);
  for (int t = 0; t < cfg.decode_tokens; ++t) {
    const std::int64_t token = cfg.prompt_tokens + t;
    emit_step(cfg, rng, out, clock, token, 1, token + 1);
  }

  TraceEvent end;
  end.type = EventType::RunEnd;
  end.name = "run_end";
  end.category = Category::Session;
  end.timestamp_ns = clock;
  out.push_back(end);

  for (std::size_t i = 0; i < out.size(); ++i) out[i].sequence = i;
  return out;
}

void SyntheticAdapter::run(Collector& collector, const std::atomic<bool>& stop) {
  collector.set_capturing(true);
  const auto events = generate_synthetic(config_);
  const auto per_step = static_cast<std::size_t>(synthetic_events_per_step(config_));
  const auto delay =
      config_.steps_per_second > 0
          ? std::chrono::duration<double>(1.0 / config_.steps_per_second)
          : std::chrono::duration<double>(0);
  // run_start + prefill step, then one decode step per tick.
  std::size_t index = 0;
  std::size_t batch = 1 + per_step;
  while (index < events.size() && !stop.load()) {
    const std::size_t end = std::min(events.size(), index + batch);
    for (; index < end; ++index) collector.submit(events[index]);
    batch = per_step;
    if (delay.count() > 0 && index < events.size()) {
      const auto until = std::chrono::steady_clock::now() + delay;
      while (!stop.load() && std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  collector.set_capturing(false);
}

}  // namespace llm_trace
