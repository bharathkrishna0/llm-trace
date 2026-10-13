#include <catch2/catch_test_macros.hpp>

#include "llm_trace/synthetic.hpp"

using namespace llm_trace;

TEST_CASE("synthetic trace is deterministic for a seed") {
  SyntheticConfig cfg;
  cfg.layers = 2;
  cfg.decode_tokens = 3;
  const auto a = generate_synthetic(cfg);
  const auto b = generate_synthetic(cfg);
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].name == b[i].name);
    CHECK(a[i].timestamp_ns == b[i].timestamp_ns);
    CHECK(a[i].duration_ns == b[i].duration_ns);
  }
  cfg.seed = 7;
  const auto c = generate_synthetic(cfg);
  bool differs = false;
  for (std::size_t i = 0; i < a.size(); ++i)
    differs = differs || a[i].duration_ns != c[i].duration_ns;
  CHECK(differs);
}

TEST_CASE("synthetic trace has the expected structure") {
  SyntheticConfig cfg;
  cfg.layers = 3;
  cfg.decode_tokens = 2;
  const auto events = generate_synthetic(cfg);
  const auto per_step = static_cast<std::size_t>(synthetic_events_per_step(cfg));
  CHECK(events.size() == 2 + per_step * 3);
  CHECK(events.front().type == EventType::RunStart);
  CHECK(events.back().type == EventType::RunEnd);
  std::uint64_t last_ts = 0;
  for (std::size_t i = 0; i < events.size(); ++i) {
    CHECK(events[i].sequence == i);
    CHECK(validate(events[i]).empty());
    CHECK(events[i].timestamp_ns >= last_ts);
    last_ts = events[i].timestamp_ns;
  }
}

TEST_CASE("synthetic metadata is labelled and claims no attention values") {
  const auto meta = synthetic_metadata(SyntheticConfig{});
  CHECK(meta.source == DataSource::Synthetic);
  CHECK(meta.origin == DataSource::Synthetic);
  CHECK(meta.capabilities.operation_timing);
  CHECK_FALSE(meta.capabilities.attention_values);
  CHECK_FALSE(meta.capabilities.activation_summaries);
  CHECK_FALSE(meta.notes.empty());
}

TEST_CASE("synthetic adapter submits every event and stops capturing") {
  SyntheticConfig cfg;
  cfg.layers = 1;
  cfg.decode_tokens = 4;
  cfg.steps_per_second = 0;
  SyntheticAdapter adapter(cfg);
  Collector collector(16, adapter.metadata());
  std::atomic<bool> stop{false};
  adapter.run(collector, stop);
  const auto stats = collector.stats();
  CHECK_FALSE(stats.capturing);
  CHECK(stats.accepted == generate_synthetic(cfg).size());
  CHECK(stats.retained == 16);
  CHECK(stats.overwritten == stats.accepted - 16);
}

TEST_CASE("synthetic adapter honours stop") {
  SyntheticConfig cfg;
  cfg.decode_tokens = 1000;
  cfg.steps_per_second = 1;
  SyntheticAdapter adapter(cfg);
  Collector collector(1 << 16, adapter.metadata());
  std::atomic<bool> stop{true};
  adapter.run(collector, stop);
  CHECK(collector.stats().accepted < generate_synthetic(cfg).size());
}
