#pragma once

// Deterministic synthetic transformer trace for the demo. Values come from a fixed
// cost model and a seeded PRNG; they are NOT measurements and are always labelled
// as synthetic (DataSource::Synthetic / "DEMO").

#include <atomic>
#include <cstdint>
#include <vector>

#include "llm_trace/adapter.hpp"

namespace llm_trace {

struct SyntheticConfig {
  int layers = 12;
  int prompt_tokens = 16;
  int decode_tokens = 64;
  int hidden_size = 768;
  int heads = 12;
  int vocab_size = 32000;
  std::uint64_t seed = 42;
  // Live pacing only; does not affect generated timestamps. 0 = no delay.
  double steps_per_second = 8.0;
};

// Number of events produced for one decode step (excluding run_start/run_end).
int synthetic_events_per_step(const SyntheticConfig& config);

// Full deterministic trace: run_start, one prefill step, decode steps, run_end.
std::vector<TraceEvent> generate_synthetic(const SyntheticConfig& config);
TraceMetadata synthetic_metadata(const SyntheticConfig& config);

class SyntheticAdapter : public LiveAdapter {
 public:
  explicit SyntheticAdapter(SyntheticConfig config) : config_(config) {}
  std::string name() const override { return "synthetic"; }
  AdapterCapabilities capabilities() const override;
  TraceMetadata metadata() const override { return synthetic_metadata(config_); }
  void run(Collector& collector, const std::atomic<bool>& stop) override;

 private:
  SyntheticConfig config_;
};

}  // namespace llm_trace
