#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include "llm_trace/activation.hpp"
#include "llm_trace/trace_io.hpp"

using namespace llm_trace;
using Catch::Approx;

TEST_CASE("summarize_activation computes statistics over every element") {
  const std::vector<float> v{1.0f, -1.0f, 0.0f, 3.0f};
  const auto s = summarize_activation(v.data(), v.size(), 0.5);
  CHECK(s.mean == Approx(0.75));
  CHECK(s.min == -1.0);
  CHECK(s.max == 3.0);
  CHECK(s.stddev == Approx(std::sqrt(2.1875)));
  CHECK(s.sparsity == Approx(0.25));
  CHECK(s.sparsity_epsilon == 0.5);
  CHECK(s.sampled_elements == 4);
  CHECK(s.full_tensor);
  CHECK(s.nonfinite_count == 0);
}

TEST_CASE("non-finite values are counted and excluded from the statistics") {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> v{nan, inf, -inf, 2.0f};
  const auto s = summarize_activation(v.data(), v.size(), 1e-6);
  CHECK(s.nonfinite_count == 3);
  CHECK(s.mean == 2.0);
  CHECK(s.min == 2.0);
  CHECK(s.max == 2.0);
  CHECK(s.stddev == 0.0);
  CHECK(s.sampled_elements == 4);

  const std::vector<float> only_nan{nan, nan};
  const auto none = summarize_activation(only_nan.data(), only_nan.size(), 1e-6);
  CHECK(none.nonfinite_count == 2);
  CHECK(none.mean == 0.0);
  CHECK(none.sparsity == 0.0);
  CHECK(summarize_activation(nullptr, 0, 1e-6).sampled_elements == 0);
}

namespace {

TraceEvent kernel(const std::string& name, int run) {
  TraceEvent e;
  e.name = name;
  e.category = Category::MatMul;
  e.duration_ns = 1000;
  e.attributes.emplace_back("run_index", std::to_string(run));
  return e;
}

LoadedTrace profile_like() {
  LoadedTrace t;
  t.metadata.source = DataSource::OnnxProfile;
  t.metadata.notes.push_back(
      "ONNX Runtime profiles contain no tensor values: attention weights and activation "
      "statistics are not available.");
  TraceEvent run;
  run.name = "model_run";
  run.category = Category::Session;
  run.duration_ns = 5000;
  t.events = {run, kernel("a", 0), kernel("a", 1), kernel("b", 1)};
  for (std::size_t i = 0; i < t.events.size(); ++i) t.events[i].sequence = i;
  return t;
}

ActivationRecord record(const std::string& node, std::size_t run, double mean) {
  ActivationRecord r;
  r.node = node;
  r.tensor = node + "_out";
  r.run_index = run;
  r.summary.mean = mean;
  r.summary.sampled_elements = 8;
  r.summary.full_tensor = true;
  return r;
}

}  // namespace

TEST_CASE("attach_activations matches node name and run") {
  auto t = profile_like();
  const auto r = attach_activations(
      t, {record("a", 1, 0.5), record("c", 0, 1.0), record("a", 1, 9.0), record("b", 3, 1.0)});
  CHECK(r.attached == 1);
  CHECK(r.unmatched == 2);
  CHECK(r.duplicates == 1);
  CHECK_FALSE(t.events[1].activation);
  REQUIRE(t.events[2].activation);
  CHECK(t.events[2].activation->mean == 0.5);
  CHECK(t.events[2].attributes.back() ==
        std::make_pair(std::string("activation_tensor"), std::string("a_out")));
  CHECK_FALSE(t.events[3].activation);
  CHECK(t.metadata.capabilities.activation_summaries);
  for (const auto& n : t.metadata.notes)
    CHECK(n.find("contain no tensor values") == std::string::npos);
  CHECK(t.warnings.size() == 2);
}

TEST_CASE("attach_activations without matches leaves the capability off") {
  auto t = profile_like();
  const auto r = attach_activations(t, {record("zzz", 0, 1.0)});
  CHECK(r.attached == 0);
  CHECK_FALSE(t.metadata.capabilities.activation_summaries);
  CHECK(t.metadata.notes.size() == 1);
}

TEST_CASE("attached summaries survive a JSONL round trip") {
  auto t = profile_like();
  attach_activations(t, {record("a", 0, 0.25)});
  std::stringstream ss;
  write_jsonl(ss, t.metadata, t.events);
  const auto back = read_jsonl(ss);
  CHECK(back.metadata.capabilities.activation_summaries);
  REQUIRE(back.events[1].activation);
  CHECK(back.events[1].activation->mean == 0.25);
  CHECK(back.events[1].activation->full_tensor);
}
