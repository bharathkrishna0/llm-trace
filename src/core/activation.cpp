#include "llm_trace/activation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace llm_trace {

ActivationSummary summarize_activation(const float* data, std::size_t count, double epsilon) {
  ActivationSummary s;
  s.sparsity_epsilon = epsilon;
  s.sampled_elements = count;
  s.full_tensor = true;
  // Two passes (sum, then squared deviations): accurate in double and avoids a
  // division per element.
  std::size_t finite = 0, small = 0;
  double sum = 0.0;
  double lo = std::numeric_limits<double>::infinity();
  double hi = -lo;
  for (std::size_t i = 0; i < count; ++i) {
    const double x = data[i];
    if (!std::isfinite(x)) {
      ++s.nonfinite_count;
      continue;
    }
    ++finite;
    sum += x;
    lo = std::min(lo, x);
    hi = std::max(hi, x);
    if (std::fabs(x) < epsilon) ++small;
  }
  if (finite == 0) return s;
  const double mean = sum / static_cast<double>(finite);
  double m2 = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double x = data[i];
    if (std::isfinite(x)) m2 += (x - mean) * (x - mean);
  }
  s.mean = mean;
  s.min = lo;
  s.max = hi;
  s.stddev = std::sqrt(m2 / static_cast<double>(finite));
  s.sparsity = static_cast<double>(small) / static_cast<double>(finite);
  return s;
}

namespace {

const std::string* attribute(const TraceEvent& e, const std::string& key) {
  for (const auto& [k, v] : e.attributes)
    if (k == key) return &v;
  return nullptr;
}

}  // namespace

AttachResult attach_activations(LoadedTrace& trace, const std::vector<ActivationRecord>& records) {
  AttachResult result;
  std::map<std::pair<std::string, std::string>, std::size_t> by_node_run;
  for (std::size_t i = 0; i < trace.events.size(); ++i) {
    const TraceEvent& e = trace.events[i];
    if (e.type != EventType::Operation || e.category == Category::Session) continue;
    if (const std::string* run = attribute(e, "run_index"))
      by_node_run.emplace(std::make_pair(e.name, *run), i);
  }
  for (const auto& r : records) {
    const auto it = by_node_run.find({r.node, std::to_string(r.run_index)});
    if (it == by_node_run.end()) {
      ++result.unmatched;
      continue;
    }
    TraceEvent& e = trace.events[it->second];
    if (e.activation) {
      ++result.duplicates;
      continue;
    }
    e.activation = r.summary;
    e.attributes.emplace_back("activation_tensor", r.tensor);
    ++result.attached;
  }
  if (result.attached > 0) {
    TraceMetadata& m = trace.metadata;
    m.capabilities.activation_summaries = true;
    for (auto it = m.notes.begin(); it != m.notes.end();)
      it = it->find("contain no tensor values") != std::string::npos ? m.notes.erase(it) : it + 1;
    m.notes.push_back(
        "Activation summaries were computed from tensor values ONNX Runtime returned, for " +
        std::to_string(result.attached) +
        " kernel event(s). Only nodes whose output was exposed as a graph output have one; "
        "exposing outputs can change which kernels ONNX Runtime fuses and how long they take. "
        "Attention weights are not captured.");
  }
  if (result.unmatched > 0)
    trace.warnings.push_back(std::to_string(result.unmatched) +
                             " activation summary record(s) had no kernel event with the same "
                             "node name and run, and were dropped.");
  if (result.duplicates > 0)
    trace.warnings.push_back(std::to_string(result.duplicates) +
                             " extra activation summary record(s) for an already summarised "
                             "event were dropped (one summary per event).");
  return result;
}

}  // namespace llm_trace
