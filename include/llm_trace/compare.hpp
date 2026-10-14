#pragma once

// Side-by-side comparison of two loaded traces (`llm-trace --compare A B`).
//
// Only timed operations are compared (type operation, category other than session,
// with a duration). Rows compare the mean time per call, so a trace with more runs or
// tokens is not reported as slower merely for being longer. Matching is by key only:
// the comparison cannot know whether two ops with the same name did the same work, and
// it does not test changes for statistical significance. Warnings say when the two
// traces come from different sources, models, runtimes or devices.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "llm_trace/trace_io.hpp"

namespace llm_trace {

enum class CompareKey {
  Name,    // event name (node name for ONNX Runtime): exact matching within one model
  OpType,  // op type when present, else name: coarser, works across models
};

struct CompareOptions {
  CompareKey key = CompareKey::Name;
  std::size_t min_calls = 5;  // rows with fewer calls on a side are marked
};

struct SideStat {
  std::size_t count = 0;
  std::uint64_t total_ns = 0;
  double mean_ns = 0.0;
  double median_ns = 0.0;
  double share = 0.0;  // total_ns / summed operation time of that trace
};

struct CompareRow {
  std::string key;
  std::optional<SideStat> a;  // absent: the key does not occur in that trace
  std::optional<SideStat> b;

  // (B mean - A mean) / A mean. Empty unless both sides have calls and A's mean is > 0.
  std::optional<double> mean_change() const;
  bool few_samples(std::size_t min_calls) const;
};

struct CompareSide {
  std::string label;
  TraceMetadata metadata;
  std::size_t events = 0;
  std::size_t operations = 0;        // timed operations compared
  std::size_t missing_duration = 0;  // operations skipped because they have no duration
  std::uint64_t summed_ns = 0;
  std::vector<std::string> devices;
};

struct TraceComparison {
  CompareOptions options;
  CompareSide a;
  CompareSide b;
  std::vector<CompareRow> by_category;  // category order
  std::vector<CompareRow> by_layer;     // layer order, "no layer" last
  std::vector<CompareRow> by_op;        // in both traces, largest |mean change| first
  std::vector<CompareRow> only_in_a;    // largest total first
  std::vector<CompareRow> only_in_b;
  std::vector<std::string> warnings;
};

TraceComparison compare_traces(const LoadedTrace& a, const std::string& label_a,
                               const LoadedTrace& b, const std::string& label_b,
                               const CompareOptions& options = {});

struct CompareTextOptions {
  std::size_t top = 20;  // rows per operation table
  bool ascii = false;
};

std::string format_comparison(const TraceComparison& comparison,
                              const CompareTextOptions& options = {});
// Machine-readable form ("schema": "llm-trace-compare/1"); all durations in ns.
std::string comparison_to_json(const TraceComparison& comparison);

}  // namespace llm_trace
