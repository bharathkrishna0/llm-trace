#pragma once

// Activation summaries computed from real tensor values (M9). The values come from a
// runtime that returned the tensors, e.g. examples/onnx/onnx_profile_runner with a
// model whose intermediate tensors were exposed as graph outputs. Nothing here
// estimates values that were not observed.

#include <cstddef>
#include <string>
#include <vector>

#include "llm_trace/event.hpp"
#include "llm_trace/trace_io.hpp"

namespace llm_trace {

// Statistics over every element. mean/min/max/stddev (population) cover finite
// values only; NaN and +-Inf are counted in nonfinite_count. sparsity is the share
// of finite values with |x| < epsilon. With no finite values, those fields are 0.
ActivationSummary summarize_activation(const float* data, std::size_t count, double epsilon);

struct ActivationRecord {
  std::size_t run_index = 0;  // 0-based Run() call in the profiled session
  std::string node;           // ONNX node name that produced the tensor
  std::string tensor;         // tensor (graph output) name
  ActivationSummary summary;
};

struct AttachResult {
  std::size_t attached = 0;
  std::size_t unmatched = 0;   // records with no kernel event for (node, run)
  std::size_t duplicates = 0;  // records for an event that already had a summary
};

// Attaches each record to the kernel event with the same node name and run_index
// attribute. Sets the activation_summaries capability and notes how the values
// were obtained; unmatched records become a warning.
AttachResult attach_activations(LoadedTrace& trace, const std::vector<ActivationRecord>& records);

}  // namespace llm_trace
