#pragma once

// ONNX Runtime profile-file adapter. Parses the Chrome-trace JSON written by
// Ort::SessionOptions::EnableProfiling. Tested against ONNX Runtime 1.20.1; see
// docs/adapters/onnxruntime.md for the supported fields and limitations.

#include <istream>
#include <string>

#include "llm_trace/adapter.hpp"
#include "llm_trace/trace_io.hpp"

namespace llm_trace {

class OnnxProfileAdapter : public TraceAdapter {
 public:
  std::string name() const override { return "onnxruntime-profile"; }
  AdapterCapabilities capabilities() const override;
  LoadedTrace load(const std::string& path, const ReadLimits& limits = {}) const;
  LoadedTrace parse(std::istream& in, const std::string& source_name,
                    const ReadLimits& limits = {}) const;
};

// Maps an ONNX operator type (e.g. "MatMul") to a coarse category. This is a fixed
// lookup table, not an inference about the model's architecture.
Category category_for_onnx_op(const std::string& op_type);

// Layer number from node names such as "/layers.3/attn/MatMul" or "h.3.mlp";
// nullopt when the name has no such component. ONNX Runtime itself has no layers.
std::optional<std::int64_t> layer_from_node_name(const std::string& name);

}  // namespace llm_trace
