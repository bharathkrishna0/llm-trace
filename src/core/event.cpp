#include "llm_trace/event.hpp"

#include <array>
#include <limits>

namespace llm_trace {

namespace {

struct EventTypeName {
  EventType type;
  const char* name;
};

constexpr std::array<EventTypeName, 9> kEventTypes{{
    {EventType::RunStart, "run_start"},
    {EventType::RunEnd, "run_end"},
    {EventType::Operation, "operation"},
    {EventType::TensorMetadata, "tensor_metadata"},
    {EventType::ActivationSummary, "activation_summary"},
    {EventType::AttentionSummary, "attention_summary"},
    {EventType::Warning, "warning"},
    {EventType::AdapterError, "adapter_error"},
    {EventType::DroppedEvents, "dropped_events"},
}};

struct CategoryName {
  Category category;
  const char* name;
};

constexpr std::array<CategoryName, 12> kCategories{{
    {Category::Unknown, "unknown"},
    {Category::Session, "session"},
    {Category::Embedding, "embedding"},
    {Category::Attention, "attention"},
    {Category::Mlp, "mlp"},
    {Category::Normalization, "normalization"},
    {Category::Activation, "activation"},
    {Category::MatMul, "matmul"},
    {Category::Elementwise, "elementwise"},
    {Category::Shape, "shape"},
    {Category::Output, "output"},
    {Category::Other, "other"},
}};

struct SourceName {
  DataSource source;
  const char* name;
};

constexpr std::array<SourceName, 4> kSources{{
    {DataSource::None, "none"},
    {DataSource::Synthetic, "synthetic"},
    {DataSource::OnnxProfile, "onnx_profile"},
    {DataSource::Replay, "replay"},
}};

}  // namespace

const char* to_string(EventType type) {
  for (const auto& entry : kEventTypes)
    if (entry.type == type) return entry.name;
  return "operation";
}

const char* to_string(Category category) {
  for (const auto& entry : kCategories)
    if (entry.category == category) return entry.name;
  return "unknown";
}

const char* to_string(DataSource source) {
  for (const auto& entry : kSources)
    if (entry.source == source) return entry.name;
  return "none";
}

std::optional<EventType> parse_event_type(const std::string& text) {
  for (const auto& entry : kEventTypes)
    if (text == entry.name) return entry.type;
  return std::nullopt;
}

std::optional<Category> parse_category(const std::string& text) {
  for (const auto& entry : kCategories)
    if (text == entry.name) return entry.category;
  return std::nullopt;
}

std::optional<DataSource> parse_data_source(const std::string& text) {
  for (const auto& entry : kSources)
    if (text == entry.name) return entry.source;
  return std::nullopt;
}

std::string validate(const TraceEvent& event) {
  if (event.name.empty()) return "event name is empty";
  if (event.type == EventType::Operation && !event.duration_ns)
    return {};  // allowed: duration may be unavailable, shown as such
  if (event.duration_ns &&
      event.timestamp_ns > std::numeric_limits<std::uint64_t>::max() - *event.duration_ns)
    return "timestamp + duration overflows";
  for (const auto* tensors : {&event.inputs, &event.outputs}) {
    for (const auto& tensor : *tensors) {
      if (!tensor.shape) continue;
      for (auto dim : *tensor.shape)
        if (dim < -1) return "tensor '" + tensor.name + "' has a negative dimension";
    }
  }
  return {};
}

std::uint64_t element_count_of(const std::vector<std::int64_t>& shape) {
  std::uint64_t count = 1;
  for (auto dim : shape) {
    if (dim < 0) return 0;
    const auto udim = static_cast<std::uint64_t>(dim);
    if (udim != 0 && count > std::numeric_limits<std::uint64_t>::max() / udim) return 0;
    count *= udim;
  }
  return count;
}

std::optional<std::uint64_t> dtype_size_bytes(const std::string& dtype) {
  if (dtype == "float32" || dtype == "float" || dtype == "int32" || dtype == "uint32") return 4;
  if (dtype == "float16" || dtype == "bfloat16" || dtype == "int16" || dtype == "uint16") return 2;
  if (dtype == "float64" || dtype == "double" || dtype == "int64" || dtype == "uint64") return 8;
  if (dtype == "int8" || dtype == "uint8" || dtype == "bool") return 1;
  return std::nullopt;
}

}  // namespace llm_trace
