#pragma once

// Runtime-independent trace event model.
//
// Units and conventions (see docs/architecture/event-schema.md):
//   * timestamp_ns is a monotonic offset in nanoseconds from the start of the run.
//   * duration_ns is inclusive: it covers the operation and any nested child work.
//   * std::nullopt means "not present in this trace"; zero is always a real value.

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llm_trace {

inline constexpr int kSchemaVersion = 1;

enum class EventType {
  RunStart,
  RunEnd,
  Operation,
  TensorMetadata,
  ActivationSummary,
  AttentionSummary,
  Warning,
  AdapterError,
  DroppedEvents,
};

enum class Category {
  Unknown,
  Session,
  Embedding,
  Attention,
  Mlp,
  Normalization,
  Activation,
  MatMul,
  Elementwise,
  Shape,
  Output,
  Other,
};

const char* to_string(EventType type);
const char* to_string(Category category);
std::optional<EventType> parse_event_type(const std::string& text);
std::optional<Category> parse_category(const std::string& text);

struct TensorMetadata {
  std::string name;
  std::optional<std::vector<std::int64_t>> shape;  // -1 marks a dynamic dimension
  std::optional<std::string> dtype;
  std::optional<std::uint64_t> element_count;
  std::optional<std::uint64_t> byte_count;
};

// Summary statistics computed from captured values. Only adapters that can read
// tensor values may populate this; no current adapter does.
struct ActivationSummary {
  double mean = 0.0;
  double min = 0.0;
  double max = 0.0;
  double stddev = 0.0;
  std::uint64_t nonfinite_count = 0;
  double sparsity = 0.0;  // fraction of values with |x| < sparsity_epsilon
  double sparsity_epsilon = 0.0;
  std::uint64_t sampled_elements = 0;
  bool full_tensor = false;  // false means the statistics describe a sample
};

struct AdapterCapabilities {
  bool operation_timing = false;
  bool tensor_metadata = false;
  bool layer_hierarchy = false;
  bool token_index = false;
  bool activation_summaries = false;
  bool attention_values = false;
};

struct TraceEvent {
  std::uint64_t sequence = 0;
  std::uint64_t timestamp_ns = 0;
  std::optional<std::uint64_t> duration_ns;
  EventType type = EventType::Operation;
  std::string name;
  Category category = Category::Unknown;
  std::optional<std::string> op_type;
  std::optional<std::string> parent_id;
  std::optional<std::int64_t> layer_index;
  std::optional<std::int64_t> token_index;
  std::optional<std::int64_t> head_index;
  std::optional<std::string> device;
  std::optional<std::int64_t> thread_id;
  std::vector<TensorMetadata> inputs;
  std::vector<TensorMetadata> outputs;
  std::optional<ActivationSummary> activation;
  std::string message;  // warning/error text; empty for operations
  // Source-specific key/value pairs preserved for debugging (e.g. raw ONNX names).
  std::vector<std::pair<std::string, std::string>> attributes;
};

enum class DataSource { None, Synthetic, OnnxProfile, Replay };
const char* to_string(DataSource source);
std::optional<DataSource> parse_data_source(const std::string& text);

struct TraceMetadata {
  int schema_version = kSchemaVersion;
  std::string run_id;
  DataSource source = DataSource::None;
  // Source of the events before they were saved; a replayed synthetic trace keeps
  // origin = Synthetic so it is still labelled DEMO.
  DataSource origin = DataSource::None;
  std::string adapter;
  std::string runtime;
  std::string runtime_version;
  std::string model;
  std::string created_utc;  // human-readable wall-clock time; never used for durations
  AdapterCapabilities capabilities;
  std::uint64_t buffer_capacity = 0;
  std::uint64_t overwritten_events = 0;
  std::uint64_t rejected_events = 0;
  std::vector<std::string> notes;
};

// Returns an empty string if the event is valid, otherwise a short reason.
std::string validate(const TraceEvent& event);

std::uint64_t element_count_of(const std::vector<std::int64_t>& shape);
std::optional<std::uint64_t> dtype_size_bytes(const std::string& dtype);

}  // namespace llm_trace
