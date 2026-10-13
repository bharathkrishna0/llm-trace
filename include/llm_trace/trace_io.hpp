#pragma once

// Versioned JSON Lines import/export (schema: docs/architecture/event-schema.md).
// Trace files are untrusted input: sizes, types and required fields are validated
// before allocation.

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

#include "llm_trace/event.hpp"

namespace llm_trace {

struct ReadLimits {
  std::uint64_t max_file_bytes = 1ull << 30;  // 1 GiB
  std::size_t max_line_bytes = 1u << 20;      // 1 MiB per JSONL record
  std::size_t max_events = 2'000'000;
  std::size_t max_string_bytes = 4096;
  std::size_t max_tensors_per_event = 64;
  std::size_t max_shape_rank = 16;
  std::size_t max_attributes = 64;
};

struct LoadedTrace {
  TraceMetadata metadata;
  std::vector<TraceEvent> events;
  std::vector<std::string> warnings;  // non-fatal import issues
};

class TraceIoError : public std::runtime_error {
 public:
  TraceIoError(const std::string& message, std::size_t line = 0)
      : std::runtime_error(line ? "line " + std::to_string(line) + ": " + message : message),
        line_(line) {}
  std::size_t line() const { return line_; }

 private:
  std::size_t line_;
};

enum class TraceFormat { LlmTraceJsonl, OnnxProfile, Unknown };
TraceFormat detect_format(const std::string& path);

std::string serialize_metadata(const TraceMetadata& metadata);
std::string serialize_event(const TraceEvent& event);

void write_jsonl(std::ostream& out, const TraceMetadata& metadata,
                 const std::vector<TraceEvent>& events);
// Writes to "<path>.tmp" and renames over <path> only after a successful flush.
void write_jsonl(const std::string& path, const TraceMetadata& metadata,
                 const std::vector<TraceEvent>& events);

LoadedTrace read_jsonl(std::istream& in, const ReadLimits& limits = {});
LoadedTrace read_jsonl(const std::string& path, const ReadLimits& limits = {});

// Opens either an LLM Trace JSONL file or an ONNX Runtime profile, by content.
LoadedTrace load_trace(const std::string& path, const ReadLimits& limits = {});

}  // namespace llm_trace
