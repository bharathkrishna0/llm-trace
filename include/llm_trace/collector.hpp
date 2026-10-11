#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "llm_trace/event.hpp"
#include "llm_trace/ring_buffer.hpp"

namespace llm_trace {

struct CollectorStats {
  std::uint64_t capacity = 0;
  std::uint64_t retained = 0;
  std::uint64_t accepted = 0;
  std::uint64_t overwritten = 0;
  std::uint64_t rejected = 0;
  std::uint64_t version = 0;  // increments on every accepted event or state change
  bool capturing = false;
};

struct CollectorSnapshot {
  TraceMetadata metadata;
  std::vector<TraceEvent> events;  // oldest first
  CollectorStats stats;
  std::vector<std::string> rejection_reasons;  // most recent distinct reasons, bounded
};

// Thread-safe event sink. Validates events, assigns sequence numbers and stores
// them in a bounded ring buffer (overwrite-oldest). submit() never blocks on I/O or
// rendering; it takes one short mutex section.
class Collector {
 public:
  explicit Collector(std::size_t capacity, TraceMetadata metadata = {});

  // Assigns the next sequence number. Returns false if the event was rejected.
  bool submit(TraceEvent event);

  // Monotonic nanoseconds since this collector was created (steady_clock).
  std::uint64_t now_ns() const;

  void set_capturing(bool capturing);
  CollectorStats stats() const;
  CollectorSnapshot snapshot() const;
  TraceMetadata metadata() const;
  void add_note(std::string note);

 private:
  mutable std::mutex mutex_;
  RingBuffer<TraceEvent> buffer_;
  TraceMetadata metadata_;
  std::uint64_t next_sequence_ = 0;
  std::uint64_t rejected_ = 0;
  std::uint64_t version_ = 0;
  bool capturing_ = false;
  std::vector<std::string> rejection_reasons_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace llm_trace
