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

// Events added since a given sequence number, for incremental viewers.
struct CollectorDelta {
  TraceMetadata metadata;
  std::vector<TraceEvent> events;  // sequence >= the requested one, oldest first
  // Oldest sequence still retained (equal to the next sequence when the buffer is empty).
  // Viewers drop anything older: it has been overwritten.
  std::uint64_t first_retained_sequence = 0;
  CollectorStats stats;
};

// Thread-safe event sink. Validates events, assigns sequence numbers and stores
// them in a bounded ring buffer (overwrite-oldest). submit() never blocks on I/O or
// rendering; it takes one short mutex section.
class Collector {
 public:
  explicit Collector(std::size_t capacity, TraceMetadata metadata = {});

  // Assigns the next sequence number. Returns false if the event was rejected.
  bool submit(TraceEvent event);
  // Validates outside the lock and stores the whole batch under one lock acquisition.
  // Same per-event rules as submit(). Returns the number of events accepted.
  std::size_t submit_batch(std::vector<TraceEvent> events);

  // Monotonic nanoseconds since this collector was created (steady_clock).
  std::uint64_t now_ns() const;

  void set_capturing(bool capturing);
  CollectorStats stats() const;
  CollectorSnapshot snapshot() const;
  // Copies only the retained events with sequence >= `sequence`, so the lock is held
  // for O(new events) rather than O(capacity).
  CollectorDelta delta_since(std::uint64_t sequence) const;
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
  CollectorStats stats_locked() const;
  void record_rejection_locked(std::string reason);
  std::vector<std::string> rejection_reasons_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace llm_trace
