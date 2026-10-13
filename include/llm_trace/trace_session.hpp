#pragma once

// Query layer between stored events and the UI. Owns filtering, sorting and
// aggregation so that the TUI never computes these inside render functions.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "llm_trace/collector.hpp"
#include "llm_trace/event.hpp"
#include "llm_trace/event_filter.hpp"

namespace llm_trace {

enum class SortMode { Source, DurationDesc, Name };
const char* to_string(SortMode mode);

struct GroupStat {
  std::string key;
  std::uint64_t count = 0;
  std::uint64_t total_ns = 0;
  std::uint64_t max_ns = 0;
};

// All aggregates use operation events (EventType::Operation) that carry a
// duration and are not in Category::Session. See docs/architecture/event-schema.md.
struct SessionMetrics {
  std::size_t event_count = 0;
  std::size_t operation_count = 0;  // operations included in the aggregates
  std::size_t session_event_count = 0;
  std::size_t warning_count = 0;  // warning + adapter_error events
  std::size_t missing_duration_count = 0;
  // Earliest timestamp to latest (timestamp + duration) over all timed events.
  std::optional<std::uint64_t> observed_span_ns;
  std::optional<std::uint64_t> first_timestamp_ns;
  // Sum of inclusive operation durations. Operations can overlap (threads) so this
  // is not end-to-end latency.
  std::uint64_t summed_operation_ns = 0;
  // Sum of durations reported by the source's own run-level events (for example the
  // ONNX Runtime "model_run" record), when present.
  std::optional<std::uint64_t> source_run_total_ns;
  std::size_t source_run_count = 0;
  std::vector<std::size_t> top_operations;  // indices into events(), longest first
  std::vector<GroupStat> by_category;       // sorted by total_ns descending
  std::vector<GroupStat> by_name;           // grouped by op_type if present, else name
};

class TraceSession {
 public:
  TraceSession() = default;
  TraceSession(TraceMetadata metadata, std::vector<TraceEvent> events,
               std::optional<CollectorStats> live_stats = std::nullopt);

  const TraceMetadata& metadata() const { return metadata_; }
  // Oldest first. A deque so live updates can drop overwritten events from the front.
  const std::deque<TraceEvent>& events() const { return events_; }
  const SessionMetrics& metrics() const { return metrics_; }
  const std::optional<CollectorStats>& live_stats() const { return live_stats_; }
  bool empty() const { return events_.empty(); }

  // Live update from a Collector: drops events older than
  // delta.first_retained_sequence, appends the new ones and recomputes metrics. Costs
  // O(retained) for the metrics pass but copies only the new events.
  void apply(CollectorDelta delta);

  // Returns indices into events() that match `filter` (see event_filter.hpp), sorted.
  // An invalid filter matches nothing. Never mutates the stored events.
  std::vector<std::size_t> query(const EventFilter& filter, SortMode sort) const;
  std::vector<std::size_t> query(const std::string& filter, SortMode sort) const;

  std::optional<std::size_t> find_sequence(std::uint64_t sequence) const;
  std::vector<std::size_t> children_of(const std::string& parent_id) const;
  // Distinct warning/adapter_error messages plus metadata notes, in first-seen order.
  std::vector<std::string> warnings() const;

 private:
  void index_sequences();
  void compute_metrics();

  TraceMetadata metadata_;
  std::deque<TraceEvent> events_;
  std::optional<CollectorStats> live_stats_;
  SessionMetrics metrics_;
  // Sources normally emit strictly increasing sequences, so lookups binary-search
  // events_. The map is built only for out-of-order input.
  bool sequences_sorted_ = true;
  std::unordered_map<std::uint64_t, std::size_t> by_sequence_;
};

}  // namespace llm_trace
