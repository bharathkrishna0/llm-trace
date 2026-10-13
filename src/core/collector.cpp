#include "llm_trace/collector.hpp"

#include <algorithm>

namespace llm_trace {

namespace {
constexpr std::size_t kMaxRejectionReasons = 8;
}

Collector::Collector(std::size_t capacity, TraceMetadata metadata)
    : buffer_(capacity), metadata_(std::move(metadata)), start_(std::chrono::steady_clock::now()) {
  metadata_.buffer_capacity = capacity;
}

bool Collector::submit(TraceEvent event) {
  std::string reason = validate(event);
  TraceEvent evicted;  // destroyed after the lock is released
  std::lock_guard<std::mutex> lock(mutex_);
  if (!reason.empty()) {
    record_rejection_locked(std::move(reason));
    return false;
  }
  event.sequence = next_sequence_++;
  buffer_.push(std::move(event), &evicted);
  ++version_;
  return true;
}

std::size_t Collector::submit_batch(std::vector<TraceEvent> events) {
  std::vector<std::string> reasons(events.size());
  for (std::size_t i = 0; i < events.size(); ++i) reasons[i] = validate(events[i]);
  std::vector<TraceEvent> evicted(events.size());  // destroyed after the lock is released
  std::size_t accepted = 0;
  std::lock_guard<std::mutex> lock(mutex_);
  for (std::size_t i = 0; i < events.size(); ++i) {
    if (!reasons[i].empty()) {
      record_rejection_locked(std::move(reasons[i]));
      continue;
    }
    events[i].sequence = next_sequence_++;
    buffer_.push(std::move(events[i]), &evicted[i]);
    ++version_;
    ++accepted;
  }
  return accepted;
}

void Collector::record_rejection_locked(std::string reason) {
  ++rejected_;
  ++version_;
  if (std::find(rejection_reasons_.begin(), rejection_reasons_.end(), reason) ==
      rejection_reasons_.end()) {
    if (rejection_reasons_.size() == kMaxRejectionReasons)
      rejection_reasons_.erase(rejection_reasons_.begin());
    rejection_reasons_.push_back(std::move(reason));
  }
}

std::uint64_t Collector::now_ns() const {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - start_)
                                        .count());
}

void Collector::set_capturing(bool capturing) {
  std::lock_guard<std::mutex> lock(mutex_);
  capturing_ = capturing;
  ++version_;
}

CollectorStats Collector::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_locked();
}

CollectorStats Collector::stats_locked() const {
  CollectorStats stats;
  stats.capacity = buffer_.capacity();
  stats.retained = buffer_.size();
  stats.accepted = buffer_.pushed();
  stats.overwritten = buffer_.overwritten();
  stats.rejected = rejected_;
  stats.version = version_;
  stats.capturing = capturing_;
  return stats;
}

CollectorSnapshot Collector::snapshot() const {
  CollectorSnapshot snap;
  std::lock_guard<std::mutex> lock(mutex_);
  snap.events = buffer_.snapshot();
  snap.metadata = metadata_;
  snap.metadata.overwritten_events = buffer_.overwritten();
  snap.metadata.rejected_events = rejected_;
  snap.stats = stats_locked();
  snap.rejection_reasons = rejection_reasons_;
  return snap;
}

CollectorDelta Collector::delta_since(std::uint64_t sequence) const {
  CollectorDelta delta;
  std::lock_guard<std::mutex> lock(mutex_);
  // Accepted events get consecutive sequences and are all pushed, so the buffer holds
  // the last size() sequences.
  delta.first_retained_sequence = next_sequence_ - buffer_.size();
  const std::uint64_t from = std::max(sequence, delta.first_retained_sequence);
  if (from < next_sequence_) {
    const auto start = static_cast<std::size_t>(from - delta.first_retained_sequence);
    delta.events.reserve(buffer_.size() - start);
    for (std::size_t i = start; i < buffer_.size(); ++i) delta.events.push_back(buffer_.at(i));
  }
  delta.metadata = metadata_;
  delta.metadata.overwritten_events = buffer_.overwritten();
  delta.metadata.rejected_events = rejected_;
  delta.stats = stats_locked();
  return delta;
}

TraceMetadata Collector::metadata() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return metadata_;
}

void Collector::add_note(std::string note) {
  std::lock_guard<std::mutex> lock(mutex_);
  metadata_.notes.push_back(std::move(note));
  ++version_;
}

}  // namespace llm_trace
