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
  std::lock_guard<std::mutex> lock(mutex_);
  if (!reason.empty()) {
    ++rejected_;
    ++version_;
    if (std::find(rejection_reasons_.begin(), rejection_reasons_.end(), reason) ==
        rejection_reasons_.end()) {
      if (rejection_reasons_.size() == kMaxRejectionReasons)
        rejection_reasons_.erase(rejection_reasons_.begin());
      rejection_reasons_.push_back(std::move(reason));
    }
    return false;
  }
  event.sequence = next_sequence_++;
  buffer_.push(std::move(event));
  ++version_;
  return true;
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
  snap.stats.capacity = buffer_.capacity();
  snap.stats.retained = buffer_.size();
  snap.stats.accepted = buffer_.pushed();
  snap.stats.overwritten = buffer_.overwritten();
  snap.stats.rejected = rejected_;
  snap.stats.version = version_;
  snap.stats.capturing = capturing_;
  snap.rejection_reasons = rejection_reasons_;
  return snap;
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
