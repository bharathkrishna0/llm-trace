#pragma once

// Runtime adapter interfaces. Adapters are the only components that know about a
// specific runtime; the collector, storage and TUI see only TraceEvent.

#include <atomic>
#include <string>

#include "llm_trace/collector.hpp"
#include "llm_trace/event.hpp"

namespace llm_trace {

class TraceAdapter {
 public:
  virtual ~TraceAdapter() = default;
  virtual std::string name() const = 0;
  virtual AdapterCapabilities capabilities() const = 0;
};

// Pushes events into a Collector while a workload runs. run() is called on a
// worker thread and must return promptly once `stop` becomes true.
class LiveAdapter : public TraceAdapter {
 public:
  virtual TraceMetadata metadata() const = 0;
  virtual void run(Collector& collector, const std::atomic<bool>& stop) = 0;
};

}  // namespace llm_trace
