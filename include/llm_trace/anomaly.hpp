#pragma once

// Configurable anomaly rules (FR-30). A rule match is a heuristic flag ("this op is
// unusually slow relative to its peers"), never a measured finding or a cause.
//
// Rule kinds:
//   p95_multiple  duration > factor x p95 of the same key: event name (or op type with
//                 group "op") plus input shapes when recorded; keys with fewer than
//                 min_calls timed calls are skipped
//   share         one event's duration > `above` of the summed operation time
//   match         any event matching `filter` (event_filter.hpp syntax), e.g. "dur>5ms"
// Every rule may carry a `scope` filter that limits the events it looks at. Rules that
// need a duration ignore events without one.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "llm_trace/event.hpp"

namespace llm_trace {

struct AnomalyRule {
  enum class Kind { P95Multiple, Share, Match };
  std::string id;
  Kind kind = Kind::Match;
  double factor = 3.0;         // p95_multiple
  std::size_t min_calls = 20;  // p95_multiple
  bool group_by_op = false;    // p95_multiple: group by op type instead of event name
  // p95_multiple: events shorter than this are never flagged (timer resolution noise;
  // ONNX Runtime profiles report whole microseconds).
  std::uint64_t min_ns = 10'000;
  double above = 0.25;  // share, as a fraction of summed operation time
  std::string filter;   // match
  std::string scope;    // optional filter applied before the rule
};

struct AnomalyHit {
  std::size_t event_index = 0;  // index into the evaluated events
  std::string rule_id;
  std::string detail;  // e.g. "1.20 ms = 4.1x p95 of MatMul (0.29 ms, 340 calls)"
};

struct AnomalyReport {
  std::vector<AnomalyHit> hits;  // event order, then rule order
  std::size_t skipped_missing_duration = 0;
};

// Built-in rules used when no --rules file is given.
std::vector<AnomalyRule> default_rules();

// Parses {"rules": [{"id": ..., "kind": "p95_multiple"|"share"|"match", ...}]}.
// Throws std::runtime_error naming the rule and field on any invalid input.
std::vector<AnomalyRule> parse_rules_json(const std::string& text);
std::vector<AnomalyRule> load_rules_file(const std::string& path);

std::string describe_rule(const AnomalyRule& rule, bool ascii = false);

AnomalyReport evaluate_rules(const std::vector<AnomalyRule>& rules,
                             const std::vector<TraceEvent>& events);
AnomalyReport evaluate_rules(const std::vector<AnomalyRule>& rules,
                             const std::deque<TraceEvent>& events);

// Plain-text --check report: the rules, then up to `top` flagged events.
std::string format_check_report(const std::string& label, const TraceMetadata& metadata,
                                const std::vector<TraceEvent>& events,
                                const std::vector<AnomalyRule>& rules, const AnomalyReport& report,
                                std::size_t top = 20);

}  // namespace llm_trace
