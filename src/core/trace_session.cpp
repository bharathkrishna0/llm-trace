#include "llm_trace/trace_session.hpp"

#include <algorithm>
#include <map>
#include <unordered_set>

#include "llm_trace/metrics.hpp"

namespace llm_trace {

const char* to_string(SortMode mode) {
  switch (mode) {
    case SortMode::Source:
      return "source order";
    case SortMode::DurationDesc:
      return "duration";
    case SortMode::Name:
      return "name";
  }
  return "source order";
}

namespace {

constexpr std::size_t kTopOperations = 10;

bool is_aggregated_operation(const TraceEvent& event) {
  return event.type == EventType::Operation && event.category != Category::Session &&
         event.duration_ns.has_value();
}

bool is_source_run_event(const TraceEvent& event) {
  if (event.type != EventType::Operation || event.category != Category::Session) return false;
  if (!event.duration_ns) return false;
  return event.name == "model_run" || event.name == "decode.step" || event.name == "prefill";
}

std::vector<GroupStat> sorted_groups(const std::map<std::string, GroupStat>& groups) {
  std::vector<GroupStat> out;
  out.reserve(groups.size());
  for (const auto& entry : groups) out.push_back(entry.second);
  std::stable_sort(out.begin(), out.end(),
                   [](const GroupStat& a, const GroupStat& b) { return a.total_ns > b.total_ns; });
  return out;
}

}  // namespace

TraceSession::TraceSession(TraceMetadata metadata, std::vector<TraceEvent> events,
                           std::optional<CollectorStats> live_stats)
    : metadata_(std::move(metadata)), events_(std::move(events)), live_stats_(live_stats) {
  by_sequence_.reserve(events_.size());
  for (std::size_t i = 0; i < events_.size(); ++i) by_sequence_.emplace(events_[i].sequence, i);
  compute_metrics();
}

void TraceSession::compute_metrics() {
  SessionMetrics m;
  m.event_count = events_.size();
  std::map<std::string, GroupStat> categories;
  std::map<std::string, GroupStat> names;
  std::vector<std::size_t> timed_ops;
  std::uint64_t source_run_total = 0;

  for (std::size_t i = 0; i < events_.size(); ++i) {
    const auto& e = events_[i];
    if (e.type == EventType::Warning || e.type == EventType::AdapterError) ++m.warning_count;
    if (e.type == EventType::Operation && !e.duration_ns) ++m.missing_duration_count;

    if (e.type == EventType::Operation || e.type == EventType::RunStart ||
        e.type == EventType::RunEnd)
      m.first_timestamp_ns =
          m.first_timestamp_ns ? std::min(*m.first_timestamp_ns, e.timestamp_ns) : e.timestamp_ns;

    if (e.type == EventType::Operation && e.category == Category::Session) {
      ++m.session_event_count;
      if (is_source_run_event(e)) {
        source_run_total += *e.duration_ns;
        ++m.source_run_count;
      }
      continue;
    }
    if (!is_aggregated_operation(e)) continue;

    ++m.operation_count;
    m.summed_operation_ns += *e.duration_ns;
    timed_ops.push_back(i);

    auto& cat = categories[to_string(e.category)];
    cat.key = to_string(e.category);
    ++cat.count;
    cat.total_ns += *e.duration_ns;
    cat.max_ns = std::max(cat.max_ns, *e.duration_ns);

    const std::string key = e.op_type ? *e.op_type : e.name;
    auto& group = names[key];
    group.key = key;
    ++group.count;
    group.total_ns += *e.duration_ns;
    group.max_ns = std::max(group.max_ns, *e.duration_ns);
  }

  if (m.first_timestamp_ns) {
    std::uint64_t last_end = *m.first_timestamp_ns;
    for (const auto& e : events_) {
      if (e.type == EventType::Operation || e.type == EventType::RunStart ||
          e.type == EventType::RunEnd)
        last_end = std::max(last_end, e.timestamp_ns + e.duration_ns.value_or(0));
    }
    m.observed_span_ns = last_end - *m.first_timestamp_ns;
  }

  if (m.source_run_count > 0) m.source_run_total_ns = source_run_total;

  const std::size_t top = std::min(kTopOperations, timed_ops.size());
  std::partial_sort(timed_ops.begin(), timed_ops.begin() + static_cast<std::ptrdiff_t>(top),
                    timed_ops.end(), [this](std::size_t a, std::size_t b) {
                      if (*events_[a].duration_ns != *events_[b].duration_ns)
                        return *events_[a].duration_ns > *events_[b].duration_ns;
                      return a < b;
                    });
  timed_ops.resize(top);
  m.top_operations = std::move(timed_ops);
  m.by_category = sorted_groups(categories);
  m.by_name = sorted_groups(names);
  metrics_ = std::move(m);
}

bool TraceSession::matches(const TraceEvent& event, const std::string& lowered_filter) {
  if (lowered_filter.empty()) return true;
  auto contains = [&](const std::string& text) {
    return to_lower(text).find(lowered_filter) != std::string::npos;
  };
  if (contains(event.name)) return true;
  if (contains(to_string(event.category))) return true;
  if (contains(to_string(event.type))) return true;
  if (event.op_type && contains(*event.op_type)) return true;
  if (event.layer_index && (std::to_string(*event.layer_index) == lowered_filter ||
                            "layer " + std::to_string(*event.layer_index) == lowered_filter))
    return true;
  return false;
}

std::vector<std::size_t> TraceSession::query(const std::string& filter, SortMode sort) const {
  const std::string lowered = to_lower(filter);
  std::vector<std::size_t> out;
  out.reserve(events_.size());
  for (std::size_t i = 0; i < events_.size(); ++i)
    if (matches(events_[i], lowered)) out.push_back(i);

  switch (sort) {
    case SortMode::Source:
      break;
    case SortMode::DurationDesc:
      std::stable_sort(out.begin(), out.end(), [this](std::size_t a, std::size_t b) {
        // Events without a duration sort last.
        const auto& da = events_[a].duration_ns;
        const auto& db = events_[b].duration_ns;
        if (da.has_value() != db.has_value()) return da.has_value();
        return da.value_or(0) > db.value_or(0);
      });
      break;
    case SortMode::Name:
      std::stable_sort(out.begin(), out.end(), [this](std::size_t a, std::size_t b) {
        return events_[a].name < events_[b].name;
      });
      break;
  }
  return out;
}

std::optional<std::size_t> TraceSession::find_sequence(std::uint64_t sequence) const {
  auto it = by_sequence_.find(sequence);
  if (it == by_sequence_.end()) return std::nullopt;
  return it->second;
}

std::vector<std::size_t> TraceSession::children_of(const std::string& parent_id) const {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < events_.size(); ++i)
    if (events_[i].parent_id && *events_[i].parent_id == parent_id) out.push_back(i);
  return out;
}

std::vector<std::string> TraceSession::warnings() const {
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  auto add = [&](const std::string& text) {
    if (!text.empty() && seen.insert(text).second) out.push_back(text);
  };
  for (const auto& note : metadata_.notes) add(note);
  for (const auto& e : events_) {
    if (e.type == EventType::Warning || e.type == EventType::AdapterError)
      add(e.message.empty() ? e.name : e.message);
  }
  return out;
}

}  // namespace llm_trace
