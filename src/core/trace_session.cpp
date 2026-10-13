#include "llm_trace/trace_session.hpp"

#include <algorithm>
#include <array>
#include <iterator>
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

// Longest total first; ties by key so the order never depends on hashing.
void sort_groups(std::vector<GroupStat>& groups) {
  std::sort(groups.begin(), groups.end(), [](const GroupStat& a, const GroupStat& b) {
    if (a.total_ns != b.total_ns) return a.total_ns > b.total_ns;
    return a.key < b.key;
  });
}

void add_to(GroupStat& group, std::uint64_t ns) {
  ++group.count;
  group.total_ns += ns;
  group.max_ns = std::max(group.max_ns, ns);
}

constexpr std::size_t kCategoryCount = static_cast<std::size_t>(Category::Other) + 1;

}  // namespace

TraceSession::TraceSession(TraceMetadata metadata, std::vector<TraceEvent> events,
                           std::optional<CollectorStats> live_stats)
    : metadata_(std::move(metadata)),
      events_(std::make_move_iterator(events.begin()), std::make_move_iterator(events.end())),
      live_stats_(live_stats) {
  index_sequences();
  compute_metrics();
}

void TraceSession::apply(CollectorDelta delta) {
  metadata_ = std::move(delta.metadata);
  live_stats_ = delta.stats;
  while (!events_.empty() && events_.front().sequence < delta.first_retained_sequence)
    events_.pop_front();
  for (auto& e : delta.events)
    if (events_.empty() || e.sequence > events_.back().sequence) events_.push_back(std::move(e));
  index_sequences();
  compute_metrics();
}

void TraceSession::index_sequences() {
  sequences_sorted_ = true;
  for (std::size_t i = 1; i < events_.size() && sequences_sorted_; ++i)
    sequences_sorted_ = events_[i - 1].sequence < events_[i].sequence;
  by_sequence_.clear();
  if (sequences_sorted_) return;
  by_sequence_.reserve(events_.size());
  for (std::size_t i = 0; i < events_.size(); ++i) by_sequence_.emplace(events_[i].sequence, i);
}

void TraceSession::compute_metrics() {
  SessionMetrics m;
  m.event_count = events_.size();
  std::array<GroupStat, kCategoryCount> categories{};
  std::unordered_map<std::string, std::size_t> name_index;
  std::vector<GroupStat> names;
  std::vector<std::size_t> timed_ops;
  std::uint64_t source_run_total = 0;
  std::uint64_t last_end = 0;

  for (std::size_t i = 0; i < events_.size(); ++i) {
    const auto& e = events_[i];
    if (e.type == EventType::Warning || e.type == EventType::AdapterError) ++m.warning_count;
    if (e.type == EventType::Operation && !e.duration_ns) ++m.missing_duration_count;

    if (e.type == EventType::Operation || e.type == EventType::RunStart ||
        e.type == EventType::RunEnd) {
      m.first_timestamp_ns =
          m.first_timestamp_ns ? std::min(*m.first_timestamp_ns, e.timestamp_ns) : e.timestamp_ns;
      last_end = std::max(last_end, e.timestamp_ns + e.duration_ns.value_or(0));
    }

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

    add_to(categories[static_cast<std::size_t>(e.category)], *e.duration_ns);

    const std::string& key = e.op_type ? *e.op_type : e.name;
    auto it = name_index.find(key);
    if (it == name_index.end()) {
      it = name_index.emplace(key, names.size()).first;
      names.push_back(GroupStat{key, 0, 0, 0});
    }
    add_to(names[it->second], *e.duration_ns);
  }

  if (m.first_timestamp_ns) m.observed_span_ns = last_end - *m.first_timestamp_ns;

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
  for (std::size_t c = 0; c < kCategoryCount; ++c) {
    if (categories[c].count == 0) continue;
    categories[c].key = to_string(static_cast<Category>(c));
    m.by_category.push_back(std::move(categories[c]));
  }
  sort_groups(m.by_category);
  sort_groups(names);
  m.by_name = std::move(names);
  metrics_ = std::move(m);
}

std::vector<std::size_t> TraceSession::query(const std::string& filter, SortMode sort) const {
  return query(EventFilter::parse(filter), sort);
}

std::vector<std::size_t> TraceSession::query(const EventFilter& filter, SortMode sort) const {
  std::vector<std::size_t> out;
  if (!filter.valid()) return out;
  out.reserve(events_.size());
  if (filter.empty()) {
    for (std::size_t i = 0; i < events_.size(); ++i) out.push_back(i);
  } else {
    for (std::size_t i = 0; i < events_.size(); ++i)
      if (filter.matches(events_[i])) out.push_back(i);
  }

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
  if (sequences_sorted_) {
    const auto it = std::lower_bound(
        events_.begin(), events_.end(), sequence,
        [](const TraceEvent& e, std::uint64_t value) { return e.sequence < value; });
    if (it == events_.end() || it->sequence != sequence) return std::nullopt;
    return static_cast<std::size_t>(it - events_.begin());
  }
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
