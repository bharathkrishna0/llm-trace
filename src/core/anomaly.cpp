#include "llm_trace/anomaly.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "llm_trace/event_filter.hpp"
#include "llm_trace/metrics.hpp"

namespace llm_trace {

namespace {

constexpr std::size_t kMaxRules = 64;

bool is_timed_op(const TraceEvent& e) {
  return e.type == EventType::Operation && e.category != Category::Session &&
         e.duration_ns.has_value();
}

// Like-for-like key: the op name (or type) plus its input shapes when the source
// records them, so a 16-token prefill call is not compared with 1-token decode calls.
std::string group_key(const TraceEvent& e, bool by_op) {
  std::string key = by_op && e.op_type ? *e.op_type : e.name;
  bool first = true;
  for (const auto& t : e.inputs) {
    if (!t.shape) continue;
    key += first ? " [" : ", ";
    first = false;
    for (std::size_t i = 0; i < t.shape->size(); ++i)
      key += (i ? "x" : "") + std::to_string((*t.shape)[i]);
  }
  if (!first) key += "]";
  return key;
}

std::string fixed(double v, int digits) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*f", digits, v);
  return buf;
}

EventFilter checked_filter(const std::string& text, const std::string& where) {
  EventFilter f = EventFilter::parse(text);
  if (!f.valid()) throw std::runtime_error(where + ": " + f.error());
  return f;
}

// 64-bit FNV-1a over the like-for-like key, so grouping does not build a string per event.
std::uint64_t group_hash(const TraceEvent& e, bool by_op) {
  std::uint64_t h = 1469598103934665603ull;
  auto mix = [&h](const void* data, std::size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  };
  const std::string& name = by_op && e.op_type ? *e.op_type : e.name;
  mix(name.data(), name.size());
  for (const auto& t : e.inputs) {
    if (!t.shape) continue;
    const std::uint64_t marker = t.shape->size();
    mix(&marker, sizeof marker);
    mix(t.shape->data(), t.shape->size() * sizeof(std::int64_t));
  }
  return h;
}

template <typename Events>
AnomalyReport evaluate(const std::vector<AnomalyRule>& rules, const Events& events) {
  AnomalyReport report;
  std::uint64_t summed = 0;
  for (const auto& e : events) {
    if (is_timed_op(e))
      summed += *e.duration_ns;
    else if (e.type == EventType::Operation && e.category != Category::Session)
      ++report.skipped_missing_duration;
  }
  for (const auto& rule : rules) {
    const EventFilter scope = EventFilter::parse(rule.scope);
    if (!scope.valid()) continue;  // rules are validated when parsed
    auto in_scope = [&](const TraceEvent& e) { return scope.empty() || scope.matches(e); };
    if (rule.kind == AnomalyRule::Kind::Match) {
      const EventFilter f = EventFilter::parse(rule.filter);
      if (!f.valid() || f.empty()) continue;
      for (std::size_t i = 0; i < events.size(); ++i)
        if (in_scope(events[i]) && f.matches(events[i]))
          report.hits.push_back({i, rule.id, "matches \"" + rule.filter + "\""});
    } else if (rule.kind == AnomalyRule::Kind::Share) {
      if (summed == 0) continue;
      for (std::size_t i = 0; i < events.size(); ++i) {
        const auto& e = events[i];
        if (!is_timed_op(e) || !in_scope(e)) continue;
        const double share = static_cast<double>(*e.duration_ns) / static_cast<double>(summed);
        if (share > rule.above)
          report.hits.push_back({i, rule.id,
                                 format_percent(share) + " of summed op time (limit " +
                                     format_percent(rule.above) + ")"});
      }
    } else {
      struct Group {
        std::vector<std::uint64_t> durations;
        std::uint64_t p95 = 0;
      };
      std::unordered_map<std::uint64_t, Group> groups;
      std::vector<std::pair<std::size_t, std::uint64_t>> candidates;  // event, group hash
      for (std::size_t i = 0; i < events.size(); ++i) {
        const auto& e = events[i];
        if (!is_timed_op(e) || !in_scope(e)) continue;
        const std::uint64_t h = group_hash(e, rule.group_by_op);
        groups[h].durations.push_back(*e.duration_ns);
        if (*e.duration_ns >= rule.min_ns) candidates.emplace_back(i, h);
      }
      for (auto& [h, g] : groups) {
        auto& d = g.durations;
        if (d.size() < rule.min_calls) continue;
        // Nearest-rank p95.
        const auto rank = static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(d.size())));
        const auto nth =
            d.begin() + static_cast<std::ptrdiff_t>(std::max<std::size_t>(rank, 1) - 1);
        std::nth_element(d.begin(), nth, d.end());
        g.p95 = *nth;
      }
      for (const auto& [i, h] : candidates) {
        const Group& g = groups[h];
        if (g.p95 == 0) continue;
        const auto& e = events[i];
        const double ratio = static_cast<double>(*e.duration_ns) / static_cast<double>(g.p95);
        if (ratio > rule.factor)
          report.hits.push_back({i, rule.id,
                                 format_duration_ns(*e.duration_ns) + " = " + fixed(ratio, 1) +
                                     "x p95 of " + group_key(e, rule.group_by_op) + " (" +
                                     format_duration_ns(g.p95) + ", " +
                                     format_count(g.durations.size()) + " calls)"});
      }
    }
  }
  std::stable_sort(
      report.hits.begin(), report.hits.end(),
      [](const AnomalyHit& a, const AnomalyHit& b) { return a.event_index < b.event_index; });
  return report;
}

}  // namespace

std::vector<AnomalyRule> default_rules() {
  AnomalyRule slow;
  slow.id = "slow-vs-p95";
  slow.kind = AnomalyRule::Kind::P95Multiple;
  slow.factor = 3.0;
  slow.min_calls = 20;
  AnomalyRule share;
  share.id = "large-share";
  share.kind = AnomalyRule::Kind::Share;
  share.above = 0.25;
  return {slow, share};
}

std::vector<AnomalyRule> parse_rules_json(const std::string& text) {
  nlohmann::json doc;
  try {
    doc = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error(std::string("rules file is not valid JSON: ") + e.what());
  }
  if (!doc.is_object() || !doc.contains("rules") || !doc["rules"].is_array())
    throw std::runtime_error("rules file needs an object with a \"rules\" array");
  const auto& list = doc["rules"];
  if (list.size() > kMaxRules)
    throw std::runtime_error("too many rules (max " + std::to_string(kMaxRules) + ")");
  std::vector<AnomalyRule> rules;
  for (std::size_t i = 0; i < list.size(); ++i) {
    const auto& j = list[i];
    std::string where = "rule " + std::to_string(i + 1);
    if (!j.is_object()) throw std::runtime_error(where + ": must be an object");
    AnomalyRule r;
    auto str = [&](const char* key, std::string& out, bool required) {
      if (!j.contains(key)) {
        if (required) throw std::runtime_error(where + ": missing \"" + key + "\"");
        return;
      }
      if (!j[key].is_string() || j[key].get<std::string>().size() > 256)
        throw std::runtime_error(where + ": \"" + key + "\" must be a string (max 256 bytes)");
      out = j[key].get<std::string>();
    };
    auto num = [&](const char* key, double& out, double lo, double hi) {
      if (!j.contains(key)) return;
      if (!j[key].is_number() || !(j[key].get<double>() >= lo && j[key].get<double>() <= hi))
        throw std::runtime_error(where + ": \"" + key + "\" must be a number in [" + fixed(lo, 2) +
                                 ", " + fixed(hi, 2) + "]");
      out = j[key].get<double>();
    };
    str("id", r.id, true);
    if (r.id.empty()) throw std::runtime_error(where + ": \"id\" must not be empty");
    where += " (" + r.id + ")";
    for (const auto& prev : rules)
      if (prev.id == r.id) throw std::runtime_error(where + ": duplicate id");
    std::string kind;
    str("kind", kind, true);
    if (kind == "p95_multiple") {
      r.kind = AnomalyRule::Kind::P95Multiple;
      num("factor", r.factor, 1.0, 1000.0);
      double min_calls = static_cast<double>(r.min_calls);
      num("min_calls", min_calls, 2.0, 1e6);
      r.min_calls = static_cast<std::size_t>(min_calls);
      std::string group = "name";
      str("group", group, false);
      if (group != "name" && group != "op")
        throw std::runtime_error(where + ": \"group\" must be \"name\" or \"op\"");
      r.group_by_op = group == "op";
      double min_us = static_cast<double>(r.min_ns) / 1000.0;
      num("min_us", min_us, 0.0, 1e9);
      r.min_ns = static_cast<std::uint64_t>(min_us * 1000.0);
    } else if (kind == "share") {
      r.kind = AnomalyRule::Kind::Share;
      num("above", r.above, 0.0, 1.0);
    } else if (kind == "match") {
      r.kind = AnomalyRule::Kind::Match;
      str("filter", r.filter, true);
      if (checked_filter(r.filter, where + " filter").empty())
        throw std::runtime_error(where + ": \"filter\" must not be empty");
    } else {
      throw std::runtime_error(where + ": unknown kind \"" + kind +
                               "\" (use p95_multiple, share or match)");
    }
    str("scope", r.scope, false);
    checked_filter(r.scope, where + " scope");
    for (const auto& item : j.items()) {
      static const char* known[] = {"id",     "kind",  "factor", "min_calls", "group",
                                    "min_us", "above", "filter", "scope"};
      if (std::find_if(std::begin(known), std::end(known),
                       [&](const char* k) { return item.key() == k; }) == std::end(known))
        throw std::runtime_error(where + ": unknown field \"" + item.key() + "\"");
    }
    rules.push_back(std::move(r));
  }
  return rules;
}

std::vector<AnomalyRule> load_rules_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::ostringstream text;
  text << in.rdbuf();
  if (text.str().size() > (1u << 20)) throw std::runtime_error(path + " is larger than 1 MiB");
  return parse_rules_json(text.str());
}

std::string describe_rule(const AnomalyRule& r, bool ascii) {
  std::string out = r.id + ": ";
  switch (r.kind) {
    case AnomalyRule::Kind::P95Multiple:
      out += "duration > " + fixed(r.factor, 1) + "x p95 of the same " +
             (r.group_by_op ? "op type" : "op name") + " and input shapes (" +
             std::to_string(r.min_calls) + "+ timed calls, ignoring ops under " +
             format_duration_ns(r.min_ns, ascii) + ")";
      break;
    case AnomalyRule::Kind::Share:
      out += "one op > " + format_percent(r.above) + " of summed op time";
      break;
    case AnomalyRule::Kind::Match:
      out += "matches \"" + r.filter + "\"";
      break;
  }
  if (!r.scope.empty()) out += ", within \"" + r.scope + "\"";
  return out;
}

AnomalyReport evaluate_rules(const std::vector<AnomalyRule>& rules,
                             const std::vector<TraceEvent>& events) {
  return evaluate(rules, events);
}

AnomalyReport evaluate_rules(const std::vector<AnomalyRule>& rules,
                             const std::deque<TraceEvent>& events) {
  return evaluate(rules, events);
}

std::string format_check_report(const std::string& label, const TraceMetadata& metadata,
                                const std::vector<TraceEvent>& events,
                                const std::vector<AnomalyRule>& rules, const AnomalyReport& report,
                                std::size_t top) {
  std::string out = "llm-trace check: " + label + "\n";
  out += "source: " + std::string(to_string(metadata.origin)) + ", " + format_count(events.size()) +
         " events\n";
  if (metadata.origin == DataSource::Synthetic)
    out += "note: synthetic demo data; timings come from a cost model, not a real model\n";
  if (metadata.overwritten_events > 0)
    out += "note: " + format_count(metadata.overwritten_events) +
           " event(s) were overwritten before saving; p95 and shares cover retained events\n";
  if (report.skipped_missing_duration > 0)
    out += "note: " + format_count(report.skipped_missing_duration) +
           " op(s) without a duration were skipped by duration rules\n";
  out += "\nRules (heuristics; a match is not a measured finding or a cause):\n";
  if (rules.empty()) out += "  none\n";
  for (const auto& r : rules) out += "  " + describe_rule(r) + "\n";

  std::vector<std::size_t> flagged;
  for (const auto& h : report.hits)
    if (flagged.empty() || flagged.back() != h.event_index) flagged.push_back(h.event_index);
  out += "\nFlagged events: " + format_count(flagged.size());
  if (flagged.size() > top) out += " (showing first " + format_count(top) + ")";
  out += "\n";
  std::size_t shown = 0;
  for (std::size_t i = 0; i < report.hits.size();) {
    const std::size_t index = report.hits[i].event_index;
    if (shown++ >= top) break;
    const TraceEvent& e = events[index];
    out += "  #" + std::to_string(e.sequence) + "  " + e.name;
    if (e.layer_index) out += "  layer " + std::to_string(*e.layer_index);
    out += "  " + format_duration_ns(e.duration_ns) + "\n";
    for (; i < report.hits.size() && report.hits[i].event_index == index; ++i)
      out += "      " + report.hits[i].rule_id + ": " + report.hits[i].detail + "\n";
  }
  return out;
}

}  // namespace llm_trace
