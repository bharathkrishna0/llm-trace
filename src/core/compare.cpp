#include "llm_trace/compare.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <utility>

#include "llm_trace/metrics.hpp"

namespace llm_trace {

namespace {

bool is_compared_op(const TraceEvent& e) {
  return e.type == EventType::Operation && e.category != Category::Session &&
         e.duration_ns.has_value();
}

struct Acc {
  std::vector<std::uint64_t> durations;
  std::uint64_t total = 0;
  void add(std::uint64_t ns) {
    durations.push_back(ns);
    total += ns;
  }
};

// {no layer, index}: layers in numeric order, events without a layer last.
using LayerKey = std::pair<bool, std::int64_t>;

struct Groups {
  std::map<Category, Acc> category;
  std::map<LayerKey, Acc> layer;
  std::map<std::string, Acc> op;
};

SideStat finish(Acc& acc, std::uint64_t summed) {
  SideStat s;
  s.count = acc.durations.size();
  s.total_ns = acc.total;
  if (s.count) {
    s.mean_ns = static_cast<double>(acc.total) / static_cast<double>(s.count);
    auto& d = acc.durations;
    const auto mid = d.begin() + static_cast<std::ptrdiff_t>(s.count / 2);
    std::nth_element(d.begin(), mid, d.end());
    s.median_ns = static_cast<double>(*mid);
    if (s.count % 2 == 0)
      s.median_ns = (s.median_ns + static_cast<double>(*std::max_element(d.begin(), mid))) / 2.0;
  }
  s.share = summed ? static_cast<double>(acc.total) / static_cast<double>(summed) : 0.0;
  return s;
}

void collect(const LoadedTrace& trace, const CompareOptions& options, CompareSide& side,
             Groups& groups) {
  side.metadata = trace.metadata;
  side.events = trace.events.size();
  std::set<std::string> devices;
  for (const auto& e : trace.events) {
    if (e.type == EventType::Operation && e.category != Category::Session && !e.duration_ns)
      ++side.missing_duration;
    if (!is_compared_op(e)) continue;
    const std::uint64_t ns = *e.duration_ns;
    ++side.operations;
    side.summed_ns += ns;
    if (e.device) devices.insert(*e.device);
    groups.category[e.category].add(ns);
    groups.layer[e.layer_index ? LayerKey{false, *e.layer_index} : LayerKey{true, 0}].add(ns);
    const std::string& key = options.key == CompareKey::OpType && e.op_type ? *e.op_type : e.name;
    groups.op[key].add(ns);
  }
  side.devices.assign(devices.begin(), devices.end());
}

template <typename K, typename Name>
std::vector<CompareRow> merge(std::map<K, Acc>& a, std::uint64_t summed_a, std::map<K, Acc>& b,
                              std::uint64_t summed_b, Name name) {
  std::vector<CompareRow> rows;
  auto ia = a.begin();
  auto ib = b.begin();
  while (ia != a.end() || ib != b.end()) {
    CompareRow row;
    if (ib == b.end() || (ia != a.end() && ia->first < ib->first)) {
      row.key = name(ia->first);
      row.a = finish(ia->second, summed_a);
      ++ia;
    } else if (ia == a.end() || ib->first < ia->first) {
      row.key = name(ib->first);
      row.b = finish(ib->second, summed_b);
      ++ib;
    } else {
      row.key = name(ia->first);
      row.a = finish(ia->second, summed_a);
      row.b = finish(ib->second, summed_b);
      ++ia;
      ++ib;
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

std::string join(const std::vector<std::string>& items) {
  std::string out;
  for (const auto& item : items) out += (out.empty() ? "" : ", ") + item;
  return out.empty() ? "none recorded" : out;
}

void add_side_warnings(const CompareSide& s, const std::string& tag,
                       std::vector<std::string>& out) {
  if (s.metadata.origin == DataSource::Synthetic)
    out.push_back(tag +
                  " is synthetic demo data: its timings come from a cost model, not a "
                  "measurement.");
  if (s.operations == 0) out.push_back(tag + " has no timed operations.");
  if (s.missing_duration)
    out.push_back(tag + " has " + format_count(s.missing_duration) +
                  " operations without a duration; they are not compared.");
  if (s.metadata.overwritten_events)
    out.push_back(tag + " lost " + format_count(s.metadata.overwritten_events) +
                  " events to buffer overwrites; it covers only the retained window.");
  if (s.metadata.rejected_events)
    out.push_back(tag + " had " + format_count(s.metadata.rejected_events) +
                  " invalid events rejected at capture.");
}

std::vector<std::string> make_warnings(const CompareSide& a, const CompareSide& b) {
  std::vector<std::string> w;
  const auto& ma = a.metadata;
  const auto& mb = b.metadata;
  if (ma.origin != mb.origin)
    w.push_back(std::string("Different sources: A is ") + to_string(ma.origin) + ", B is " +
                to_string(mb.origin) + ". Timings are not comparable.");
  if (!ma.model.empty() && !mb.model.empty() && ma.model != mb.model)
    w.push_back("Different models: " + ma.model + " vs " + mb.model +
                ". Ops with the same key may not do the same work.");
  if (ma.runtime != mb.runtime || ma.runtime_version != mb.runtime_version)
    w.push_back("Different runtimes: " + (ma.runtime + " " + ma.runtime_version) + " vs " +
                (mb.runtime + " " + mb.runtime_version) + ".");
  if (a.devices != b.devices)
    w.push_back("Different devices: " + join(a.devices) + " vs " + join(b.devices) + ".");
  add_side_warnings(a, "A", w);
  add_side_warnings(b, "B", w);
  return w;
}

std::string fmt(const char* format, ...) {
  char buffer[512];
  va_list args;
  va_start(args, format);
  std::vsnprintf(buffer, sizeof buffer, format, args);
  va_end(args);
  return buffer;
}

std::string duration(double ns, bool ascii) {
  return format_duration_ns(static_cast<std::uint64_t>(std::llround(std::max(0.0, ns))), ascii);
}

// Right-aligns to `width` code points (printf widths count bytes, which misaligns "µs").
std::string pad_left(const std::string& text, std::size_t width) {
  const std::size_t n = utf8_length(text);
  return (n < width ? std::string(width - n, ' ') : std::string()) + text;
}

std::string change_text(const CompareRow& row) {
  const auto c = row.mean_change();
  if (!c) return "n/a";
  return fmt("%+.1f%%", *c * 100.0);
}

std::string side_line(const CompareSide& s, const char* tag, bool ascii) {
  std::string source = to_string(s.metadata.origin);
  if (!s.metadata.runtime.empty())
    source += ", " + s.metadata.runtime +
              (s.metadata.runtime_version.empty() ? "" : " " + s.metadata.runtime_version);
  if (!s.metadata.model.empty()) source += ", model " + s.metadata.model;
  return fmt("  %s  %s\n     %s; %s events, %s timed ops, summed %s\n", tag, s.label.c_str(),
             source.c_str(), format_count(s.events).c_str(), format_count(s.operations).c_str(),
             duration(static_cast<double>(s.summed_ns), ascii).c_str());
}

std::string table(const char* title, const char* key_header, const std::vector<CompareRow>& rows,
                  std::size_t limit, std::size_t min_calls, bool ascii) {
  std::string out = std::string("\n") + title + "\n";
  out += fmt("  %-36s %11s %11s %9s %9s %9s\n", key_header, "A mean", "B mean", "change", "A calls",
             "B calls");
  const std::string dash = ascii ? "-" : "\xE2\x80\x94";
  std::size_t shown = 0;
  for (const auto& r : rows) {
    if (shown++ == limit) break;
    const std::string key =
        truncate_utf8(r.key, 34, ascii) + (r.few_samples(min_calls) ? " *" : "");
    // Pad by code points so UTF-8 keys line up.
    const std::string padded =
        key + std::string(36 - std::min<std::size_t>(36, utf8_length(key)), ' ');
    out += "  " + padded + " " + pad_left(r.a ? duration(r.a->mean_ns, ascii) : dash, 11) + " " +
           pad_left(r.b ? duration(r.b->mean_ns, ascii) : dash, 11) + " " +
           pad_left(change_text(r), 9) + " " + pad_left(r.a ? format_count(r.a->count) : "0", 9) +
           " " + pad_left(r.b ? format_count(r.b->count) : "0", 9) + "\n";
  }
  if (rows.size() > limit) out += fmt("  ... %zu more\n", rows.size() - limit);
  if (rows.empty()) out += "  (none)\n";
  return out;
}

std::string only_table(const char* title, const std::vector<CompareRow>& rows, bool in_a,
                       std::size_t limit, bool ascii) {
  std::string out = fmt("\n%s: %zu\n", title, rows.size());
  std::size_t shown = 0;
  for (const auto& r : rows) {
    if (shown++ == limit) break;
    const auto& s = in_a ? *r.a : *r.b;
    const std::string key = truncate_utf8(r.key, 36, ascii);
    out += "  " + key + std::string(36 - std::min<std::size_t>(36, utf8_length(key)), ' ') +
           fmt(" %9s calls, total %s\n", format_count(s.count).c_str(),
               duration(static_cast<double>(s.total_ns), ascii).c_str());
  }
  if (rows.size() > limit) out += fmt("  ... %zu more\n", rows.size() - limit);
  return out;
}

nlohmann::ordered_json side_json(const CompareSide& s) {
  nlohmann::ordered_json j;
  j["label"] = s.label;
  j["origin"] = to_string(s.metadata.origin);
  j["source"] = to_string(s.metadata.source);
  j["runtime"] = s.metadata.runtime;
  j["runtime_version"] = s.metadata.runtime_version;
  j["model"] = s.metadata.model;
  j["events"] = s.events;
  j["operations"] = s.operations;
  j["missing_duration"] = s.missing_duration;
  j["summed_ns"] = s.summed_ns;
  j["overwritten_events"] = s.metadata.overwritten_events;
  j["devices"] = s.devices;
  return j;
}

nlohmann::ordered_json stat_json(const std::optional<SideStat>& s) {
  if (!s) return nullptr;
  nlohmann::ordered_json j;
  j["count"] = s->count;
  j["total_ns"] = s->total_ns;
  j["mean_ns"] = s->mean_ns;
  j["median_ns"] = s->median_ns;
  j["share"] = s->share;
  return j;
}

nlohmann::ordered_json rows_json(const std::vector<CompareRow>& rows) {
  auto out = nlohmann::ordered_json::array();
  for (const auto& r : rows) {
    nlohmann::ordered_json j;
    j["key"] = r.key;
    j["a"] = stat_json(r.a);
    j["b"] = stat_json(r.b);
    const auto c = r.mean_change();
    j["mean_change"] = c ? nlohmann::ordered_json(*c) : nlohmann::ordered_json(nullptr);
    out.push_back(std::move(j));
  }
  return out;
}

}  // namespace

std::optional<double> CompareRow::mean_change() const {
  if (!a || !b || a->count == 0 || b->count == 0 || a->mean_ns <= 0.0) return std::nullopt;
  return (b->mean_ns - a->mean_ns) / a->mean_ns;
}

bool CompareRow::few_samples(std::size_t min_calls) const {
  return (a && a->count < min_calls) || (b && b->count < min_calls);
}

TraceComparison compare_traces(const LoadedTrace& a, const std::string& label_a,
                               const LoadedTrace& b, const std::string& label_b,
                               const CompareOptions& options) {
  TraceComparison c;
  c.options = options;
  c.a.label = label_a;
  c.b.label = label_b;
  Groups ga, gb;
  collect(a, options, c.a, ga);
  collect(b, options, c.b, gb);
  const auto sa = c.a.summed_ns;
  const auto sb = c.b.summed_ns;

  c.by_category = merge(ga.category, sa, gb.category, sb,
                        [](Category cat) { return std::string(to_string(cat)); });
  c.by_layer = merge(ga.layer, sa, gb.layer, sb, [](const LayerKey& k) {
    return k.first ? std::string("(no layer)") : "layer " + std::to_string(k.second);
  });
  for (auto& row : merge(ga.op, sa, gb.op, sb, [](const std::string& k) { return k; })) {
    if (row.a && row.b)
      c.by_op.push_back(std::move(row));
    else
      (row.a ? c.only_in_a : c.only_in_b).push_back(std::move(row));
  }
  std::sort(c.by_op.begin(), c.by_op.end(), [](const CompareRow& x, const CompareRow& y) {
    const double dx = std::fabs(x.b->mean_ns - x.a->mean_ns);
    const double dy = std::fabs(y.b->mean_ns - y.a->mean_ns);
    if (dx != dy) return dx > dy;
    return x.key < y.key;
  });
  auto by_total = [](bool in_a) {
    return [in_a](const CompareRow& x, const CompareRow& y) {
      const auto tx = (in_a ? x.a : x.b)->total_ns;
      const auto ty = (in_a ? y.a : y.b)->total_ns;
      if (tx != ty) return tx > ty;
      return x.key < y.key;
    };
  };
  std::sort(c.only_in_a.begin(), c.only_in_a.end(), by_total(true));
  std::sort(c.only_in_b.begin(), c.only_in_b.end(), by_total(false));
  c.warnings = make_warnings(c.a, c.b);
  return c;
}

std::string format_comparison(const TraceComparison& c, const CompareTextOptions& o) {
  std::string out = "llm-trace compare: mean time per call, B against A\n";
  out += side_line(c.a, "A", o.ascii);
  out += side_line(c.b, "B", o.ascii);
  if (c.warnings.empty()) {
    out += "\nNo warnings: same source, model, runtime and devices.\n";
  } else {
    out += "\nWarnings:\n";
    for (const auto& w : c.warnings) out += "  ! " + w + "\n";
  }
  const std::size_t all = static_cast<std::size_t>(-1);
  out += table("By category", "category", c.by_category, all, c.options.min_calls, o.ascii);
  out += table("By layer", "layer", c.by_layer, all, c.options.min_calls, o.ascii);
  const bool by_name = c.options.key == CompareKey::Name;
  out += table(by_name ? "By operation name (largest change in mean first)"
                       : "By op type (largest change in mean first)",
               by_name ? "name" : "op type", c.by_op, o.top, c.options.min_calls, o.ascii);
  out += only_table("Only in A", c.only_in_a, true, o.top, o.ascii);
  out += only_table("Only in B", c.only_in_b, false, o.top, o.ascii);
  out += fmt(
      "\n* fewer than %zu calls on a side. Changes are not tested for statistical significance;\n"
      "  repeat each run and compare medians (--compare-json) before drawing conclusions.\n",
      c.options.min_calls);
  return out;
}

std::string comparison_to_json(const TraceComparison& c) {
  nlohmann::ordered_json j;
  j["schema"] = "llm-trace-compare/1";
  j["key"] = c.options.key == CompareKey::Name ? "name" : "op_type";
  j["a"] = side_json(c.a);
  j["b"] = side_json(c.b);
  j["warnings"] = c.warnings;
  j["by_category"] = rows_json(c.by_category);
  j["by_layer"] = rows_json(c.by_layer);
  j["by_op"] = rows_json(c.by_op);
  j["only_in_a"] = rows_json(c.only_in_a);
  j["only_in_b"] = rows_json(c.only_in_b);
  return j.dump(2) + "\n";
}

}  // namespace llm_trace
