#include <algorithm>
#include <ftxui/screen/screen.hpp>
#include <string>
#include <vector>

#include "llm_trace/metrics.hpp"
#include "tui/render.hpp"

namespace llm_trace::tui {

using namespace ftxui;

namespace {

constexpr int kLabelWidth = 18;

struct Ctx {
  const Controller& c;
  const Theme theme;
  LayoutMode layout;
  int width;
  int height;
  const TraceMetadata& meta() const { return c.session().metadata(); }
  bool ascii() const { return theme.ascii(); }
};

std::string dur(const Ctx& ctx, const std::optional<std::uint64_t>& ns) {
  return ns ? format_duration_ns(*ns, ctx.ascii()) : std::string(ctx.theme.missing());
}

bool is_demo(const TraceMetadata& m) {
  return m.source == DataSource::Synthetic || m.origin == DataSource::Synthetic;
}

std::string source_label(const TraceMetadata& m) {
  if (is_demo(m)) return "SYNTHETIC DEMO";
  switch (m.source) {
    case DataSource::OnnxProfile:
      return "ONNX PROFILE";
    case DataSource::Replay:
      return "TRACE REPLAY";
    default:
      return "NO TRACE";
  }
}

// Rule details are formatted by the core with UTF-8 units; ASCII mode needs plain "us".
std::string display_text(const Ctx& ctx, std::string s) {
  if (!ctx.ascii()) return s;
  for (std::size_t at = s.find("\xC2\xB5"); at != std::string::npos; at = s.find("\xC2\xB5", at))
    s.replace(at, 2, "u");
  return s;
}

Element field(const Ctx& ctx, const std::string& label, Element value) {
  return hbox({text(label) | ctx.theme.text_secondary() | size(WIDTH, EQUAL, kLabelWidth),
               std::move(value) | flex});
}

Element field(const Ctx& ctx, const std::string& label, const std::string& value) {
  return field(ctx, label, paragraph(value));
}

// DESIGN.md 6.2: three distinct unavailable states.
Element unavailable(const Ctx& ctx, bool capability) {
  return text(capability ? "Not present in this trace" : "Not exposed by adapter") |
         ctx.theme.unavailable();
}

template <typename T, typename F>
Element optional_field(const Ctx& ctx, const std::string& label, const std::optional<T>& value,
                       bool capability, F format) {
  if (value) return field(ctx, label, format(*value));
  return field(ctx, label, unavailable(ctx, capability));
}

Element section(const Ctx& ctx, const std::string& title) {
  return text(title) | ctx.theme.heading() | ctx.theme.text_secondary();
}

Element header(const Ctx& ctx) {
  const auto& m = ctx.meta();
  Elements left{text("LLM TRACE") | bold, text("  ")};
  if (is_demo(m)) {
    left.push_back(text("[DEMO]") | bold | ctx.theme.demo());
    left.push_back(text(" "));
  }
  const bool narrow = ctx.layout == LayoutMode::Narrow;
  if (!(narrow && is_demo(m)))
    left.push_back(text(source_label(m)) | (is_demo(m) ? ctx.theme.demo() : ctx.theme.accent()));
  if (!m.run_id.empty() && !narrow) {
    left.push_back(text(ctx.theme.dot()) | ctx.theme.text_secondary());
    left.push_back(text(m.run_id));
  }
  const auto& live = ctx.c.session().live_stats();
  std::string state = live ? (live->capturing ? "LIVE"
                              : narrow        ? "STOPPED"
                                              : "CAPTURE STOPPED")
                           : "REPLAY";
  if (m.source == DataSource::None && !live) state.clear();
  if (!state.empty()) {
    left.push_back(text(ctx.theme.dot()) | ctx.theme.text_secondary());
    left.push_back(text(state));
  }
  left.push_back(filler());
  left.push_back(text(narrow ? "? q" : "? help  q quit") | ctx.theme.text_secondary());
  return hbox(std::move(left));
}

std::string list_title(const Ctx& ctx) {
  const auto& st = ctx.c.state();
  std::string title = "OPERATIONS";
  if (st.sort != SortMode::Source) title += std::string(" (by ") + to_string(st.sort) + ")";
  const std::size_t total = ctx.c.session().events().size();
  if (!st.filter.empty() || st.mode == Mode::Filter) {
    title += std::string(ctx.theme.dot()) + format_count(ctx.c.view().size()) + " of " +
             format_count(total);
  } else {
    title += std::string(ctx.theme.dot()) + format_count(total) + " events";
  }
  return title;
}

Element empty_list(const Ctx& ctx) {
  const auto& st = ctx.c.state();
  const auto& live = ctx.c.session().live_stats();
  if (!st.filter.empty()) {
    return vbox({text("No events match \"" + st.filter + "\""),
                 text("Esc clears the filter.") | ctx.theme.text_secondary()});
  }
  if (live && live->capturing) return text("Waiting for events...") | ctx.theme.text_secondary();
  return vbox({text("No trace loaded") | bold, text(""),
               text("Launch the synthetic demo:") | ctx.theme.text_secondary(),
               text("  llm-trace --demo")});
}

Element operation_list(const Ctx& ctx, int pane_width) {
  const auto& st = ctx.c.state();
  const bool focused = st.focus == Panel::Operations;
  Elements rows;
  Element title = text(list_title(ctx)) | bold;
  if (focused) title = title | ctx.theme.accent();
  rows.push_back(title);
  if (st.mode == Mode::Filter || !st.filter.empty()) {
    Elements line{text("/ ") | ctx.theme.accent(),
                  text(st.filter + (st.mode == Mode::Filter ? "_" : ""))};
    if (!st.filter_error.empty())
      line.push_back(text("  " + std::string(ctx.theme.warn()) + " " + st.filter_error) |
                     ctx.theme.warning() | flex_shrink);
    rows.push_back(hbox(std::move(line)));
  }
  const auto& view = ctx.c.view();
  if (view.empty()) {
    rows.push_back(empty_list(ctx));
    return vbox(std::move(rows));
  }
  const std::size_t visible =
      list_rows(ctx.height, ctx.layout) - ((st.mode == Mode::Filter || !st.filter.empty()) ? 1 : 0);
  const std::size_t page = std::max<std::size_t>(1, visible);
  std::size_t top = std::min(st.scroll_top, view.size() - 1);
  if (st.selected_row < top) top = st.selected_row;
  if (st.selected_row >= top + page) top = st.selected_row + 1 - page;

  const bool show_category = ctx.layout == LayoutMode::Wide;
  const int dur_w = 10;
  const int cat_w = show_category ? 14 : 0;
  const int name_w = std::max(8, pane_width - 2 - dur_w - cat_w - 1);
  const auto& events = ctx.c.session().events();
  for (std::size_t row = top; row < std::min(view.size(), top + page); ++row) {
    const TraceEvent& e = events[view[row]];
    const bool selected = row == st.selected_row;
    std::string name = e.name;
    const bool flagged = e.type == EventType::Warning || e.type == EventType::AdapterError;
    if (flagged) name = std::string(ctx.theme.warn()) + " " + name + ": " + e.message;
    const bool heuristic = ctx.c.flags_for(e) != nullptr;
    Elements cells{text(selected ? std::string(ctx.theme.selected()) : " "),
                   text(heuristic ? ctx.theme.warn() : " ") | ctx.theme.warning(),
                   text(truncate_utf8(name, static_cast<std::size_t>(name_w), ctx.ascii())) | flex};
    if (show_category)
      cells.push_back(text(to_string(e.category)) | ctx.theme.text_secondary() |
                      size(WIDTH, EQUAL, cat_w));
    cells.push_back(text(dur(ctx, e.duration_ns)) | align_right | size(WIDTH, EQUAL, dur_w));
    Element line = hbox(std::move(cells));
    if (flagged) line = line | ctx.theme.warning();
    if (selected) line = focused ? (line | inverted) : (line | bold);
    rows.push_back(line);
  }
  return vbox(std::move(rows));
}

std::string shape_dtype(const TensorMetadata& t) {
  std::string s = t.shape ? format_shape(*t.shape) : "[?]";
  if (t.dtype) s += "  " + *t.dtype;
  return s;
}

Element tensors(const Ctx& ctx, const TraceEvent& e) {
  const auto& caps = ctx.meta().capabilities;
  Elements rows{section(ctx, "TENSOR METADATA")};
  if (!caps.tensor_metadata) {
    rows.push_back(unavailable(ctx, false));
    return vbox(std::move(rows));
  }
  if (e.inputs.empty() && e.outputs.empty()) {
    rows.push_back(unavailable(ctx, true));
    return vbox(std::move(rows));
  }
  auto add = [&](const char* dir, const TensorMetadata& t) {
    std::string detail = shape_dtype(t);
    if (ctx.layout == LayoutMode::Wide || ctx.layout == LayoutMode::Narrow) {
      if (t.element_count) detail += "  " + format_count(*t.element_count) + " el";
      if (t.byte_count) detail += "  " + format_bytes(*t.byte_count);
    }
    rows.push_back(
        hbox({text(dir) | ctx.theme.text_secondary() | size(WIDTH, EQUAL, 4),
              text(t.name.empty() ? "(unnamed)" : t.name) | size(WIDTH, EQUAL, kLabelWidth - 4),
              paragraph(detail) | flex}));
  };
  for (const auto& t : e.inputs) add("in", t);
  for (const auto& t : e.outputs) add("out", t);
  return vbox(std::move(rows));
}

Element details(const Ctx& ctx) {
  const bool focused = ctx.c.state().focus == Panel::Details;
  Element title = text("SELECTED OPERATION") | bold;
  if (focused) title = title | ctx.theme.accent();
  const TraceEvent* e = ctx.c.selected();
  if (!e) return vbox({title, text("No event selected") | ctx.theme.text_secondary()});
  const auto& caps = ctx.meta().capabilities;
  const auto& m = ctx.c.session().metrics();
  const std::uint64_t base = m.first_timestamp_ns.value_or(0);
  auto str = [](const std::string& s) { return s; };
  auto num = [](std::int64_t v) { return std::to_string(v); };

  Elements rows{title, text(e->name) | bold};
  rows.push_back(field(ctx, "type", to_string(e->type)));
  rows.push_back(field(ctx, "category", to_string(e->category)));
  rows.push_back(optional_field(ctx, "op type", e->op_type, true, str));
  rows.push_back(
      optional_field(ctx, "duration", e->duration_ns, caps.operation_timing,
                     [&](std::uint64_t ns) { return format_duration_ns(ns, ctx.ascii()); }));
  rows.push_back(field(ctx, "timestamp",
                       format_offset_ns(e->timestamp_ns >= base ? e->timestamp_ns - base : 0)));
  rows.push_back(field(ctx, "sequence", std::to_string(e->sequence)));
  rows.push_back(optional_field(ctx, "layer", e->layer_index, caps.layer_hierarchy, num));
  rows.push_back(optional_field(ctx, "token", e->token_index, caps.token_index, num));
  rows.push_back(optional_field(ctx, "parent", e->parent_id, caps.layer_hierarchy, str));
  rows.push_back(optional_field(ctx, "device", e->device, true, str));
  if (ctx.layout != LayoutMode::Medium)
    rows.push_back(optional_field(ctx, "thread", e->thread_id, true, num));
  if (!e->message.empty()) rows.push_back(field(ctx, "message", e->message));
  rows.push_back(text(""));
  rows.push_back(tensors(ctx, *e));
  if (e->category == Category::Attention) {
    rows.push_back(text(""));
    rows.push_back(section(ctx, "ATTENTION VALUES"));
    rows.push_back(unavailable(ctx, caps.attention_values));
  }
  if (e->activation || caps.activation_summaries || ctx.layout == LayoutMode::Wide) {
    rows.push_back(text(""));
    rows.push_back(section(ctx, "ACTIVATION SUMMARY"));
    if (e->activation) {
      const auto& a = *e->activation;
      auto fmt = [](double v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6g", v);
        return std::string(buf);
      };
      rows.push_back(field(ctx, "mean / std", fmt(a.mean) + " / " + fmt(a.stddev)));
      rows.push_back(field(ctx, "min / max", fmt(a.min) + " / " + fmt(a.max)));
    } else {
      rows.push_back(unavailable(ctx, caps.activation_summaries));
    }
  }
  if (const auto* hits = ctx.c.flags_for(*e)) {
    rows.push_back(text(""));
    rows.push_back(section(ctx, "HEURISTIC FLAGS  (rule matches, not measured findings)"));
    for (const auto& h : *hits)
      rows.push_back(
          hbox({text(std::string(ctx.theme.warn()) + " " + h.rule_id + ": ") | ctx.theme.warning(),
                paragraph(display_text(ctx, h.detail)) | flex}));
  }
  if (!e->attributes.empty() && ctx.layout == LayoutMode::Wide) {
    rows.push_back(text(""));
    rows.push_back(section(ctx, "SOURCE ATTRIBUTES"));
    for (const auto& [k, v] : e->attributes) rows.push_back(field(ctx, k, v));
  }
  return vbox(std::move(rows));
}

Element metrics_panel(const Ctx& ctx) {
  const bool focused = ctx.c.state().focus == Panel::Metrics;
  Element title = text("METRICS") | bold;
  if (focused) title = title | ctx.theme.accent();
  const auto& s = ctx.c.session();
  const auto& m = s.metrics();
  Elements rows{title};
  if (is_demo(ctx.meta()))
    rows.push_back(text("Synthetic values from a cost model, not measurements.") |
                   ctx.theme.demo());
  auto defined = [&](const std::string& label, const std::string& value, const std::string& def) {
    rows.push_back(field(ctx, label,
                         hbox({text(value) | size(WIDTH, EQUAL, 12),
                               text(def) | ctx.theme.text_secondary() | flex})));
  };
  defined("observed span", dur(ctx, m.observed_span_ns), "first event start to last event end");
  defined("summed op time", format_duration_ns(m.summed_operation_ns, ctx.ascii()),
          "sum of op durations; may overlap, not end-to-end latency");
  if (m.source_run_total_ns)
    defined("source run total", format_duration_ns(*m.source_run_total_ns, ctx.ascii()),
            format_count(m.source_run_count) + " run event(s) reported by the source");
  rows.push_back(field(ctx, "events", format_count(m.event_count)));
  rows.push_back(field(ctx, "timed operations", format_count(m.operation_count)));
  if (m.missing_duration_count)
    rows.push_back(field(ctx, "missing duration", format_count(m.missing_duration_count)));

  rows.push_back(text(""));
  rows.push_back(section(ctx, "BUFFER"));
  if (const auto& live = s.live_stats()) {
    rows.push_back(field(ctx, "capacity", format_count(live->capacity)));
    rows.push_back(field(ctx, "retained", format_count(live->retained)));
    rows.push_back(field(ctx, "overwritten", format_count(live->overwritten)));
    rows.push_back(field(ctx, "rejected", format_count(live->rejected)));
  } else if (ctx.meta().buffer_capacity > 0) {
    rows.push_back(field(ctx, "capacity", format_count(ctx.meta().buffer_capacity)));
    rows.push_back(field(ctx, "overwritten", format_count(ctx.meta().overwritten_events)));
  } else {
    rows.push_back(text("Not captured") | ctx.theme.unavailable());
  }

  const double total = static_cast<double>(m.summed_operation_ns);
  auto share = [&](std::uint64_t ns) {
    return total > 0 ? format_percent(static_cast<double>(ns) / total) : std::string("-");
  };
  rows.push_back(text(""));
  rows.push_back(section(ctx, "TOP OPERATIONS  (share of summed op time)"));
  const std::size_t top_n = ctx.layout == LayoutMode::Wide ? 8 : 5;
  for (std::size_t i = 0; i < std::min(top_n, m.top_operations.size()); ++i) {
    const TraceEvent& e = s.events()[m.top_operations[i]];
    rows.push_back(hbox({text(dur(ctx, e.duration_ns)) | align_right | size(WIDTH, EQUAL, 10),
                         text(share(e.duration_ns.value_or(0))) | align_right |
                             size(WIDTH, EQUAL, 8) | ctx.theme.text_secondary(),
                         text("  " + e.name) | flex}));
  }
  if (m.top_operations.empty())
    rows.push_back(text("No timed operations") | ctx.theme.unavailable());
  rows.push_back(text(""));
  rows.push_back(section(ctx, "BY CATEGORY"));
  for (const auto& g : m.by_category) {
    rows.push_back(hbox(
        {text(g.key) | size(WIDTH, EQUAL, 16),
         text(format_duration_ns(g.total_ns, ctx.ascii())) | align_right | size(WIDTH, EQUAL, 10),
         text(share(g.total_ns)) | align_right | size(WIDTH, EQUAL, 8) | ctx.theme.text_secondary(),
         text("  " + format_count(g.count) + " ops") | ctx.theme.text_secondary()}));
  }
  const auto warnings = s.warnings();
  rows.push_back(text(""));
  rows.push_back(section(ctx, "WARNINGS AND NOTES"));
  if (warnings.empty() && ctx.c.flagged_count() == 0)
    rows.push_back(text("None") | ctx.theme.text_secondary());
  if (ctx.c.flagged_count() > 0)
    rows.push_back(hbox({text(std::string(ctx.theme.warn()) + " ") | ctx.theme.warning(),
                         paragraph(format_count(ctx.c.flagged_count()) +
                                   " event(s) matched a heuristic rule; press n to step "
                                   "through them. Rule matches are not measured findings.") |
                             flex}));
  for (const auto& w : warnings)
    rows.push_back(hbox(
        {text(std::string(ctx.theme.warn()) + " ") | ctx.theme.warning(), paragraph(w) | flex}));
  return vbox(std::move(rows));
}

std::string view_status(const Ctx& ctx) {
  const auto& live = ctx.c.session().live_stats();
  const bool paused = ctx.c.state().paused;
  const std::string dot = ctx.theme.dot();
  if (!live) return paused ? "VIEW PAUSED" : "REPLAY";
  if (paused)
    return live->capturing ? "VIEW PAUSED" + dot + "CAPTURE CONTINUES"
                           : "VIEW PAUSED" + dot + "CAPTURE STOPPED";
  return live->capturing ? "VIEW LIVE" + dot + "CAPTURING" : std::string("CAPTURE STOPPED");
}

Elements status_strip(const Ctx& ctx) {
  const auto& s = ctx.c.session();
  const auto& m = s.metrics();
  const std::string dot = ctx.theme.dot();
  const bool narrow = ctx.layout == LayoutMode::Narrow;
  std::string line = narrow ? "" : "METRICS  ";
  line += (narrow ? "obs " : "observed ") + dur(ctx, m.observed_span_ns);
  line += dot + (narrow ? "ev " : "events ") + format_count(m.event_count);
  std::uint64_t overwritten = ctx.meta().overwritten_events;
  if (const auto& live = s.live_stats()) {
    overwritten = live->overwritten;
    const double fill =
        live->capacity ? static_cast<double>(live->retained) / static_cast<double>(live->capacity)
                       : 0.0;
    line += dot + (narrow ? "buf " : "buffer ") + format_percent(fill);
  }
  line += dot + (narrow ? "ow " : "overwritten ") + format_count(overwritten);
  Elements left{text(line) | flex_shrink};  // the live/paused status wins when space is short
  if (m.warning_count)
    left.push_back(text(dot + std::string(ctx.theme.warn()) + " " + format_count(m.warning_count) +
                        " warnings") |
                   ctx.theme.warning());
  Element status =
      text(view_status(ctx)) | (ctx.c.state().paused ? ctx.theme.warning() : ctx.theme.success());
  if (narrow) return {hbox(std::move(left)), status};
  left.push_back(filler());
  left.push_back(text("  "));
  left.push_back(status);
  return {hbox(std::move(left))};
}

Element footer(const Ctx& ctx) {
  const auto mode = ctx.c.state().mode;
  // One-shot notices (save result, overwritten selection) replace the key hints until
  // the next key press, so they never crowd the metrics strip.
  const std::string& notice = ctx.c.state().notice;
  if (!notice.empty() && mode == Mode::Normal)
    return text(std::string(ctx.theme.warn()) + " " + notice) | ctx.theme.warning();
  std::string hints;
  if (mode == Mode::Filter)
    hints = "type to filter  Enter apply  Backspace delete  Esc clear";
  else if (mode == Mode::Help)
    hints = "Esc or ? close help  q quit";
  else if (ctx.layout == LayoutMode::Narrow)
    hints = "Tab panel  j/k move  / filter  m metrics  ? help  q quit";
  else
    hints =
        "Tab panels  j/k move  Enter inspect  / filter  t sort  m metrics  p pause  s save  ? help "
        " "
        "q quit";
  return text(hints) | ctx.theme.text_secondary();
}

Element help_overlay(const Ctx& ctx) {
  const auto& m = ctx.meta();
  const auto& caps = m.capabilities;
  auto key = [&](const std::string& k, const std::string& what) {
    return hbox({text(k) | bold | size(WIDTH, EQUAL, 16), text(what)});
  };
  auto cap = [&](const std::string& name, bool on) {
    return hbox(
        {text(name) | size(WIDTH, EQUAL, 22),
         on ? text("available") : text("Not exposed by adapter") | ctx.theme.unavailable()});
  };
  Elements rows{
      ctx.theme.hsep(),
      text(" HELP" + std::string(ctx.theme.dot()) + source_label(m)) | bold,
      ctx.theme.hsep(),
      key("j / k, arrows", "move selection"),
      key("PgUp / PgDn", "move one page"),
      key("g / G", "first / last event"),
      key("Tab / Shift+Tab", "switch panel"),
      key("Enter", "inspect selected operation"),
      key("/", "filter: text  layer:3  op:MatMul  cat:attention  dur>50us"),
      key("t", "sort: source order, duration, name"),
      key("m", "toggle metrics panel"),
      key("p", "pause view (capture continues)"),
      key("n", "next event flagged by a heuristic rule (!)"),
      key("s", "save trace as JSONL (all retained events)"),
      key("Esc", "close filter, help or panel"),
      key("q", "quit"),
      text(""),
      text("DATA SOURCE") | bold,
      cap("operation timing", caps.operation_timing),
      cap("tensor metadata", caps.tensor_metadata),
      cap("layer hierarchy", caps.layer_hierarchy),
      cap("activation summaries", caps.activation_summaries),
      cap("attention values", caps.attention_values),
  };
  rows.push_back(text(""));
  rows.push_back(text("HEURISTIC RULES") | bold);
  if (ctx.c.rules().empty()) rows.push_back(text("None loaded") | ctx.theme.text_secondary());
  for (const auto& r : ctx.c.rules()) rows.push_back(paragraph(describe_rule(r, ctx.ascii())));
  if (is_demo(m))
    rows.push_back(paragraph("Demo timings come from a synthetic cost model, not a real model.") |
                   ctx.theme.demo());
  rows.push_back(ctx.theme.hsep());
  rows.push_back(text("Esc or ? closes this overlay") | ctx.theme.text_secondary());
  return vbox(std::move(rows)) | size(WIDTH, EQUAL, std::min(72, ctx.width - 2)) | clear_under |
         center;
}

}  // namespace

Element render_app(const Controller& controller, int width, int height) {
  const LayoutMode layout = choose_layout(width, height);
  Ctx ctx{controller, Theme(controller.display()), layout, width, height};
  const auto& st = controller.state();

  Element workspace;
  if (layout == LayoutMode::Narrow) {
    const Panel panel = st.focus;
    workspace = panel == Panel::Operations ? operation_list(ctx, width)
                : panel == Panel::Details  ? details(ctx)
                                           : metrics_panel(ctx);
  } else {
    const int left_w = layout == LayoutMode::Wide ? width * 9 / 20 : width / 2;
    Element right = st.right_panel == Panel::Metrics ? metrics_panel(ctx) : details(ctx);
    workspace = hbox({operation_list(ctx, left_w) | size(WIDTH, EQUAL, left_w), ctx.theme.vsep(),
                      text(" "), right | flex});
  }

  Elements rows{header(ctx), ctx.theme.hsep(), workspace | flex, ctx.theme.hsep()};
  for (auto& line : status_strip(ctx)) rows.push_back(std::move(line));
  rows.push_back(ctx.theme.hsep());
  rows.push_back(footer(ctx));
  Element app = vbox(std::move(rows));
  if (st.mode == Mode::Help) app = dbox({app, help_overlay(ctx)});
  return app;
}

namespace {

Screen render_screen(const Controller& controller, int width, int height) {
  auto screen = Screen::Create(Dimension::Fixed(width), Dimension::Fixed(height));
  Render(screen, render_app(controller, width, height));
  return screen;
}

}  // namespace

std::string render_plain(const Controller& controller, int width, int height) {
  auto screen = render_screen(controller, width, height);
  std::string out;
  for (int y = 0; y < height; ++y) {
    std::string line;
    for (int x = 0; x < width; ++x) line += screen.PixelAt(x, y).character;
    while (!line.empty() && line.back() == ' ') line.pop_back();
    out += line + "\n";
  }
  return out;
}

std::string render_styled(const Controller& controller, int width, int height) {
  return render_screen(controller, width, height).ToString() + "\n";
}

}  // namespace llm_trace::tui
