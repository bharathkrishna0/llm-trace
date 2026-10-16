#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "cli.hpp"
#include "llm_trace/anomaly.hpp"
#include "llm_trace/compare.hpp"
#include "tui/app.hpp"

int main(int argc, char** argv) {
  using namespace llm_trace;
  const std::vector<std::string> args(argv + 1, argv + argc);
  const cli::ParseResult parsed = cli::parse(args);
  if (!parsed.error.empty()) {
    std::fprintf(stderr, "llm-trace: %s\n\n%s", parsed.error.c_str(), cli::usage().c_str());
    return 2;
  }
  const cli::Options& o = parsed.options;
  if (o.help) {
    std::fputs(cli::usage().c_str(), stdout);
    return 0;
  }
  if (o.version) {
    std::puts("llm-trace " LLM_TRACE_VERSION);
    return 0;
  }

  std::vector<AnomalyRule> rules = default_rules();
  if (o.rules) {
    try {
      rules = load_rules_file(*o.rules);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "llm-trace: invalid rules %s: %s\n", o.rules->c_str(), e.what());
      return 2;
    }
  }

  if (o.check) {
    LoadedTrace trace;
    try {
      trace = load_trace(*o.check);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "llm-trace: cannot load %s: %s\n", o.check->c_str(), e.what());
      return 2;
    }
    for (const auto& w : trace.warnings)
      std::fprintf(stderr, "llm-trace: warning: %s\n", w.c_str());
    const AnomalyReport report = evaluate_rules(rules, trace.events);
    std::fputs(
        format_check_report(*o.check, trace.metadata, trace.events, rules, report, o.top).c_str(),
        stdout);
    return report.hits.empty() ? 0 : 1;
  }

  if (!o.compare.empty()) {
    LoadedTrace traces[2];
    for (int i = 0; i < 2; ++i) {
      const std::string& path = o.compare[static_cast<std::size_t>(i)];
      try {
        traces[i] = load_trace(path);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "llm-trace: cannot load %s: %s\n", path.c_str(), e.what());
        return 1;
      }
      for (const auto& w : traces[i].warnings)
        std::fprintf(stderr, "llm-trace: warning (%c): %s\n", i == 0 ? 'A' : 'B', w.c_str());
    }
    CompareOptions copt;
    copt.key = o.compare_by_op ? CompareKey::OpType : CompareKey::Name;
    const TraceComparison cmp =
        compare_traces(traces[0], o.compare[0], traces[1], o.compare[1], copt);
    if (o.compare_json && *o.compare_json == "-") {
      std::fputs(comparison_to_json(cmp).c_str(), stdout);
      return 0;
    }
    CompareTextOptions text;
    text.top = o.top;
    text.ascii = o.ascii;
    std::fputs(format_comparison(cmp, text).c_str(), stdout);
    if (o.compare_json) {
      std::ofstream out(*o.compare_json, std::ios::binary | std::ios::trunc);
      out << comparison_to_json(cmp);
      if (!out.flush()) {
        std::fprintf(stderr, "llm-trace: could not write %s\n", o.compare_json->c_str());
        return 1;
      }
    }
    return 0;
  }

  tui::DemoOptions demo;
  const char* no_color_env = std::getenv("NO_COLOR");
  demo.display.color = !o.no_color && !(no_color_env && *no_color_env);
  demo.display.ascii = o.ascii;
  demo.synthetic.layers = o.layers;
  demo.synthetic.decode_tokens = o.tokens;
  demo.synthetic.seed = o.seed;
  demo.synthetic.steps_per_second = o.rate;
  demo.buffer_capacity = o.buffer;
  demo.rules = rules;

  demo.output = o.output;

  if (o.open) {
    LoadedTrace trace;
    try {
      trace = load_trace(*o.open);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "llm-trace: cannot load %s: %s\n", o.open->c_str(), e.what());
      return 1;
    }
    for (const auto& w : trace.warnings)
      std::fprintf(stderr, "llm-trace: warning: %s\n", w.c_str());
    if (o.print_frame) {
      if (o.output) {
        try {
          write_jsonl(*o.output, trace.metadata, trace.events);
        } catch (const std::exception& e) {
          std::fprintf(stderr, "llm-trace: could not save %s: %s\n", o.output->c_str(), e.what());
          return 1;
        }
      }
      const auto [w, h] = *o.print_frame;
      std::fputs(
          tui::render_replay_frame(trace, demo.display, w, h, demo.display.color, rules).c_str(),
          stdout);
      return 0;
    }
    tui::ReplayOptions replay;
    replay.display = demo.display;
    replay.output = o.output;
    replay.rules = rules;
    return tui::run_replay(trace, replay);
  }

  if (o.print_frame) {
    const auto [w, h] = *o.print_frame;
    std::string frame;
    try {
      frame = o.demo ? tui::render_demo_frame(demo, w, h, demo.display.color)
                     : tui::render_empty_frame(demo.display, w, h, demo.display.color);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "llm-trace: %s\n", e.what());
      return 1;
    }
    std::fputs(frame.c_str(), stdout);
    return 0;
  }
  if (o.demo) return tui::run_demo(demo);

  std::fputs(cli::usage().c_str(), stdout);
  return 0;
}
