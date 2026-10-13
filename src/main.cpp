#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cli.hpp"
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

  tui::DemoOptions demo;
  const char* no_color_env = std::getenv("NO_COLOR");
  demo.display.color = !o.no_color && !(no_color_env && *no_color_env);
  demo.display.ascii = o.ascii;
  demo.synthetic.layers = o.layers;
  demo.synthetic.decode_tokens = o.tokens;
  demo.synthetic.seed = o.seed;
  demo.synthetic.steps_per_second = o.rate;
  demo.buffer_capacity = o.buffer;

  if (o.print_frame) {
    const auto [w, h] = *o.print_frame;
    const std::string frame = o.demo
                                  ? tui::render_demo_frame(demo, w, h, demo.display.color)
                                  : tui::render_empty_frame(demo.display, w, h, demo.display.color);
    std::fputs(frame.c_str(), stdout);
    return 0;
  }
  if (o.demo) return tui::run_demo(demo);

  std::fputs(cli::usage().c_str(), stdout);
  return 0;
}
