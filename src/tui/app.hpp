#pragma once

#include <cstddef>
#include <string>

#include "llm_trace/synthetic.hpp"
#include "tui/theme.hpp"

namespace llm_trace::tui {

struct DemoOptions {
  DisplayOptions display;
  SyntheticConfig synthetic;
  std::size_t buffer_capacity = 65536;
};

// Interactive synthetic demo. Returns the process exit code.
int run_demo(const DemoOptions& options);

// Generates the whole demo trace without pacing and renders one frame.
std::string render_demo_frame(const DemoOptions& options, int width, int height, bool styled);

// Renders the empty "no trace loaded" state.
std::string render_empty_frame(const DisplayOptions& display, int width, int height, bool styled);

}  // namespace llm_trace::tui
