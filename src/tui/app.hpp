#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "llm_trace/synthetic.hpp"
#include "llm_trace/trace_io.hpp"
#include "tui/theme.hpp"

namespace llm_trace::tui {

struct DemoOptions {
  DisplayOptions display;
  SyntheticConfig synthetic;
  std::size_t buffer_capacity = 65536;
  std::optional<std::string> output;  // saved on exit and by 's'
};

struct ReplayOptions {
  DisplayOptions display;
  std::optional<std::string> output;  // target for 's'; defaults to a name from the run id
};

// Interactive offline replay of a loaded trace. Returns the process exit code.
int run_replay(const LoadedTrace& trace, const ReplayOptions& options);

// Renders one frame of a loaded trace.
std::string render_replay_frame(const LoadedTrace& trace, const DisplayOptions& display, int width,
                                int height, bool styled);

// "llm-trace-<run id>.jsonl", with the run id reduced to portable file-name characters.
std::string default_output_path(const TraceMetadata& metadata);

// Interactive synthetic demo. Returns the process exit code.
int run_demo(const DemoOptions& options);

// Generates the whole demo trace without pacing and renders one frame. If
// options.output is set the trace is also saved there (throws TraceIoError).
std::string render_demo_frame(const DemoOptions& options, int width, int height, bool styled);

// Renders the empty "no trace loaded" state.
std::string render_empty_frame(const DisplayOptions& display, int width, int height, bool styled);

}  // namespace llm_trace::tui
