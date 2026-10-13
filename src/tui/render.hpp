#pragma once

#include <ftxui/dom/elements.hpp>
#include <string>

#include "tui/app_state.hpp"

namespace llm_trace::tui {

// Pure rendering: reads controller state, never mutates it.
ftxui::Element render_app(const Controller& controller, int width, int height);

// Renders one frame into a fixed-size screen and returns it as plain text (no
// escape codes). Used by tests and `--print-frame`.
std::string render_plain(const Controller& controller, int width, int height);
// Same, but with terminal styling escapes.
std::string render_styled(const Controller& controller, int width, int height);

}  // namespace llm_trace::tui
