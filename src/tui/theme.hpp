#pragma once

// Semantic style tokens (DESIGN.md 3.2) and ASCII symbol fallbacks (3.3). Every
// colour has a monochrome fallback; meaning is always carried by text as well.

#include <ftxui/dom/elements.hpp>

namespace llm_trace::tui {

struct DisplayOptions {
  bool color = true;
  bool ascii = false;
};

class Theme {
 public:
  explicit Theme(DisplayOptions options) : options_(options) {}

  bool ascii() const { return options_.ascii; }
  bool color() const { return options_.color; }

  ftxui::Decorator text_secondary() const { return ftxui::dim; }
  ftxui::Decorator unavailable() const { return ftxui::dim; }
  ftxui::Decorator heading() const { return ftxui::bold; }
  ftxui::Decorator accent() const { return tint(ftxui::Color::Cyan); }
  ftxui::Decorator success() const { return tint(ftxui::Color::Green); }
  ftxui::Decorator warning() const { return tint(ftxui::Color::Yellow); }
  ftxui::Decorator error() const { return tint(ftxui::Color::Red); }
  ftxui::Decorator demo() const { return tint(ftxui::Color::Magenta); }
  ftxui::Decorator border_subtle() const { return tint(ftxui::Color::GrayDark); }

  ftxui::Element hsep() const {
    return (options_.ascii ? ftxui::separatorCharacter("-") : ftxui::separatorLight()) |
           border_subtle();
  }
  ftxui::Element vsep() const {
    return (options_.ascii ? ftxui::separatorCharacter("|") : ftxui::separatorLight()) |
           border_subtle();
  }

  const char* dot() const { return options_.ascii ? " | " : " \xC2\xB7 "; }
  const char* selected() const { return options_.ascii ? "*" : "\xE2\x97\x8F"; }
  const char* missing() const { return options_.ascii ? "-" : "\xE2\x80\x94"; }
  const char* warn() const { return "!"; }

 private:
  ftxui::Decorator tint(ftxui::Color c) const {
    if (!options_.color) return ftxui::nothing;
    return ftxui::color(c);
  }

  DisplayOptions options_;
};

}  // namespace llm_trace::tui
