#include <ftxui/component/event.hpp>

#include "tui/app_state.hpp"

namespace llm_trace::tui {

using ftxui::Event;

KeyResult map_key(const Event& event, Mode mode) {
  if (mode == Mode::Filter) {
    if (event == Event::Escape) return {Action::Close, {}};
    if (event == Event::Return) return {Action::FilterCommit, {}};
    if (event == Event::Backspace) return {Action::FilterBackspace, {}};
    if (event == Event::ArrowDown) return {Action::MoveDown, {}};
    if (event == Event::ArrowUp) return {Action::MoveUp, {}};
    if (event.is_character()) return {Action::FilterInput, event.character()};
    return {};
  }
  if (mode == Mode::Help) {
    if (event == Event::Escape || event == Event::Character('?')) return {Action::ToggleHelp, {}};
    if (event == Event::Character('q')) return {Action::Quit, {}};
    return {};
  }
  if (event == Event::Character('j') || event == Event::ArrowDown) return {Action::MoveDown, {}};
  if (event == Event::Character('k') || event == Event::ArrowUp) return {Action::MoveUp, {}};
  if (event == Event::PageDown) return {Action::PageDown, {}};
  if (event == Event::PageUp) return {Action::PageUp, {}};
  if (event == Event::Home || event == Event::Character('g')) return {Action::Home, {}};
  if (event == Event::End || event == Event::Character('G')) return {Action::End, {}};
  if (event == Event::Tab) return {Action::NextPanel, {}};
  if (event == Event::TabReverse) return {Action::PrevPanel, {}};
  if (event == Event::Return) return {Action::Inspect, {}};
  if (event == Event::Character('/')) return {Action::OpenFilter, {}};
  if (event == Event::Character('p')) return {Action::TogglePause, {}};
  if (event == Event::Character('m')) return {Action::ToggleMetrics, {}};
  if (event == Event::Character('t')) return {Action::CycleSort, {}};
  if (event == Event::Character('?')) return {Action::ToggleHelp, {}};
  if (event == Event::Escape) return {Action::Close, {}};
  if (event == Event::Character('q')) return {Action::Quit, {}};
  return {};
}

}  // namespace llm_trace::tui
