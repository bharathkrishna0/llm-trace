#pragma once

// UI state and the controller that applies key actions to it. The controller owns
// no rendering and no trace processing beyond calling TraceSession::query.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llm_trace/trace_session.hpp"
#include "tui/theme.hpp"

namespace ftxui {
struct Event;
}

namespace llm_trace::tui {

enum class Panel { Operations, Details, Metrics };
enum class LayoutMode { Wide, Medium, Narrow };
enum class Mode { Normal, Filter, Help };

// Thresholds from DESIGN.md 4.3. Short terminals fall back to a single panel.
LayoutMode choose_layout(int width, int height);
// Rows available to the operation list for a terminal of the given height.
std::size_t list_rows(int height, LayoutMode layout);

enum class Action {
  None,
  MoveDown,
  MoveUp,
  PageDown,
  PageUp,
  Home,
  End,
  NextPanel,
  PrevPanel,
  Inspect,
  OpenFilter,
  TogglePause,
  ToggleMetrics,
  CycleSort,
  ToggleHelp,
  Close,
  Save,  // handled by the app, which owns file I/O
  Quit,
  FilterInput,
  FilterBackspace,
  FilterCommit,
};

struct KeyResult {
  Action action = Action::None;
  std::string text;  // FilterInput only
};

// Central key map (DESIGN.md 5.1). In filter mode printable keys are text, never
// global actions.
KeyResult map_key(const ftxui::Event& event, Mode mode);

struct AppState {
  Panel focus = Panel::Operations;
  Panel right_panel = Panel::Details;  // what the right pane shows in two-pane layouts
  Mode mode = Mode::Normal;
  bool paused = false;
  std::string filter;
  std::string filter_error;  // set while the typed filter does not parse
  SortMode sort = SortMode::Source;
  std::optional<std::uint64_t> selected_sequence;
  std::size_t selected_row = 0;
  std::size_t scroll_top = 0;
  std::string notice;
};

class Controller {
 public:
  explicit Controller(DisplayOptions display = {}) : display_(display) {}

  // Replaces the displayed session and keeps the selection on the same event.
  void set_session(TraceSession session);
  // Live mode: applies only what changed in the collector since the last update.
  void apply_delta(CollectorDelta delta);

  // Applies an action. Returns false when the application should quit.
  bool apply(const KeyResult& key, LayoutMode layout, std::size_t page_rows);

  const TraceSession& session() const { return session_; }
  const std::vector<std::size_t>& view() const { return view_; }
  const AppState& state() const { return state_; }
  const DisplayOptions& display() const { return display_; }
  const TraceEvent* selected() const;
  void set_notice(std::string notice) { state_.notice = std::move(notice); }

 private:
  void rebuild_view();
  void select_row(std::size_t row, std::size_t page_rows);

  DisplayOptions display_;
  TraceSession session_;
  std::vector<std::size_t> view_;
  EventFilter active_filter_;  // last filter that parsed; used while the input is invalid
  AppState state_;
};

}  // namespace llm_trace::tui
