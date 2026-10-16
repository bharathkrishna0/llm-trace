#include <algorithm>
#include <cstdlib>

#include "tui/app_state.hpp"

namespace llm_trace::tui {

namespace {

constexpr std::size_t kMaxFilterBytes = 128;

void pop_utf8(std::string& text) {
  while (!text.empty()) {
    const auto byte = static_cast<unsigned char>(text.back());
    text.pop_back();
    if ((byte & 0xC0u) != 0x80u) break;
  }
}

}  // namespace

LayoutMode choose_layout(int width, int height) {
  if (height < 14) return LayoutMode::Narrow;
  if (width >= 110) return LayoutMode::Wide;
  if (width >= 80) return LayoutMode::Medium;
  return LayoutMode::Narrow;
}

std::size_t list_rows(int height, LayoutMode layout) {
  // header, separator, panel title, separator, metrics strip(s), separator, footer
  const int chrome = layout == LayoutMode::Narrow ? 8 : 7;
  return static_cast<std::size_t>(std::max(1, height - chrome));
}

const TraceEvent* Controller::selected() const {
  if (view_.empty() || state_.selected_row >= view_.size()) return nullptr;
  return &session_.events()[view_[state_.selected_row]];
}

void Controller::set_session(TraceSession session) {
  session_ = std::move(session);
  evaluate_flags();
  rebuild_view();
}

void Controller::apply_delta(CollectorDelta delta) {
  session_.apply(std::move(delta));
  const auto now = std::chrono::steady_clock::now();
  if (!flags_at_ || now - *flags_at_ >= kFlagInterval) evaluate_flags();
  rebuild_view();
}

void Controller::evaluate_flags() {
  flags_at_ = std::chrono::steady_clock::now();
  flags_.clear();
  if (rules_.empty()) return;
  const auto& events = session_.events();
  for (auto& hit : evaluate_rules(rules_, events).hits)
    flags_[events[hit.event_index].sequence].push_back(std::move(hit));
}

const std::vector<AnomalyHit>* Controller::flags_for(const TraceEvent& event) const {
  const auto it = flags_.find(event.sequence);
  return it == flags_.end() ? nullptr : &it->second;
}

void Controller::rebuild_view() {
  EventFilter parsed = EventFilter::parse(state_.filter);
  if (parsed.valid()) {
    active_filter_ = std::move(parsed);
    state_.filter_error.clear();
  } else {
    state_.filter_error = parsed.error();  // keep showing the last valid result
  }
  view_ = session_.query(active_filter_, state_.sort);
  if (view_.empty()) {
    state_.selected_row = 0;
    state_.scroll_top = 0;
    return;
  }
  const auto& events = session_.events();
  if (!state_.selected_sequence) {
    state_.selected_row = 0;
    state_.selected_sequence = events[view_[0]].sequence;
    return;
  }
  const std::uint64_t wanted = *state_.selected_sequence;
  for (std::size_t row = 0; row < view_.size(); ++row) {
    if (events[view_[row]].sequence == wanted) {
      state_.selected_row = row;
      return;
    }
  }
  // Not visible: either filtered out or overwritten in the ring buffer.
  const bool overwritten = !session_.find_sequence(wanted).has_value();
  std::size_t best = 0;
  std::uint64_t best_gap = UINT64_MAX;
  for (std::size_t row = 0; row < view_.size(); ++row) {
    const std::uint64_t seq = events[view_[row]].sequence;
    const std::uint64_t gap = seq > wanted ? seq - wanted : wanted - seq;
    if (gap < best_gap) {
      best_gap = gap;
      best = row;
    }
  }
  state_.selected_row = overwritten ? best : 0;
  state_.selected_sequence = events[view_[state_.selected_row]].sequence;
  if (overwritten) state_.notice = "Selection overwritten; moved to the nearest retained event";
}

void Controller::select_row(std::size_t row, std::size_t page_rows) {
  if (view_.empty()) return;
  row = std::min(row, view_.size() - 1);
  state_.selected_row = row;
  state_.selected_sequence = session_.events()[view_[row]].sequence;
  page_rows = std::max<std::size_t>(1, page_rows);
  if (row < state_.scroll_top) state_.scroll_top = row;
  if (row >= state_.scroll_top + page_rows) state_.scroll_top = row + 1 - page_rows;
}

bool Controller::apply(const KeyResult& key, LayoutMode layout, std::size_t page_rows) {
  const std::size_t row = state_.selected_row;
  const std::size_t page = std::max<std::size_t>(1, page_rows);
  if (key.action != Action::None) state_.notice.clear();
  switch (key.action) {
    case Action::None:
    case Action::Save:
      break;
    case Action::MoveDown:
      select_row(row + 1, page);
      break;
    case Action::MoveUp:
      select_row(row == 0 ? 0 : row - 1, page);
      break;
    case Action::PageDown:
      select_row(row + page, page);
      break;
    case Action::PageUp:
      select_row(row > page ? row - page : 0, page);
      break;
    case Action::Home:
      select_row(0, page);
      break;
    case Action::End:
      if (!view_.empty()) select_row(view_.size() - 1, page);
      break;
    case Action::NextPanel:
    case Action::PrevPanel: {
      if (layout == LayoutMode::Narrow) {
        static const Panel order[] = {Panel::Operations, Panel::Details, Panel::Metrics};
        int index = 0;
        for (int i = 0; i < 3; ++i)
          if (order[i] == state_.focus) index = i;
        index = (index + (key.action == Action::NextPanel ? 1 : 2)) % 3;
        state_.focus = order[index];
        if (state_.focus != Panel::Operations) state_.right_panel = state_.focus;
      } else {
        state_.focus = state_.focus == Panel::Operations ? state_.right_panel : Panel::Operations;
      }
      break;
    }
    case Action::Inspect:
      if (state_.focus == Panel::Operations) {
        state_.right_panel = Panel::Details;
        state_.focus = Panel::Details;
      }
      break;
    case Action::ToggleMetrics:
      state_.right_panel = state_.right_panel == Panel::Metrics ? Panel::Details : Panel::Metrics;
      if (layout == LayoutMode::Narrow || state_.focus != Panel::Operations)
        state_.focus = state_.right_panel;
      break;
    case Action::CycleSort:
      state_.sort = state_.sort == SortMode::Source         ? SortMode::DurationDesc
                    : state_.sort == SortMode::DurationDesc ? SortMode::Name
                                                            : SortMode::Source;
      rebuild_view();
      select_row(state_.selected_row, page);
      break;
    case Action::OpenFilter:
      state_.mode = Mode::Filter;
      state_.focus = Panel::Operations;
      break;
    case Action::FilterInput:
      if (state_.filter.size() + key.text.size() <= kMaxFilterBytes) {
        state_.filter += key.text;
        rebuild_view();
        select_row(state_.selected_row, page);
      }
      break;
    case Action::FilterBackspace:
      pop_utf8(state_.filter);
      rebuild_view();
      select_row(state_.selected_row, page);
      break;
    case Action::FilterCommit:
      state_.mode = Mode::Normal;
      break;
    case Action::TogglePause:
      state_.paused = !state_.paused;
      break;
    case Action::ToggleHelp:
      state_.mode = state_.mode == Mode::Help ? Mode::Normal : Mode::Help;
      break;
    case Action::Close:
      if (state_.mode == Mode::Help) {
        state_.mode = Mode::Normal;
      } else if (state_.mode == Mode::Filter || !state_.filter.empty()) {
        state_.mode = Mode::Normal;
        state_.filter.clear();
        rebuild_view();
        select_row(state_.selected_row, page);
      } else if (state_.focus != Panel::Operations) {
        state_.focus = Panel::Operations;
        state_.right_panel = Panel::Details;
      }
      break;
    case Action::NextFlagged: {
      const auto& events = session_.events();
      for (std::size_t step = 1; step <= view_.size(); ++step) {
        const std::size_t next = (row + step) % view_.size();
        if (flags_.count(events[view_[next]].sequence)) {
          select_row(next, page);
          return true;
        }
      }
      state_.notice = rules_.empty() ? "No heuristic rules loaded"
                                     : "No event in this view matches a heuristic rule";
      break;
    }
    case Action::Quit:
      return false;
  }
  return true;
}

}  // namespace llm_trace::tui
