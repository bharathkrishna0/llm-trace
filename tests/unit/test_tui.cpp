#include <catch2/catch_test_macros.hpp>
#include <ftxui/component/event.hpp>

#include "llm_trace/synthetic.hpp"
#include "tui/app.hpp"
#include "tui/app_state.hpp"
#include "tui/render.hpp"

using namespace llm_trace;
using namespace llm_trace::tui;
using ftxui::Event;

namespace {

TraceSession demo_session(std::size_t capacity = 4096, int tokens = 2) {
  SyntheticConfig cfg;
  cfg.layers = 2;
  cfg.decode_tokens = tokens;
  cfg.steps_per_second = 0;
  SyntheticAdapter adapter(cfg);
  Collector collector(capacity, adapter.metadata());
  std::atomic<bool> stop{false};
  adapter.run(collector, stop);
  auto snap = collector.snapshot();
  return TraceSession(snap.metadata, std::move(snap.events), snap.stats);
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

KeyResult key(Action a, std::string text = {}) {
  return {a, std::move(text)};
}

}  // namespace

TEST_CASE("layout thresholds follow the design") {
  CHECK(choose_layout(140, 40) == LayoutMode::Wide);
  CHECK(choose_layout(110, 40) == LayoutMode::Wide);
  CHECK(choose_layout(95, 40) == LayoutMode::Medium);
  CHECK(choose_layout(79, 40) == LayoutMode::Narrow);
  CHECK(choose_layout(140, 10) == LayoutMode::Narrow);
  CHECK(list_rows(3, LayoutMode::Wide) == 1);
}

TEST_CASE("key map is centralised and filter mode captures text") {
  CHECK(map_key(Event::Character('j'), Mode::Normal).action == Action::MoveDown);
  CHECK(map_key(Event::ArrowUp, Mode::Normal).action == Action::MoveUp);
  CHECK(map_key(Event::Character('q'), Mode::Normal).action == Action::Quit);
  CHECK(map_key(Event::Escape, Mode::Normal).action == Action::Close);
  const auto typed = map_key(Event::Character('q'), Mode::Filter);
  CHECK(typed.action == Action::FilterInput);
  CHECK(typed.text == "q");
  CHECK(map_key(Event::Character('p'), Mode::Help).action == Action::None);
  CHECK(map_key(Event::Escape, Mode::Help).action == Action::ToggleHelp);
}

TEST_CASE("selection moves, clamps and survives new data") {
  Controller c;
  c.set_session(demo_session());
  REQUIRE(c.selected());
  CHECK(c.selected()->sequence == 0);
  c.apply(key(Action::MoveUp), LayoutMode::Wide, 10);
  CHECK(c.state().selected_row == 0);
  for (int i = 0; i < 5; ++i) c.apply(key(Action::MoveDown), LayoutMode::Wide, 10);
  CHECK(c.selected()->sequence == 5);
  c.apply(key(Action::End), LayoutMode::Wide, 10);
  CHECK(c.state().selected_row == c.view().size() - 1);
  CHECK(c.state().scroll_top == c.view().size() - 10);
  c.apply(key(Action::Home), LayoutMode::Wide, 10);
  c.apply(key(Action::MoveDown), LayoutMode::Wide, 10);
  c.set_session(demo_session(4096, 3));  // more events arrive
  CHECK(c.selected()->sequence == 1);    // selection does not jump to the newest event
}

TEST_CASE("overwritten selection moves to the nearest retained event") {
  Controller c;
  c.set_session(demo_session(4096));
  c.apply(key(Action::MoveDown), LayoutMode::Wide, 10);
  c.set_session(demo_session(8));  // only the last 8 events are retained
  REQUIRE(c.selected());
  CHECK(c.selected()->sequence == c.session().events().front().sequence);
  CHECK(contains(c.state().notice, "overwritten"));
}

TEST_CASE("filter narrows the view and Esc restores it") {
  Controller c;
  c.set_session(demo_session());
  const auto all = c.view().size();
  c.apply(key(Action::OpenFilter), LayoutMode::Wide, 10);
  CHECK(c.state().mode == Mode::Filter);
  for (char ch : std::string("softmax"))
    c.apply(key(Action::FilterInput, std::string(1, ch)), LayoutMode::Wide, 10);
  CHECK(c.view().size() == 2 * 3);  // 2 layers x 3 steps
  CHECK(c.session().events().size() == all);
  c.apply(key(Action::FilterBackspace), LayoutMode::Wide, 10);
  CHECK(c.state().filter == "softma");
  c.apply(key(Action::FilterCommit), LayoutMode::Wide, 10);
  CHECK(c.state().mode == Mode::Normal);
  c.apply(key(Action::Close), LayoutMode::Wide, 10);
  CHECK(c.state().filter.empty());
  CHECK(c.view().size() == all);
}

TEST_CASE("sort cycles through source, duration and name") {
  Controller c;
  c.set_session(demo_session());
  c.apply(key(Action::CycleSort), LayoutMode::Wide, 10);
  CHECK(c.state().sort == SortMode::DurationDesc);
  const auto& ev = c.session().events();
  CHECK(ev[c.view()[0]].duration_ns >= ev[c.view()[1]].duration_ns);
  c.apply(key(Action::CycleSort), LayoutMode::Wide, 10);
  c.apply(key(Action::CycleSort), LayoutMode::Wide, 10);
  CHECK(c.state().sort == SortMode::Source);
}

TEST_CASE("panels cycle in narrow mode and quit returns false") {
  Controller c;
  c.set_session(demo_session());
  c.apply(key(Action::NextPanel), LayoutMode::Narrow, 10);
  CHECK(c.state().focus == Panel::Details);
  c.apply(key(Action::NextPanel), LayoutMode::Narrow, 10);
  CHECK(c.state().focus == Panel::Metrics);
  c.apply(key(Action::PrevPanel), LayoutMode::Narrow, 10);
  CHECK(c.state().focus == Panel::Details);
  c.apply(key(Action::TogglePause), LayoutMode::Narrow, 10);
  CHECK(c.state().paused);
  CHECK_FALSE(c.apply(key(Action::Quit), LayoutMode::Narrow, 10));
}

TEST_CASE("wide frame shows demo label, list, details and metrics strip") {
  Controller c;
  c.set_session(demo_session());
  const auto frame = render_plain(c, 140, 40);
  CHECK(contains(frame, "LLM TRACE"));
  CHECK(contains(frame, "[DEMO]"));
  CHECK(contains(frame, "SYNTHETIC DEMO"));
  CHECK(contains(frame, "OPERATIONS"));
  CHECK(contains(frame, "SELECTED OPERATION"));
  CHECK(contains(frame, "run_start"));
  CHECK(contains(frame, "METRICS"));
  CHECK(contains(frame, "CAPTURE STOPPED"));
}

TEST_CASE("narrow frame shows one panel") {
  Controller c;
  c.set_session(demo_session());
  const auto frame = render_plain(c, 60, 30);
  CHECK(contains(frame, "OPERATIONS"));
  CHECK_FALSE(contains(frame, "SELECTED OPERATION"));
  c.apply(key(Action::NextPanel), LayoutMode::Narrow, 10);
  CHECK(contains(render_plain(c, 60, 30), "SELECTED OPERATION"));
}

TEST_CASE("attention operations state that values are not exposed") {
  Controller c;
  c.set_session(demo_session());
  for (char ch : std::string("scores"))
    c.apply(key(Action::FilterInput, std::string(1, ch)), LayoutMode::Wide, 10);
  const auto frame = render_plain(c, 140, 45);
  CHECK(contains(frame, "ATTENTION VALUES"));
  CHECK(contains(frame, "Not exposed by adapter"));
}

TEST_CASE("ascii mode renders only ASCII bytes") {
  Controller c(DisplayOptions{false, true});
  c.set_session(demo_session());
  for (int w : {140, 95, 60}) {
    const auto frame = render_plain(c, w, 35);
    for (unsigned char ch : frame) CHECK(ch < 128);
  }
  c.apply(key(Action::ToggleHelp), LayoutMode::Wide, 10);
  for (unsigned char ch : render_plain(c, 140, 40)) CHECK(ch < 128);
}

TEST_CASE("help overlay lists shortcuts and capabilities") {
  Controller c;
  c.set_session(demo_session());
  c.apply(key(Action::ToggleHelp), LayoutMode::Wide, 10);
  const auto frame = render_plain(c, 140, 45);
  CHECK(contains(frame, "HELP"));
  CHECK(contains(frame, "attention values"));
  CHECK(contains(frame, "Esc or ? closes this overlay"));
}

TEST_CASE("empty and no-match states are explicit") {
  Controller c;
  CHECK(contains(render_plain(c, 120, 30), "No trace loaded"));
  c.set_session(demo_session());
  c.apply(key(Action::OpenFilter), LayoutMode::Wide, 10);
  for (char ch : std::string("zzz"))
    c.apply(key(Action::FilterInput, std::string(1, ch)), LayoutMode::Wide, 10);
  CHECK(contains(render_plain(c, 120, 30), "No events match \"zzz\""));
}

TEST_CASE("paused live view says capture continues") {
  SyntheticConfig cfg;
  cfg.layers = 1;
  cfg.decode_tokens = 0;
  Collector collector(64, synthetic_metadata(cfg));
  collector.set_capturing(true);
  for (auto& e : generate_synthetic(cfg)) collector.submit(e);
  auto snap = collector.snapshot();
  Controller c;
  c.set_session(TraceSession(snap.metadata, std::move(snap.events), snap.stats));
  CHECK(contains(render_plain(c, 140, 30), "VIEW LIVE"));
  c.apply(key(Action::TogglePause), LayoutMode::Wide, 10);
  CHECK(contains(render_plain(c, 140, 30), "CAPTURE CONTINUES"));
}
