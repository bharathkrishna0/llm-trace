#include "tui/app.hpp"

#include <atomic>
#include <chrono>
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/screen/terminal.hpp>
#include <thread>

#include "llm_trace/collector.hpp"
#include "tui/app_state.hpp"
#include "tui/render.hpp"

namespace llm_trace::tui {

namespace {

// UI refresh cadence (DESIGN.md 10). Events are coalesced into one refresh per tick.
constexpr auto kRefreshInterval = std::chrono::milliseconds(125);

TraceSession session_from(const Collector& collector) {
  auto snap = collector.snapshot();
  return TraceSession(std::move(snap.metadata), std::move(snap.events), snap.stats);
}

TraceMetadata demo_metadata(const DemoOptions& options) {
  TraceMetadata meta = synthetic_metadata(options.synthetic);
  meta.buffer_capacity = options.buffer_capacity;
  return meta;
}

}  // namespace

int run_demo(const DemoOptions& options) {
  SyntheticAdapter adapter(options.synthetic);
  Collector collector(options.buffer_capacity, demo_metadata(options));
  std::atomic<bool> stop{false};
  std::thread worker([&] { adapter.run(collector, stop); });

  Controller controller(options.display);
  std::uint64_t shown_version = UINT64_MAX;
  auto refresh = [&] {
    if (controller.state().paused) return;
    const auto version = collector.stats().version;
    if (version == shown_version) return;
    shown_version = version;
    controller.set_session(session_from(collector));
  };
  refresh();

  auto screen = ftxui::ScreenInteractive::Fullscreen();
  auto renderer = ftxui::Renderer([&] {
    const auto size = ftxui::Terminal::Size();
    return render_app(controller, size.dimx, size.dimy);
  });
  auto component = ftxui::CatchEvent(renderer, [&](const ftxui::Event& event) {
    if (event == ftxui::Event::Custom) {
      refresh();
      return true;
    }
    const KeyResult key = map_key(event, controller.state().mode);
    if (key.action == Action::None) return false;
    const auto size = ftxui::Terminal::Size();
    const LayoutMode layout = choose_layout(size.dimx, size.dimy);
    if (!controller.apply(key, layout, list_rows(size.dimy, layout))) {
      screen.Exit();
      return true;
    }
    refresh();  // no-op while paused or when nothing changed
    return true;
  });

  std::thread ticker([&] {
    std::uint64_t posted = UINT64_MAX;
    while (!stop.load()) {
      std::this_thread::sleep_for(kRefreshInterval);
      const auto version = collector.stats().version;
      if (version != posted) {
        posted = version;
        screen.Post(ftxui::Event::Custom);
      }
    }
  });

  screen.Loop(component);
  stop.store(true);
  worker.join();
  ticker.join();
  return 0;
}

std::string render_demo_frame(const DemoOptions& options, int width, int height, bool styled) {
  SyntheticConfig config = options.synthetic;
  config.steps_per_second = 0;
  SyntheticAdapter adapter(config);
  Collector collector(options.buffer_capacity, demo_metadata(options));
  std::atomic<bool> stop{false};
  adapter.run(collector, stop);
  Controller controller(options.display);
  controller.set_session(session_from(collector));
  return styled ? render_styled(controller, width, height)
                : render_plain(controller, width, height);
}

std::string render_empty_frame(const DisplayOptions& display, int width, int height, bool styled) {
  Controller controller(display);
  return styled ? render_styled(controller, width, height)
                : render_plain(controller, width, height);
}

}  // namespace llm_trace::tui
