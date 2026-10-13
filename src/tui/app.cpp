#include "tui/app.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/screen/terminal.hpp>
#include <functional>
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

TraceSession session_from(const LoadedTrace& trace) {
  constexpr std::size_t kMaxShownWarnings = 20;
  TraceMetadata meta = trace.metadata;
  for (std::size_t i = 0; i < trace.warnings.size() && i < kMaxShownWarnings; ++i)
    meta.notes.push_back("Import: " + trace.warnings[i]);
  if (trace.warnings.size() > kMaxShownWarnings)
    meta.notes.push_back("Import: " + std::to_string(trace.warnings.size() - kMaxShownWarnings) +
                         " more warning(s) not shown");
  return TraceSession(std::move(meta), trace.events);
}

std::string utc_now() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char buf[32] = {};
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

// Writes the collector's retained events. Returns a notice for the status line.
std::string save_collector(const Collector& collector, const std::string& path) {
  CollectorSnapshot snap = collector.snapshot();
  TraceMetadata meta = std::move(snap.metadata);
  meta.overwritten_events = snap.stats.overwritten;
  meta.rejected_events = snap.stats.rejected;
  meta.buffer_capacity = snap.stats.capacity;
  if (meta.created_utc.empty()) meta.created_utc = utc_now();
  if (snap.stats.capturing) meta.notes.push_back("Saved while capture was still running.");
  if (snap.stats.overwritten > 0)
    meta.notes.push_back(std::to_string(snap.stats.overwritten) +
                         " older event(s) were overwritten in the ring buffer before saving.");
  write_jsonl(path, meta, snap.events);
  return "Saved " + std::to_string(snap.events.size()) + " events to " + path;
}

std::string try_save(const std::function<std::string()>& save) {
  try {
    return save();
  } catch (const std::exception& e) {
    return std::string("Save failed: ") + e.what();
  }
}

// Shared event loop. `refresh` pulls new data, `save` writes the trace.
void run_loop(Controller& controller, const std::function<void()>& refresh,
              const std::function<std::string()>& save,
              const std::function<void(ftxui::ScreenInteractive&)>& on_start) {
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
    if (key.action == Action::Save) {
      controller.set_notice(try_save(save));
      return true;
    }
    const auto size = ftxui::Terminal::Size();
    const LayoutMode layout = choose_layout(size.dimx, size.dimy);
    if (!controller.apply(key, layout, list_rows(size.dimy, layout))) {
      screen.Exit();
      return true;
    }
    refresh();  // no-op while paused or when nothing changed
    return true;
  });
  on_start(screen);
  screen.Loop(component);
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
  std::uint64_t next_sequence = 0;
  auto refresh = [&] {
    if (controller.state().paused) return;
    const auto version = collector.stats().version;
    if (version == shown_version) return;
    shown_version = version;
    CollectorDelta delta = collector.delta_since(next_sequence);
    next_sequence = delta.first_retained_sequence + delta.stats.retained;
    controller.apply_delta(std::move(delta));
  };
  refresh();
  const std::string save_path = options.output.value_or(default_output_path(collector.metadata()));

  std::thread ticker;
  run_loop(
      controller, refresh, [&] { return save_collector(collector, save_path); },
      [&](ftxui::ScreenInteractive& screen) {
        ticker = std::thread([&] {
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
      });
  stop.store(true);
  worker.join();
  ticker.join();

  if (!options.output) return 0;
  try {
    std::fprintf(stderr, "llm-trace: %s\n", save_collector(collector, *options.output).c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "llm-trace: could not save %s: %s\n", options.output->c_str(), e.what());
    return 1;
  }
  return 0;
}

int run_replay(const LoadedTrace& trace, const ReplayOptions& options) {
  Controller controller(options.display);
  controller.set_session(session_from(trace));
  const std::string save_path = options.output.value_or(default_output_path(trace.metadata));
  auto save = [&] {
    write_jsonl(save_path, trace.metadata, trace.events);
    return "Saved " + std::to_string(trace.events.size()) + " events to " + save_path;
  };
  run_loop(controller, [] {}, save, [](ftxui::ScreenInteractive&) {});
  return 0;
}

std::string render_replay_frame(const LoadedTrace& trace, const DisplayOptions& display, int width,
                                int height, bool styled) {
  Controller controller(display);
  controller.set_session(session_from(trace));
  return styled ? render_styled(controller, width, height)
                : render_plain(controller, width, height);
}

std::string default_output_path(const TraceMetadata& metadata) {
  std::string id;
  for (const char ch : metadata.run_id) {
    const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                    (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' || ch == '.';
    id.push_back(ok ? ch : '_');
    if (id.size() >= 64) break;
  }
  if (id.empty() || id.find_first_not_of('.') == std::string::npos) id = "trace";
  return "llm-trace-" + id + ".jsonl";
}

std::string render_demo_frame(const DemoOptions& options, int width, int height, bool styled) {
  SyntheticConfig config = options.synthetic;
  config.steps_per_second = 0;
  SyntheticAdapter adapter(config);
  Collector collector(options.buffer_capacity, demo_metadata(options));
  std::atomic<bool> stop{false};
  adapter.run(collector, stop);
  if (options.output) save_collector(collector, *options.output);
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
