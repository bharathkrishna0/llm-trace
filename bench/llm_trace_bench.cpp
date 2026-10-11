// Micro-benchmarks for llm-trace's own costs: storing, saving, loading and rendering
// events. Each case runs `reps` times and reports the median; nothing is extrapolated.
//
// Usage: llm_trace_bench [--reps N] [--events N]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "llm_trace/collector.hpp"
#include "llm_trace/onnx_profile_adapter.hpp"
#include "llm_trace/ring_buffer.hpp"
#include "llm_trace/synthetic.hpp"
#include "llm_trace/trace_io.hpp"
#include "tui/app.hpp"

using namespace llm_trace;
using Clock = std::chrono::steady_clock;

namespace {

double median_seconds(int reps, const std::function<void()>& body) {
  std::vector<double> s;
  for (int i = 0; i < reps; ++i) {
    const auto t0 = Clock::now();
    body();
    s.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
  }
  std::sort(s.begin(), s.end());
  return s[s.size() / 2];
}

void report(const char* name, double seconds, std::size_t items, const char* unit) {
  std::printf("%-44s %10.1f ns/%s  %12.0f %s/s\n", name, seconds * 1e9 / static_cast<double>(items),
              unit, static_cast<double>(items) / seconds, unit);
}

std::vector<TraceEvent> make_events(std::size_t n) {
  SyntheticConfig cfg;
  cfg.layers = 12;
  cfg.decode_tokens = 1024;
  std::vector<TraceEvent> base = generate_synthetic(cfg);
  std::vector<TraceEvent> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    TraceEvent e = base[i % base.size()];
    e.sequence = i;
    e.timestamp_ns = i * 1000;
    out.push_back(std::move(e));
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  int reps = 7;
  std::size_t n = 200'000;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--reps")
      reps = std::max(1, std::atoi(argv[i + 1]));
    else if (a == "--events")
      n = std::max<std::size_t>(1000, std::strtoull(argv[i + 1], nullptr, 10));
  }
  const std::vector<TraceEvent> events = make_events(n);
  const TraceMetadata meta = synthetic_metadata(SyntheticConfig{});
  std::printf("llm-trace bench: %zu synthetic events, median of %d runs, %u hardware threads\n", n,
              reps, std::thread::hardware_concurrency());
  std::printf("sizeof(TraceEvent) = %zu bytes (heap-owned strings/tensors not included)\n\n",
              sizeof(TraceEvent));

  {
    RingBuffer<TraceEvent> rb(65536);
    report("ring buffer push (copy, cap 65536)",
           median_seconds(reps,
                          [&] {
                            for (const auto& e : events) rb.push(e);
                          }),
           n, "event");
  }
  {
    Collector c(65536, meta);
    c.set_capturing(true);
    report("collector submit, 1 thread",
           median_seconds(reps,
                          [&] {
                            for (const auto& e : events) c.submit(e);
                          }),
           n, "event");
  }
  {
    Collector c(65536, meta);
    c.set_capturing(true);
    const int threads = 4;
    report("collector submit, 4 threads (total)",
           median_seconds(reps,
                          [&] {
                            std::vector<std::thread> pool;
                            for (int t = 0; t < threads; ++t)
                              pool.emplace_back([&, t] {
                                for (std::size_t i = static_cast<std::size_t>(t); i < n;
                                     i += threads)
                                  c.submit(events[i]);
                              });
                            for (auto& th : pool) th.join();
                          }),
           n, "event");
  }

  std::string jsonl;
  report("write_jsonl (to memory)",
         median_seconds(reps,
                        [&] {
                          std::ostringstream out;
                          write_jsonl(out, meta, events);
                          jsonl = out.str();
                        }),
         n, "event");
  std::printf("%-44s %10.1f bytes/event\n", "  JSONL size",
              static_cast<double>(jsonl.size()) / static_cast<double>(n));
  report("read_jsonl (from memory)",
         median_seconds(reps,
                        [&] {
                          std::istringstream in(jsonl);
                          const LoadedTrace t = read_jsonl(in);
                          if (t.events.size() != n) std::abort();
                        }),
         n, "event");

  const std::string profile = std::string(LLM_TRACE_SOURCE_DIR) +
                              "/tests/fixtures/onnx_profiles/tiny_transformer_ort-1.20.1.json";
  std::ifstream pf(profile, std::ios::binary);
  const std::string profile_text((std::istreambuf_iterator<char>(pf)), {});
  std::size_t profile_events = 0;
  const double pt = median_seconds(reps, [&] {
    std::istringstream in(profile_text);
    profile_events = OnnxProfileAdapter().parse(in, "fixture").events.size();
  });
  report("onnx profile parse (270 records fixture)", pt, profile_events, "event");

  for (const std::size_t frame_events : {std::size_t{10'000}, std::size_t{65'536}}) {
    LoadedTrace t;
    t.metadata = meta;
    t.events.assign(events.begin(),
                    events.begin() + static_cast<std::ptrdiff_t>(std::min(frame_events, n)));
    tui::DisplayOptions d;
    const double ft = median_seconds(reps, [&] { tui::render_replay_frame(t, d, 160, 45, true); });
    std::printf("%-44s %10.2f ms/frame\n",
                (std::string("render 160x45 frame, ") + std::to_string(t.events.size()) + " events")
                    .c_str(),
                ft * 1e3);
  }
  return 0;
}
