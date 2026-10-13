#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <vector>

#include "llm_trace/collector.hpp"

using namespace llm_trace;

namespace {
TraceEvent op(const std::string& name, std::uint64_t ts, std::uint64_t dur) {
  TraceEvent e;
  e.name = name;
  e.timestamp_ns = ts;
  e.duration_ns = dur;
  return e;
}
}  // namespace

TEST_CASE("collector assigns sequence numbers and stores events in order") {
  Collector collector(8);
  CHECK(collector.submit(op("a", 0, 10)));
  CHECK(collector.submit(op("b", 10, 10)));
  const auto snap = collector.snapshot();
  REQUIRE(snap.events.size() == 2);
  CHECK(snap.events[0].sequence == 0);
  CHECK(snap.events[1].sequence == 1);
  CHECK(snap.events[1].name == "b");
  CHECK(snap.metadata.buffer_capacity == 8);
}

TEST_CASE("collector rejects invalid events and counts them") {
  Collector collector(4);
  CHECK_FALSE(collector.submit(op("", 0, 1)));
  TraceEvent bad_shape = op("x", 0, 1);
  bad_shape.inputs.push_back({"t", std::vector<std::int64_t>{1, -5}, "float32", {}, {}});
  CHECK_FALSE(collector.submit(bad_shape));
  CHECK(collector.submit(op("ok", 0, 1)));
  const auto snap = collector.snapshot();
  CHECK(snap.stats.rejected == 2);
  CHECK(snap.stats.accepted == 1);
  CHECK(snap.events.front().sequence == 0);  // rejected events do not consume sequence numbers
  CHECK(snap.rejection_reasons.size() == 2);
}

TEST_CASE("collector overwrites oldest events and reports it") {
  Collector collector(3);
  for (int i = 0; i < 10; ++i) collector.submit(op("op" + std::to_string(i), i * 10u, 5));
  const auto snap = collector.snapshot();
  CHECK(snap.stats.retained == 3);
  CHECK(snap.stats.overwritten == 7);
  CHECK(snap.metadata.overwritten_events == 7);
  CHECK(snap.events.front().sequence == 7);
  CHECK(snap.events.back().sequence == 9);
}

TEST_CASE("collector version changes on every accepted event") {
  Collector collector(3);
  const auto v0 = collector.stats().version;
  collector.submit(op("a", 0, 1));
  CHECK(collector.stats().version > v0);
}

TEST_CASE("collector now_ns is monotonic") {
  Collector collector(1);
  const auto a = collector.now_ns();
  const auto b = collector.now_ns();
  CHECK(b >= a);
}

TEST_CASE("collector is safe under concurrent submission") {
  Collector collector(1000);
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t)
    threads.emplace_back([&collector] {
      for (int i = 0; i < 2500; ++i) collector.submit(op("op", 0, 1));
    });
  for (auto& thread : threads) thread.join();
  const auto stats = collector.stats();
  CHECK(stats.accepted == 10000);
  CHECK(stats.retained == 1000);
  CHECK(stats.overwritten == 9000);
  const auto snap = collector.snapshot();
  for (std::size_t i = 1; i < snap.events.size(); ++i)
    CHECK(snap.events[i].sequence == snap.events[i - 1].sequence + 1);
}

TEST_CASE("collector delta returns only new events and the retained window") {
  Collector collector(4);
  for (int i = 0; i < 3; ++i) collector.submit(op("a" + std::to_string(i), i * 10u, 1));
  auto d = collector.delta_since(0);
  CHECK(d.events.size() == 3);
  CHECK(d.first_retained_sequence == 0);
  const std::uint64_t next = d.first_retained_sequence + d.stats.retained;
  CHECK(next == 3);
  CHECK(collector.delta_since(next).events.empty());

  for (int i = 3; i < 10; ++i) collector.submit(op("a" + std::to_string(i), i * 10u, 1));
  d = collector.delta_since(next);  // seq 3..5 were overwritten
  CHECK(d.first_retained_sequence == 6);
  REQUIRE(d.events.size() == 4);
  CHECK(d.events.front().sequence == 6);
  CHECK(d.events.back().sequence == 9);
  CHECK(d.metadata.overwritten_events == 6);
}

TEST_CASE("collector submit_batch applies the same rules as submit") {
  Collector collector(4);
  std::vector<TraceEvent> batch;
  for (int i = 0; i < 6; ++i) batch.push_back(op("b" + std::to_string(i), i * 10u, 1));
  batch[2].name.clear();  // invalid
  CHECK(collector.submit_batch(std::move(batch)) == 5);
  const auto snap = collector.snapshot();
  CHECK(snap.stats.accepted == 5);
  CHECK(snap.stats.rejected == 1);
  CHECK(snap.stats.overwritten == 1);
  REQUIRE(snap.events.size() == 4);
  CHECK(snap.events.front().sequence == 1);
  CHECK(snap.events.back().sequence == 4);
  CHECK(snap.events.back().name == "b5");
}

TEST_CASE("collector submit_batch is safe under concurrent use") {
  Collector collector(1 << 12);
  std::vector<std::thread> pool;
  for (int t = 0; t < 4; ++t)
    pool.emplace_back([&collector, t] {
      for (int b = 0; b < 50; ++b) {
        std::vector<TraceEvent> batch;
        for (int i = 0; i < 10; ++i) batch.push_back(op("t" + std::to_string(t), 0, 1));
        collector.submit_batch(std::move(batch));
      }
    });
  for (auto& th : pool) th.join();
  const auto snap = collector.snapshot();
  CHECK(snap.stats.accepted == 2000);
  REQUIRE(snap.events.size() == 2000);
  for (std::size_t i = 0; i < snap.events.size(); ++i) CHECK(snap.events[i].sequence == i);
}
