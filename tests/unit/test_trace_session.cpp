#include <catch2/catch_test_macros.hpp>
#include <limits>

#include "llm_trace/collector.hpp"
#include "llm_trace/trace_session.hpp"

using namespace llm_trace;

namespace {

TraceEvent op(std::uint64_t seq, const std::string& name, Category category, std::uint64_t ts,
              std::optional<std::uint64_t> dur, std::optional<std::int64_t> layer = {}) {
  TraceEvent e;
  e.sequence = seq;
  e.name = name;
  e.category = category;
  e.timestamp_ns = ts;
  e.duration_ns = dur;
  e.layer_index = layer;
  if (layer) e.parent_id = "layer-" + std::to_string(*layer);
  return e;
}

TraceSession sample_session() {
  std::vector<TraceEvent> events;
  events.push_back(op(0, "embedding", Category::Embedding, 100, 50));
  events.push_back(op(1, "decoder.layer.0.attention", Category::Attention, 150, 400, 0));
  events.push_back(op(2, "decoder.layer.0.mlp", Category::Mlp, 550, 300, 0));
  events.push_back(op(3, "decoder.layer.1.attention", Category::Attention, 850, 500, 1));
  events.push_back(op(4, "lm_head", Category::Output, 1350, std::nullopt));
  TraceEvent warn;
  warn.sequence = 5;
  warn.type = EventType::Warning;
  warn.name = "adapter_warning";
  warn.message = "something odd";
  warn.timestamp_ns = 1400;
  events.push_back(warn);
  events.push_back(op(6, "model_run", Category::Session, 90, 1300));
  return TraceSession(TraceMetadata{}, std::move(events));
}

}  // namespace

TEST_CASE("metrics aggregate only timed non-session operations") {
  const auto session = sample_session();
  const auto& m = session.metrics();
  CHECK(m.event_count == 7);
  CHECK(m.operation_count == 4);
  CHECK(m.missing_duration_count == 1);
  CHECK(m.warning_count == 1);
  CHECK(m.session_event_count == 1);
  CHECK(m.summed_operation_ns == 1250);
  REQUIRE(m.source_run_total_ns);
  CHECK(*m.source_run_total_ns == 1300);
  REQUIRE(m.first_timestamp_ns);
  CHECK(*m.first_timestamp_ns == 90);
  REQUIRE(m.observed_span_ns);
  CHECK(*m.observed_span_ns == 1390 - 90);  // model_run ends last at 90 + 1300
}

TEST_CASE("top operations are sorted by duration") {
  const auto session = sample_session();
  const auto& top = session.metrics().top_operations;
  REQUIRE(top.size() == 4);
  CHECK(session.events()[top[0]].name == "decoder.layer.1.attention");
  CHECK(session.events()[top[1]].name == "decoder.layer.0.attention");
  CHECK(session.events()[top[3]].name == "embedding");
}

TEST_CASE("category aggregation sums durations") {
  const auto session = sample_session();
  const auto& cats = session.metrics().by_category;
  REQUIRE_FALSE(cats.empty());
  CHECK(cats[0].key == "attention");
  CHECK(cats[0].count == 2);
  CHECK(cats[0].total_ns == 900);
  CHECK(cats[0].max_ns == 500);
}

TEST_CASE("empty session has no observed span") {
  TraceSession session;
  CHECK(session.empty());
  CHECK_FALSE(session.metrics().observed_span_ns);
  CHECK(session.query("", SortMode::DurationDesc).empty());
}

TEST_CASE("filter matches name category and layer without mutating events") {
  const auto session = sample_session();
  CHECK(session.query("attention", SortMode::Source).size() == 2);
  CHECK(session.query("MLP", SortMode::Source).size() == 1);
  CHECK(session.query("1", SortMode::Source).size() == 1);  // layer index 1
  CHECK(session.query("warning", SortMode::Source).size() == 1);
  CHECK(session.query("nothing-matches", SortMode::Source).empty());
  CHECK(session.events().size() == 7);
}

TEST_CASE("sorting by duration puts missing durations last") {
  const auto session = sample_session();
  const auto idx = session.query("", SortMode::DurationDesc);
  CHECK(session.events()[idx.front()].name == "model_run");
  CHECK_FALSE(session.events()[idx.back()].duration_ns.has_value());
}

TEST_CASE("sorting by name is stable and alphabetical") {
  const auto session = sample_session();
  const auto idx = session.query("decoder", SortMode::Name);
  REQUIRE(idx.size() == 3);
  CHECK(session.events()[idx[0]].name == "decoder.layer.0.attention");
  CHECK(session.events()[idx[1]].name == "decoder.layer.0.mlp");
}

TEST_CASE("lookup by sequence and parent") {
  const auto session = sample_session();
  CHECK(session.find_sequence(3) == std::optional<std::size_t>{3});
  CHECK_FALSE(session.find_sequence(99));
  CHECK(session.children_of("layer-0").size() == 2);
  CHECK(session.children_of("layer-9").empty());
}

TEST_CASE("warnings are de-duplicated and include metadata notes") {
  std::vector<TraceEvent> events;
  for (int i = 0; i < 3; ++i) {
    TraceEvent w;
    w.sequence = static_cast<std::uint64_t>(i);
    w.type = EventType::Warning;
    w.name = "warn";
    w.message = "same message";
    events.push_back(w);
  }
  TraceMetadata meta;
  meta.notes.push_back("note");
  TraceSession session(meta, events);
  CHECK(session.warnings() == std::vector<std::string>{"note", "same message"});
}

TEST_CASE("event validation") {
  TraceEvent e;
  CHECK_FALSE(validate(e).empty());
  e.name = "ok";
  CHECK(validate(e).empty());
  e.duration_ns = std::numeric_limits<std::uint64_t>::max();
  e.timestamp_ns = 10;
  CHECK_FALSE(validate(e).empty());
}

TEST_CASE("enum string round-trips") {
  for (auto type : {EventType::RunStart, EventType::Operation, EventType::DroppedEvents})
    CHECK(parse_event_type(to_string(type)) == type);
  for (auto cat : {Category::Attention, Category::Session, Category::Other})
    CHECK(parse_category(to_string(cat)) == cat);
  CHECK_FALSE(parse_event_type("nope"));
  CHECK(element_count_of({1, 12, 4096}) == 49152);
  CHECK(element_count_of({-1, 4}) == 0);
  CHECK(dtype_size_bytes("float16") == std::optional<std::uint64_t>{2});
  CHECK_FALSE(dtype_size_bytes("string"));
}

TEST_CASE("incremental apply matches a full rebuild") {
  Collector collector(16);
  TraceSession live;
  std::uint64_t next = 0;
  for (int round = 0; round < 5; ++round) {
    for (int i = 0; i < 7; ++i) {
      const int k = round * 7 + i;
      collector.submit(
          op(0, "op" + std::to_string(k % 5), k % 2 ? Category::Mlp : Category::Attention,
             static_cast<std::uint64_t>(k) * 10, static_cast<std::uint64_t>(k + 1), k % 3));
    }
    auto delta = collector.delta_since(next);
    next = delta.first_retained_sequence + delta.stats.retained;
    live.apply(std::move(delta));

    auto snap = collector.snapshot();
    const TraceSession full(snap.metadata, std::move(snap.events), snap.stats);
    REQUIRE(live.events().size() == full.events().size());
    CHECK(live.events().front().sequence == full.events().front().sequence);
    CHECK(live.metrics().summed_operation_ns == full.metrics().summed_operation_ns);
    CHECK(live.metrics().by_name.size() == full.metrics().by_name.size());
    CHECK(live.metrics().by_name.front().key == full.metrics().by_name.front().key);
    CHECK(live.query("layer:1", SortMode::DurationDesc) ==
          full.query("layer:1", SortMode::DurationDesc));
    CHECK(live.find_sequence(full.events().back().sequence) == full.events().size() - 1);
  }
}

TEST_CASE("structured query uses the shared filter") {
  const auto s = sample_session();
  CHECK(s.query("cat:attention", SortMode::Source).size() == 2);
  CHECK(s.query("dur:missing type:operation", SortMode::Source).size() == 1);
  CHECK(s.query("layer:bogus", SortMode::Source).empty());
}
