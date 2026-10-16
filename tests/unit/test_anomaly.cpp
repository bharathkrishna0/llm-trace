#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

#include "llm_trace/anomaly.hpp"

using namespace llm_trace;

namespace {

TraceEvent op(const std::string& name, std::optional<std::uint64_t> dur,
              std::vector<std::int64_t> shape = {1, 1, 8}) {
  TraceEvent e;
  e.name = name;
  e.category = Category::MatMul;
  e.op_type = "MatMul";
  e.duration_ns = dur;
  TensorMetadata t;
  t.name = "x";
  t.shape = std::move(shape);
  e.inputs.push_back(t);
  return e;
}

AnomalyRule p95_rule(double factor = 3.0, std::size_t min_calls = 20) {
  AnomalyRule r;
  r.id = "p95";
  r.kind = AnomalyRule::Kind::P95Multiple;
  r.factor = factor;
  r.min_calls = min_calls;
  r.min_ns = 0;
  return r;
}

std::vector<TraceEvent> steady(const std::string& name, std::size_t n, std::uint64_t ns) {
  std::vector<TraceEvent> events;
  for (std::size_t i = 0; i < n; ++i) events.push_back(op(name, ns));
  return events;
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

TEST_CASE("p95_multiple flags an outlier against its own op") {
  auto events = steady("mm", 40, 1000);
  events.push_back(op("mm", 5000));
  const auto report = evaluate_rules({p95_rule()}, events);
  REQUIRE(report.hits.size() == 1);
  CHECK(report.hits[0].event_index == 40);
  CHECK(report.hits[0].rule_id == "p95");
  CHECK(contains(report.hits[0].detail, "5.0x p95"));
}

TEST_CASE("p95_multiple needs min_calls and respects the duration floor") {
  auto events = steady("mm", 30, 1000);
  events.push_back(op("mm", 9000));
  CHECK(evaluate_rules({p95_rule(3.0, 40)}, events).hits.empty());
  CHECK(evaluate_rules({p95_rule(3.0, 20)}, events).hits.size() == 1);
  auto floored = p95_rule(3.0, 20);
  floored.min_ns = 10'000;
  CHECK(evaluate_rules({floored}, events).hits.empty());
}

TEST_CASE("p95_multiple groups by input shape so prefill is not compared with decode") {
  auto events = steady("mm", 30, 1000);
  events.push_back(op("mm", 20000, {1, 16, 8}));  // different work, own group of 1
  CHECK(evaluate_rules({p95_rule()}, events).hits.empty());
}

TEST_CASE("p95_multiple group op pools ops of the same type") {
  auto events = steady("a", 15, 1000);
  for (auto& e : steady("b", 15, 1000)) events.push_back(e);
  events.push_back(op("c", 8000));
  auto by_op = p95_rule();
  by_op.group_by_op = true;
  CHECK(evaluate_rules({p95_rule()}, events).hits.empty());
  REQUIRE(evaluate_rules({by_op}, events).hits.size() == 1);
}

TEST_CASE("share rule flags one op that dominates summed time") {
  std::vector<TraceEvent> events = steady("small", 4, 100);
  events.push_back(op("big", 600));
  AnomalyRule r;
  r.id = "share";
  r.kind = AnomalyRule::Kind::Share;
  r.above = 0.5;
  const auto report = evaluate_rules({r}, events);
  REQUIRE(report.hits.size() == 1);
  CHECK(report.hits[0].event_index == 4);
}

TEST_CASE("missing durations are skipped and counted not treated as zero") {
  std::vector<TraceEvent> events = steady("mm", 25, 1000);
  events.push_back(op("mm", std::nullopt));
  AnomalyRule share;
  share.id = "share";
  share.kind = AnomalyRule::Kind::Share;
  share.above = 0.0;
  const auto report = evaluate_rules({p95_rule(), share}, events);
  CHECK(report.skipped_missing_duration == 1);
  for (const auto& h : report.hits) CHECK(h.event_index != 25);
}

TEST_CASE("match rules use the filter syntax and scope limits every rule") {
  std::vector<TraceEvent> events = steady("mm", 3, 1000);
  events.push_back(op("slow", 9'000'000));
  events[0].layer_index = 1;
  AnomalyRule m;
  m.id = "over5ms";
  m.kind = AnomalyRule::Kind::Match;
  m.filter = "dur>5ms";
  auto report = evaluate_rules({m}, events);
  REQUIRE(report.hits.size() == 1);
  CHECK(report.hits[0].event_index == 3);
  m.scope = "layer:1";
  CHECK(evaluate_rules({m}, events).hits.empty());
}

TEST_CASE("hits are ordered by event then rule") {
  std::vector<TraceEvent> events = steady("mm", 2, 10);
  events.push_back(op("big", 1000));
  AnomalyRule a;
  a.id = "a";
  a.kind = AnomalyRule::Kind::Match;
  a.filter = "dur>0ns";
  AnomalyRule b = a;
  b.id = "b";
  const auto report = evaluate_rules({a, b}, events);
  REQUIRE(report.hits.size() == 6);
  CHECK(report.hits[0].event_index == 0);
  CHECK(report.hits[0].rule_id == "a");
  CHECK(report.hits[1].rule_id == "b");
  CHECK(report.hits[5].event_index == 2);
}

TEST_CASE("rules JSON parses every kind") {
  const auto rules = parse_rules_json(R"({"rules": [
    {"id": "slow", "kind": "p95_multiple", "factor": 4, "min_calls": 10, "group": "op",
     "min_us": 50, "scope": "cat:attention"},
    {"id": "share", "kind": "share", "above": 0.4},
    {"id": "big", "kind": "match", "filter": "op:MatMul dur>5ms"}]})");
  REQUIRE(rules.size() == 3);
  CHECK(rules[0].kind == AnomalyRule::Kind::P95Multiple);
  CHECK(rules[0].factor == 4.0);
  CHECK(rules[0].min_calls == 10);
  CHECK(rules[0].group_by_op);
  CHECK(rules[0].min_ns == 50'000);
  CHECK(rules[0].scope == "cat:attention");
  CHECK(rules[1].above == 0.4);
  CHECK(rules[2].filter == "op:MatMul dur>5ms");
  CHECK(contains(describe_rule(rules[2]), "big"));
}

TEST_CASE("rules JSON rejects invalid input with the rule named") {
  auto fails_with = [](const std::string& text, const std::string& needle) {
    try {
      parse_rules_json(text);
    } catch (const std::runtime_error& e) {
      return contains(e.what(), needle);
    }
    return false;
  };
  CHECK(fails_with("not json", "not valid JSON"));
  CHECK(fails_with(R"({"rule": []})", "\"rules\" array"));
  CHECK(fails_with(R"({"rules": [{"kind": "share"}]})", "missing \"id\""));
  CHECK(fails_with(R"({"rules": [{"id": "x", "kind": "magic"}]})", "unknown kind"));
  CHECK(fails_with(R"({"rules": [{"id": "x", "kind": "share", "above": 2}]})", "\"above\""));
  CHECK(fails_with(R"({"rules": [{"id": "x", "kind": "match", "filter": "lyer:3"}]})",
                   "rule 1 (x) filter"));
  CHECK(fails_with(R"({"rules": [{"id": "x", "kind": "share", "colour": 1}]})", "unknown field"));
  CHECK(fails_with(R"({"rules": [{"id": "x", "kind": "share"}, {"id": "x", "kind": "share"}]})",
                   "duplicate id"));
  CHECK(fails_with(R"({"rules": [{"id": "x", "kind": "p95_multiple", "group": "layer"}]})",
                   "\"group\""));
}

TEST_CASE("check report labels heuristics and synthetic data") {
  auto events = steady("mm", 30, 1000);
  events.push_back(op("mm", 9000));
  TraceMetadata meta;
  meta.origin = DataSource::Synthetic;
  const std::vector<AnomalyRule> rules{p95_rule()};
  const auto text =
      format_check_report("t.jsonl", meta, events, rules, evaluate_rules(rules, events));
  CHECK(contains(text, "not a measured finding"));
  CHECK(contains(text, "synthetic demo data"));
  CHECK(contains(text, "Flagged events: 1"));
  CHECK(contains(text, "#0"));
}
