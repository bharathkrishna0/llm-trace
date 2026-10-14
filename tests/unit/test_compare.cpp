#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "llm_trace/compare.hpp"

using namespace llm_trace;

namespace {

TraceEvent op(const std::string& name, Category category, std::optional<std::uint64_t> dur,
              std::optional<std::int64_t> layer = {}, const std::string& op_type = {}) {
  TraceEvent e;
  e.name = name;
  e.category = category;
  e.duration_ns = dur;
  e.layer_index = layer;
  if (!op_type.empty()) e.op_type = op_type;
  e.device = "cpu";
  return e;
}

LoadedTrace trace(std::vector<TraceEvent> events, const std::string& model = "m") {
  LoadedTrace t;
  t.metadata.origin = DataSource::OnnxProfile;
  t.metadata.model = model;
  t.metadata.runtime = "rt";
  t.metadata.runtime_version = "1";
  t.events = std::move(events);
  return t;
}

const CompareRow* find(const std::vector<CompareRow>& rows, const std::string& key) {
  for (const auto& r : rows)
    if (r.key == key) return &r;
  return nullptr;
}

bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

TEST_CASE("identical traces show no change and no warnings", "[compare]") {
  const auto t = trace({op("qkv", Category::Attention, 100, 0), op("mlp", Category::Mlp, 300, 0),
                        op("qkv", Category::Attention, 120, 1)});
  const auto c = compare_traces(t, "a", t, "b");
  CHECK(c.warnings.empty());
  CHECK(c.only_in_a.empty());
  CHECK(c.only_in_b.empty());
  REQUIRE(c.by_op.size() == 2);
  for (const auto& r : c.by_op) CHECK(*r.mean_change() == Catch::Approx(0.0));
  REQUIRE(c.by_layer.size() == 2);
  CHECK(c.by_layer[0].key == "layer 0");
}

TEST_CASE("rows compare the mean per call not the total", "[compare]") {
  const auto a = trace({op("mlp", Category::Mlp, 100)});
  const auto b = trace({op("mlp", Category::Mlp, 100), op("mlp", Category::Mlp, 100)});
  const auto c = compare_traces(a, "a", b, "b");
  REQUIRE(c.by_op.size() == 1);
  CHECK(c.by_op[0].b->total_ns == 200);
  CHECK(*c.by_op[0].mean_change() == Catch::Approx(0.0));
}

TEST_CASE("largest change in mean comes first and keys present on one side are split out",
          "[compare]") {
  const auto a = trace({op("qkv", Category::Attention, 100), op("mlp", Category::Mlp, 1000),
                        op("old", Category::Other, 50)});
  const auto b = trace({op("qkv", Category::Attention, 200), op("mlp", Category::Mlp, 1010),
                        op("new", Category::Other, 70)});
  const auto c = compare_traces(a, "a", b, "b");
  REQUIRE(c.by_op.size() == 2);
  CHECK(c.by_op[0].key == "qkv");
  CHECK(*c.by_op[0].mean_change() == Catch::Approx(1.0));
  REQUIRE(c.only_in_a.size() == 1);
  CHECK(c.only_in_a[0].key == "old");
  REQUIRE(c.only_in_b.size() == 1);
  CHECK(c.only_in_b[0].key == "new");
  const auto* other = find(c.by_category, "other");
  REQUIRE(other);
  CHECK(*other->mean_change() == Catch::Approx(0.4));
}

TEST_CASE("op type key groups across node names", "[compare]") {
  const auto a = trace(
      {op("n1", Category::MatMul, 100, 0, "MatMul"), op("n2", Category::MatMul, 300, 1, "MatMul")});
  const auto b = trace({op("other", Category::MatMul, 400, 0, "MatMul")});
  CompareOptions o;
  o.key = CompareKey::OpType;
  const auto c = compare_traces(a, "a", b, "b", o);
  REQUIRE(c.by_op.size() == 1);
  CHECK(c.by_op[0].key == "MatMul");
  CHECK(c.by_op[0].a->mean_ns == Catch::Approx(200.0));
  CHECK(*c.by_op[0].mean_change() == Catch::Approx(1.0));
  CHECK(compare_traces(a, "a", b, "b").by_op.empty());  // names never match
}

TEST_CASE("median handles even counts and zero durations count", "[compare]") {
  const auto a = trace({op("x", Category::Other, 0), op("x", Category::Other, 10),
                        op("x", Category::Other, 20), op("x", Category::Other, 1000)});
  const auto c = compare_traces(a, "a", a, "b");
  REQUIRE(c.by_op.size() == 1);
  CHECK(c.by_op[0].a->count == 4);
  CHECK(c.by_op[0].a->median_ns == Catch::Approx(15.0));
}

TEST_CASE("operations without a duration are excluded and reported", "[compare]") {
  const auto a = trace({op("x", Category::Other, 10), op("y", Category::Other, std::nullopt)});
  const auto c = compare_traces(a, "a", a, "b");
  CHECK(c.a.operations == 1);
  CHECK(c.a.missing_duration == 1);
  CHECK(find(c.only_in_a, "y") == nullptr);
  CHECK_FALSE(c.warnings.empty());
}

TEST_CASE("comparison warns about mismatched sources and lost events", "[compare]") {
  auto a = trace({op("x", Category::Other, 10)}, "model-a");
  auto b = trace({op("x", Category::Other, 10)}, "model-b");
  b.metadata.origin = DataSource::Synthetic;
  b.metadata.overwritten_events = 7;
  b.events[0].device = "gpu";
  const auto c = compare_traces(a, "a", b, "b");
  std::string all;
  for (const auto& w : c.warnings) all += w + "\n";
  CHECK(contains(all, "Different sources"));
  CHECK(contains(all, "Different models"));
  CHECK(contains(all, "Different devices"));
  CHECK(contains(all, "synthetic demo data"));
  CHECK(contains(all, "overwrites"));
}

TEST_CASE("text and json outputs carry the tables", "[compare]") {
  const auto a = trace({op("qkv", Category::Attention, 100, 0)});
  const auto b = trace({op("qkv", Category::Attention, 150, 0), op("new", Category::Mlp, 5, 0)});
  const auto c = compare_traces(a, "a.jsonl", b, "b.jsonl");
  CompareTextOptions o;
  o.ascii = true;
  const auto text = format_comparison(c, o);
  CHECK(contains(text, "By category"));
  CHECK(contains(text, "By layer"));
  CHECK(contains(text, "+50.0%"));
  CHECK(contains(text, "Only in B: 1"));
  CHECK(contains(text, "not tested for statistical significance"));
  for (unsigned char ch : text) CHECK(ch < 0x80);
  const auto json = comparison_to_json(c);
  CHECK(contains(json, "\"schema\": \"llm-trace-compare/1\""));
  CHECK(contains(json, "\"mean_change\": 0.5"));
}
