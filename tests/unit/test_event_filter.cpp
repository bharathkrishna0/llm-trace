#include <catch2/catch_test_macros.hpp>

#include "llm_trace/event_filter.hpp"

using namespace llm_trace;

namespace {

TraceEvent make_op(const std::string& name, Category category, std::optional<std::uint64_t> dur,
                   std::optional<std::int64_t> layer = {}) {
  TraceEvent e;
  e.name = name;
  e.category = category;
  e.duration_ns = dur;
  e.layer_index = layer;
  return e;
}

bool match(const std::string& filter, const TraceEvent& e) {
  const auto f = EventFilter::parse(filter);
  REQUIRE(f.valid());
  return f.matches(e);
}

}  // namespace

TEST_CASE("empty filter is valid and empty", "[filter]") {
  const auto f = EventFilter::parse("   ");
  CHECK(f.valid());
  CHECK(f.empty());
}

TEST_CASE("free text matches name category and op type case-insensitively", "[filter]") {
  auto e = make_op("decoder.layer.0.qkv", Category::Attention, 100, 0);
  e.op_type = "MatMul";
  CHECK(match("QKV", e));
  CHECK(match("attention", e));
  CHECK(match("matmul", e));
  CHECK_FALSE(match("softmax", e));
}

TEST_CASE("bare integer matches the layer index exactly", "[filter]") {
  CHECK(match("3", make_op("mlp", Category::Mlp, 1, 3)));
  CHECK_FALSE(match("3", make_op("mlp", Category::Mlp, 1, 13)));
  CHECK(match("1", make_op("layer.1.mlp", Category::Mlp, 1, 4)));  // still a substring
}

TEST_CASE("field terms target one field", "[filter]") {
  auto e = make_op("decoder.layer.2.qkv", Category::Attention, 100, 2);
  e.op_type = "MatMul";
  e.device = "CPU";
  CHECK(match("name:qkv", e));
  CHECK_FALSE(match("name:matmul", e));
  CHECK(match("op:matmul", e));
  CHECK(match("device:cpu", e));
  CHECK(match("dev:CP", e));
  CHECK(match("cat:attention", e));
  CHECK_FALSE(EventFilter::parse("cat:atten").valid());  // category names are exact
  CHECK(match("type:operation", e));
}

TEST_CASE("missing string fields never match a contains term", "[filter]") {
  const auto e = make_op("x", Category::Other, 1);
  CHECK_FALSE(match("op:matmul", e));
  CHECK(match("op:missing", e));
  CHECK(match("-op:matmul", e));
  CHECK(match("device:missing", e));
}

TEST_CASE("integer comparisons and ranges", "[filter]") {
  const auto e = make_op("mlp", Category::Mlp, 1, 4);
  CHECK(match("layer:4", e));
  CHECK(match("layer=4", e));
  CHECK(match("layer>=4", e));
  CHECK(match("layer>3", e));
  CHECK_FALSE(match("layer<4", e));
  CHECK(match("layer<=4", e));
  CHECK(match("layer:2..5", e));
  CHECK_FALSE(match("layer:5..9", e));
  CHECK_FALSE(match("token:0", e));  // field absent
  CHECK(match("token:missing", e));
}

TEST_CASE("durations accept units", "[filter]") {
  const auto e = make_op("mlp", Category::Mlp, 60000);  // 60 us
  CHECK(match("dur>50us", e));
  CHECK(match("dur>50µs", e));
  CHECK(match("dur<=0.06ms", e));
  CHECK(match("duration>=60000ns", e));
  CHECK_FALSE(match("dur>1s", e));
  CHECK(match("dur:50us..70us", e));
  CHECK_FALSE(match("dur:missing", e));
  CHECK(match("dur:missing", make_op("lm_head", Category::Output, std::nullopt)));
  CHECK_FALSE(match("dur>0ns", make_op("lm_head", Category::Output, std::nullopt)));
}

TEST_CASE("zero duration is a value not missing", "[filter]") {
  const auto e = make_op("noop", Category::Other, 0);
  CHECK(match("dur:0ns", e));
  CHECK_FALSE(match("dur:missing", e));
}

TEST_CASE("terms are ANDed and can be negated", "[filter]") {
  const auto attn = make_op("layer.0.attn", Category::Attention, 500, 0);
  const auto mlp = make_op("layer.0.mlp", Category::Mlp, 500, 0);
  CHECK(match("cat:attention layer:0", attn));
  CHECK_FALSE(match("cat:attention layer:0", mlp));
  CHECK(match("-cat:attention", mlp));
  CHECK_FALSE(match("-cat:attention", attn));
  CHECK(match("-attn", mlp));
}

TEST_CASE("quoted text stays one term", "[filter]") {
  const auto e = make_op("model run", Category::Session, 1);
  CHECK(match("\"model run\"", e));
  CHECK(match("name:\"model run\"", e));
  CHECK_FALSE(match("\"run model\"", e));
}

TEST_CASE("syntax errors are reported and match nothing", "[filter]") {
  for (const char* bad : {"layer:abc", "dur>50", "dur>50parsecs", "layer:5..2", "cat:bogus",
                          "type:nope", "layer:>3", "\"unterminated", "name:"}) {
    INFO(bad);
    const auto f = EventFilter::parse(bad);
    CHECK_FALSE(f.valid());
    CHECK_FALSE(f.error().empty());
    CHECK_FALSE(f.matches(make_op("x", Category::Other, 1, 1)));
  }
}

TEST_CASE("term count is bounded", "[filter]") {
  std::string many;
  for (int i = 0; i < 17; ++i) many += "a ";
  CHECK_FALSE(EventFilter::parse(many).valid());
  std::string ok;
  for (int i = 0; i < 16; ++i) ok += "a ";
  CHECK(EventFilter::parse(ok).valid());
}

TEST_CASE("unknown field keys are reported so typos do not silently match", "[filter]") {
  const auto f = EventFilter::parse("lyer:3");
  CHECK_FALSE(f.valid());
  CHECK(f.error().find("unknown filter key") != std::string::npos);
  CHECK(match("a<b", make_op("a<b", Category::Other, 1)));  // no key: plain text
}
