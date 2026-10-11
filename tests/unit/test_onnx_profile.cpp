#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <sstream>

#include "llm_trace/onnx_profile_adapter.hpp"
#include "llm_trace/trace_io.hpp"

using namespace llm_trace;
using Catch::Matchers::ContainsSubstring;

namespace {

const std::string kProfile = std::string(LLM_TRACE_SOURCE_DIR) +
                             "/tests/fixtures/onnx_profiles/tiny_transformer_ort-1.20.1.json";

LoadedTrace parse(const std::string& text, const ReadLimits& limits = {}) {
  std::istringstream in(text);
  return OnnxProfileAdapter().parse(in, "test.json", limits);
}

const std::string kKernel =
    R"({"cat":"Node","ph":"X","ts":10,"dur":4,"tid":1,"name":"/layers.1/mlp/MatMul_kernel_time",)"
    R"("args":{"op_name":"MatMul","provider":"CPUExecutionProvider",)"
    R"("input_type_shape":[{"float":[1,16,32]},{"float":[32,-1]}],"output_type_shape":[{"float":[1,16,128]}]}})";

}  // namespace

TEST_CASE("real ONNX Runtime 1.20.1 profile loads") {
  const LoadedTrace t = load_trace(kProfile);
  CHECK(t.metadata.source == DataSource::OnnxProfile);
  CHECK(t.metadata.runtime == "ONNX Runtime");
  CHECK(t.metadata.capabilities.operation_timing);
  CHECK_FALSE(t.metadata.capabilities.attention_values);
  CHECK_FALSE(t.metadata.capabilities.activation_summaries);
  CHECK(t.warnings.empty());

  std::size_t kernels = 0, sessions = 0, runs = 0;
  for (std::size_t i = 0; i < t.events.size(); ++i) {
    const TraceEvent& e = t.events[i];
    CHECK(e.sequence == i);
    if (i > 0) CHECK(e.timestamp_ns >= t.events[i - 1].timestamp_ns);
    CHECK(e.name.find("fence") == std::string::npos);
    if (e.category == Category::Session) {
      ++sessions;
      if (e.name == "model_run") ++runs;
      continue;
    }
    ++kernels;
    REQUIRE(e.op_type.has_value());
    CHECK(e.device == std::optional<std::string>("CPUExecutionProvider"));
    CHECK(e.attributes.front().first == "raw_name");
    CHECK(e.attributes.front().second == e.name + "_kernel_time");
  }
  // 44 kernels per run x 2 profiled runs (fixture written by onnx_profile_runner --runs 2).
  CHECK(kernels == 88);
  CHECK(runs == 2);
  CHECK(sessions == 6);

  const auto ln = std::find_if(t.events.begin(), t.events.end(), [](const TraceEvent& e) {
    return e.name == "/layers.0/input_layernorm/LayerNormalization";
  });
  REQUIRE(ln != t.events.end());
  CHECK(ln->category == Category::Normalization);
  CHECK(ln->layer_index == 0);
  REQUIRE(ln->inputs.size() == 3);
  CHECK(*ln->inputs[0].shape == std::vector<std::int64_t>{1, 16, 32});
  CHECK(ln->inputs[0].dtype == std::optional<std::string>("float"));
  CHECK(ln->inputs[0].byte_count == std::optional<std::uint64_t>(2048));
}

TEST_CASE("kernel record fields are mapped without inventing values") {
  const LoadedTrace t = parse("[" + kKernel + "]");
  REQUIRE(t.events.size() == 1);
  const TraceEvent& e = t.events[0];
  CHECK(e.name == "/layers.1/mlp/MatMul");
  CHECK(e.timestamp_ns == 10'000);
  CHECK(e.duration_ns == std::optional<std::uint64_t>(4'000));
  CHECK(e.category == Category::MatMul);
  CHECK(e.layer_index == 1);
  CHECK(e.thread_id == 1);
  CHECK_FALSE(e.token_index.has_value());
  CHECK_FALSE(e.activation.has_value());
  REQUIRE(e.inputs.size() == 2);
  CHECK_FALSE(e.inputs[1].element_count.has_value());  // dynamic dimension
  CHECK(e.outputs[0].byte_count == std::optional<std::uint64_t>(1 * 16 * 128 * 4));
}

TEST_CASE("malformed profiles are rejected") {
  CHECK_THROWS_WITH(parse("{}"), ContainsSubstring("JSON array"));
  CHECK_THROWS_WITH(parse("[{\"cat\":"), ContainsSubstring("not valid JSON"));
  CHECK_THROWS_WITH(parse("[1]"), ContainsSubstring("entry 0"));
  CHECK_THROWS_WITH(parse(R"([{"cat":"Node","ph":"X","ts":-5,"dur":1,"name":"a_kernel_time"}])"),
                    ContainsSubstring("\"ts\""));
  CHECK_THROWS_WITH(parse(R"([{"cat":"Node","ph":"X","ts":5,"name":"a_kernel_time"}])"),
                    ContainsSubstring("missing \"dur\""));
  CHECK_THROWS_WITH(
      parse(
          R"([{"cat":"Node","ph":"X","ts":5,"dur":1,"name":"a_kernel_time","args":{"input_type_shape":[{"float":[-3]}]}}])"),
      ContainsSubstring(">= -1"));
  CHECK_THROWS_WITH(parse("[" + std::string(50, '[') + std::string(50, ']') + "]"),
                    ContainsSubstring("too deep"));
  ReadLimits small;
  small.max_profile_bytes = 16;
  CHECK_THROWS_WITH(parse("[" + kKernel + "]", small), ContainsSubstring("limit"));
  ReadLimits one;
  one.max_events = 1;
  CHECK_THROWS_WITH(parse("[" + kKernel + "," + kKernel + "]", one),
                    ContainsSubstring("more than 1"));
}

TEST_CASE("unrecognised profile records are skipped with a warning") {
  const LoadedTrace t =
      parse(R"([{"cat":"Api","ph":"X","ts":1,"dur":1,"name":"x"},{"ph":"i"},)" + kKernel + "]");
  CHECK(t.events.size() == 1);
  REQUIRE(t.warnings.size() == 1);
  CHECK_THAT(t.warnings[0], ContainsSubstring("2 profile record(s)"));
}

TEST_CASE("onnx op categories and layer names use fixed rules") {
  CHECK(category_for_onnx_op("FusedMatMul") == Category::MatMul);
  CHECK(category_for_onnx_op("Softmax") == Category::Activation);
  CHECK(category_for_onnx_op("GroupQueryAttention") == Category::Attention);
  CHECK(category_for_onnx_op("SomethingNew") == Category::Other);
  CHECK(layer_from_node_name("/layers.12/attn/MatMul") == 12);
  CHECK(layer_from_node_name("/model/layers_3/mlp") == 3);
  CHECK(layer_from_node_name("transformer.h.7.mlp") == 7);
  CHECK_FALSE(layer_from_node_name("/players.3/x").has_value());
  CHECK_FALSE(layer_from_node_name("/layers.x/attn").has_value());
  CHECK_FALSE(layer_from_node_name("/lm_head/MatMul").has_value());
}

TEST_CASE("onnx profile survives a JSONL round trip as a replay") {
  const LoadedTrace t = load_trace(kProfile);
  const auto path = (std::filesystem::temp_directory_path() / "llm_trace_ort.jsonl").string();
  write_jsonl(path, t.metadata, t.events);
  const LoadedTrace r = load_trace(path);
  CHECK(r.metadata.source == DataSource::Replay);
  CHECK(r.metadata.origin == DataSource::OnnxProfile);
  REQUIRE(r.events.size() == t.events.size());
  CHECK(serialize_event(r.events[5]) == serialize_event(t.events[5]));
  std::filesystem::remove(path);
}
