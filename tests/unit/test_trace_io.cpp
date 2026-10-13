#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "llm_trace/synthetic.hpp"
#include "llm_trace/trace_io.hpp"

using namespace llm_trace;
using Catch::Matchers::ContainsSubstring;

namespace {

const std::string kFixtures = std::string(LLM_TRACE_SOURCE_DIR) + "/tests/fixtures/jsonl/";

std::string meta_line() {
  TraceMetadata m;
  m.run_id = "t";
  m.source = DataSource::Synthetic;
  return serialize_metadata(m);
}

LoadedTrace parse(const std::string& text, const ReadLimits& limits = {}) {
  std::istringstream in(text);
  return read_jsonl(in, limits);
}

std::size_t error_line(const std::string& text, const ReadLimits& limits = {}) {
  try {
    parse(text, limits);
  } catch (const TraceIoError& e) {
    return e.line();
  }
  FAIL("expected TraceIoError");
  return 0;
}

TraceEvent full_event() {
  TraceEvent e;
  e.sequence = 7;
  e.timestamp_ns = 100;
  e.duration_ns = 0;  // zero is a real value and must survive
  e.name = "decoder.layer.00.attention.softmax";
  e.category = Category::Attention;
  e.op_type = "Softmax";
  e.parent_id = "decoder.layer.00";
  e.layer_index = 0;
  e.token_index = 3;
  e.head_index = -1;
  e.device = "cpu";
  e.thread_id = 12;
  e.inputs.push_back({"scores", std::vector<std::int64_t>{1, 12, -1, 64}, "float32", 768, 3072});
  e.outputs.push_back({"probs", std::nullopt, std::nullopt, std::nullopt, std::nullopt});
  e.activation = ActivationSummary{0.5, -1.0, 2.0, 0.25, 0, 0.1, 1e-6, 1024, false};
  e.attributes = {{"raw_name", "/model/Softmax_3"}, {"raw_name", "dup kept"}};
  return e;
}

}  // namespace

TEST_CASE("event round-trips with optional fields preserved") {
  TraceMetadata m = synthetic_metadata({});
  const TraceEvent e = full_event();
  std::ostringstream out;
  write_jsonl(out, m, {e});
  const LoadedTrace t = parse(out.str());
  REQUIRE(t.events.size() == 1);
  const TraceEvent& r = t.events[0];
  CHECK(r.sequence == 7);
  CHECK(r.duration_ns == std::optional<std::uint64_t>(0));
  CHECK(r.name == e.name);
  CHECK(r.category == Category::Attention);
  CHECK(r.op_type == e.op_type);
  CHECK(r.parent_id == e.parent_id);
  CHECK(r.layer_index == 0);
  CHECK(r.token_index == 3);
  CHECK(r.head_index == -1);
  CHECK(r.device == e.device);
  CHECK(r.thread_id == 12);
  REQUIRE(r.inputs.size() == 1);
  CHECK(*r.inputs[0].shape == std::vector<std::int64_t>{1, 12, -1, 64});
  CHECK(r.inputs[0].byte_count == std::optional<std::uint64_t>(3072));
  REQUIRE(r.outputs.size() == 1);
  CHECK_FALSE(r.outputs[0].shape.has_value());
  CHECK_FALSE(r.outputs[0].dtype.has_value());
  REQUIRE(r.activation.has_value());
  CHECK(r.activation->sampled_elements == 1024);
  CHECK(r.attributes == e.attributes);
  CHECK(t.metadata.source == DataSource::Synthetic);
  CHECK(t.metadata.notes == m.notes);
  CHECK(t.metadata.capabilities.attention_values == false);
  CHECK(t.metadata.capabilities.operation_timing == true);
}

TEST_CASE("missing optional fields stay absent") {
  const LoadedTrace t = parse(
      meta_line() +
      "\n{\"record\":\"event\",\"sequence\":1,\"timestamp_ns\":5,\"type\":\"operation\",\"name\":"
      "\"x\"}\n");
  REQUIRE(t.events.size() == 1);
  CHECK_FALSE(t.events[0].duration_ns.has_value());
  CHECK_FALSE(t.events[0].layer_index.has_value());
  CHECK(t.events[0].category == Category::Unknown);
}

TEST_CASE("whole synthetic trace round-trips through a file") {
  SyntheticConfig cfg;
  cfg.layers = 2;
  cfg.decode_tokens = 3;
  const auto events = generate_synthetic(cfg);
  const auto path = (std::filesystem::temp_directory_path() / "llm_trace_roundtrip.jsonl").string();
  write_jsonl(path, synthetic_metadata(cfg), events);
  CHECK_FALSE(std::filesystem::exists(path + ".tmp"));
  const LoadedTrace t = load_trace(path);
  REQUIRE(t.events.size() == events.size());
  for (std::size_t i = 0; i < events.size(); ++i) {
    CHECK(serialize_event(t.events[i]) == serialize_event(events[i]));
  }
  CHECK(t.metadata.source == DataSource::Replay);
  CHECK(t.metadata.origin == DataSource::Synthetic);
  std::filesystem::remove(path);
}

TEST_CASE("export replaces an existing file only after a complete write") {
  const auto path = (std::filesystem::temp_directory_path() / "llm_trace_replace.jsonl").string();
  { std::ofstream(path) << "old contents\n"; }
  TraceMetadata m;
  TraceEvent bad = full_event();
  bad.activation->mean = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(write_jsonl(path, m, {bad}), TraceIoError);
  {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    CHECK(line == "old contents");
  }
  CHECK_FALSE(std::filesystem::exists(path + ".tmp"));
  write_jsonl(path, m, {full_event()});
  CHECK(load_trace(path).events.size() == 1);
  std::filesystem::remove(path);
}

TEST_CASE("malformed input reports the offending line") {
  const std::string m = meta_line() + "\n";
  const std::string ok =
      "{\"record\":\"event\",\"sequence\":1,\"timestamp_ns\":5,\"type\":\"operation\",\"name\":"
      "\"x\"}\n";
  CHECK(error_line(m + ok + "{not json\n") == 3);
  CHECK(error_line(m + ok + "[1,2]\n") == 3);
  CHECK(error_line(m + "{\"record\":\"event\",\"timestamp_ns\":5,\"type\":\"operation\",\"name\":"
                       "\"x\"}\n") == 2);  // missing sequence
  CHECK(error_line(m + "{\"record\":\"event\",\"sequence\":-1,\"timestamp_ns\":5,\"type\":"
                       "\"operation\",\"name\":\"x\"}\n") == 2);
  CHECK(error_line(m + "{\"record\":\"event\",\"sequence\":1,\"timestamp_ns\":\"5\",\"type\":"
                       "\"operation\",\"name\":\"x\"}\n") == 2);
  CHECK(error_line(m + "{\"record\":\"event\",\"sequence\":1,\"timestamp_ns\":5,\"type\":"
                       "\"teleport\",\"name\":\"x\"}\n") == 2);
  CHECK(error_line(m + ok + ok) == 3);  // non-increasing sequence
  CHECK(error_line(ok) == 1);           // no metadata first
  CHECK(error_line(m + m) == 2);        // duplicate metadata
  CHECK(error_line(m + "{\"record\":\"event\",\"sequence\":1,\"timestamp_ns\":5,\"type\":"
                       "\"operation\",\"name\":\"x\",\"inputs\":[{\"name\":\"t\",\"shape\":[-2]}]"
                       "}\n") == 2);
  CHECK_THROWS_WITH(parse(""), ContainsSubstring("no metadata"));
}

TEST_CASE("newer schema versions are rejected with a clear message") {
  CHECK_THROWS_WITH(parse("{\"record\":\"metadata\",\"schema_version\":99}\n"),
                    ContainsSubstring("schema_version 99"));
}

TEST_CASE("limits are enforced before allocation") {
  const std::string m = meta_line() + "\n";
  const std::string ev1 =
      "{\"record\":\"event\",\"sequence\":1,\"timestamp_ns\":5,\"type\":\"operation\",\"name\":"
      "\"x\"}\n";
  const std::string ev2 =
      "{\"record\":\"event\",\"sequence\":2,\"timestamp_ns\":6,\"type\":\"operation\",\"name\":"
      "\"y\"}\n";

  ReadLimits lines;
  lines.max_line_bytes = m.size() + 8;
  CHECK(error_line(m + ev1 + "{\"record\":\"event\",\"name\":\"" + std::string(500, 'a') + "\"}\n",
                   lines) == 3);

  ReadLimits events;
  events.max_events = 1;
  CHECK(error_line(m + ev1 + ev2, events) == 3);

  ReadLimits strings;
  strings.max_string_bytes = 16;
  const std::string long_name =
      "{\"record\":\"event\",\"sequence\":1,\"timestamp_ns\":5,\"type\":"
      "\"operation\",\"name\":\"" +
      std::string(17, 'n') + "\"}\n";
  CHECK(error_line(m + long_name, strings) == 2);

  ReadLimits tensors;
  tensors.max_tensors_per_event = 1;
  CHECK(error_line(
            m + "{\"record\":\"event\",\"sequence\":1,\"timestamp_ns\":5,\"type\":"
                "\"operation\",\"name\":\"x\",\"inputs\":[{\"name\":\"a\"},{\"name\":\"b\"}]}\n",
            tensors) == 2);

  const std::string deep =
      m + "{\"record\":\"event\",\"x\":" + std::string(100, '[') + std::string(100, ']') + "}\n";
  CHECK(error_line(deep) == 2);
}

TEST_CASE("unknown categories and records are kept as warnings") {
  const LoadedTrace t = parse(meta_line() +
                              "\n{\"record\":\"future\"}\n"
                              "{\"record\":\"event\",\"sequence\":1,\"timestamp_ns\":5,\"type\":"
                              "\"operation\",\"name\":\"x\",\"category\":\"quantum\"}\n");
  CHECK(t.events.size() == 1);
  CHECK(t.events[0].category == Category::Unknown);
  REQUIRE(t.warnings.size() == 2);
  CHECK_THAT(t.warnings[0], ContainsSubstring("line 2"));
  CHECK_THAT(t.warnings[1], ContainsSubstring("quantum"));
}

TEST_CASE("checked-in fixtures load") {
  const LoadedTrace t = load_trace(kFixtures + "synthetic_small.jsonl");
  CHECK(t.metadata.origin == DataSource::Synthetic);
  CHECK_FALSE(t.events.empty());
  CHECK_THROWS_WITH(load_trace(kFixtures + "malformed_line3.jsonl"), ContainsSubstring("line 3"));
  CHECK_THROWS_WITH(load_trace(kFixtures + "does_not_exist.jsonl"),
                    ContainsSubstring("cannot open"));
}
