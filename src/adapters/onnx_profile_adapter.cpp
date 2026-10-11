#include "llm_trace/onnx_profile_adapter.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <unordered_map>

#include "core/json_depth.hpp"

namespace llm_trace {

namespace {

using nlohmann::json;

constexpr int kMaxJsonDepth = 16;
constexpr const char* kKernelSuffix = "_kernel_time";
constexpr std::uint64_t kNsPerUs = 1000;

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

struct Parser {
  const ReadLimits& limits;
  std::size_t entry = 0;

  [[noreturn]] void fail(const std::string& message) const {
    throw TraceIoError("profile entry " + std::to_string(entry) + ": " + message);
  }

  const json* find(const json& obj, const char* key) const {
    const auto it = obj.find(key);
    return it == obj.end() || it->is_null() ? nullptr : &*it;
  }

  std::string str(const json& v, const char* key) const {
    if (!v.is_string()) fail(std::string("\"") + key + "\" must be a string");
    const auto& s = v.get_ref<const std::string&>();
    if (s.size() > limits.max_string_bytes)
      fail(std::string("\"") + key + "\" exceeds " + std::to_string(limits.max_string_bytes) +
           " bytes");
    return s;
  }

  std::uint64_t micros_to_ns(const json& obj, const char* key) const {
    const json* v = find(obj, key);
    if (!v) fail(std::string("missing \"") + key + "\"");
    if (!v->is_number_unsigned())
      fail(std::string("\"") + key + "\" must be a non-negative integer");
    const auto us = v->get<std::uint64_t>();
    if (us > std::numeric_limits<std::uint64_t>::max() / kNsPerUs)
      fail(std::string("\"") + key + "\" is too large");
    return us * kNsPerUs;
  }

  // ORT writes shapes as [{"float": [1, 16, 32]}, ...]; tensors have no names.
  std::vector<TensorMetadata> tensors(const json& args, const char* key, const char* label) const {
    std::vector<TensorMetadata> out;
    const json* arr = find(args, key);
    if (!arr) return out;
    if (!arr->is_array()) fail(std::string("\"") + key + "\" must be an array");
    if (arr->size() > limits.max_tensors_per_event)
      fail(std::string("\"") + key + "\" has more than " +
           std::to_string(limits.max_tensors_per_event) + " tensors");
    for (const auto& item : *arr) {
      if (!item.is_object() || item.size() != 1)
        fail(std::string("\"") + key + "\" entries must be {dtype: shape}");
      TensorMetadata t;
      t.name = std::string(label) + "[" + std::to_string(out.size()) + "]";
      const auto& [dtype, shape] = *item.items().begin();
      if (dtype.size() > limits.max_string_bytes) fail("dtype name is too long");
      t.dtype = dtype;
      if (!shape.is_array()) fail("tensor shape must be an array");
      if (shape.size() > limits.max_shape_rank)
        fail("tensor rank exceeds " + std::to_string(limits.max_shape_rank));
      std::vector<std::int64_t> dims;
      for (const auto& d : shape) {
        if (!d.is_number_integer()) fail("tensor dimensions must be integers");
        const auto dim = d.get<std::int64_t>();
        if (dim < -1) fail("tensor dimensions must be >= -1");
        dims.push_back(dim);
      }
      const bool dynamic = std::find(dims.begin(), dims.end(), -1) != dims.end();
      if (!dynamic) {
        t.element_count = element_count_of(dims);
        if (const auto size = dtype_size_bytes(dtype);
            size && *t.element_count > 0 && *t.element_count <= UINT64_MAX / *size)
          t.byte_count = *t.element_count * *size;
      }
      t.shape = std::move(dims);
      out.push_back(std::move(t));
    }
    return out;
  }
};

std::string read_bounded(std::istream& in, std::uint64_t max_bytes) {
  std::string text;
  char buf[64 * 1024];
  while (in.read(buf, sizeof buf) || in.gcount() > 0) {
    const auto n = static_cast<std::size_t>(in.gcount());
    if (text.size() + n > max_bytes)
      throw TraceIoError("profile is larger than the " + std::to_string(max_bytes) + "-byte limit");
    text.append(buf, n);
  }
  if (in.bad()) throw TraceIoError("read error");
  return text;
}

}  // namespace

Category category_for_onnx_op(const std::string& op) {
  static const std::unordered_map<std::string, Category> table = {
      {"MatMul", Category::MatMul},
      {"FusedMatMul", Category::MatMul},
      {"Gemm", Category::MatMul},
      {"MatMulInteger", Category::MatMul},
      {"MatMulNBits", Category::MatMul},
      {"Attention", Category::Attention},
      {"MultiHeadAttention", Category::Attention},
      {"GroupQueryAttention", Category::Attention},
      {"LayerNormalization", Category::Normalization},
      {"SimplifiedLayerNormalization", Category::Normalization},
      {"SkipLayerNormalization", Category::Normalization},
      {"SkipSimplifiedLayerNormalization", Category::Normalization},
      {"RMSNormalization", Category::Normalization},
      {"EmbedLayerNormalization", Category::Embedding},
      {"Relu", Category::Activation},
      {"Gelu", Category::Activation},
      {"FastGelu", Category::Activation},
      {"BiasGelu", Category::Activation},
      {"Sigmoid", Category::Activation},
      {"Tanh", Category::Activation},
      {"Erf", Category::Activation},
      {"Softmax", Category::Activation},
      {"Add", Category::Elementwise},
      {"Sub", Category::Elementwise},
      {"Mul", Category::Elementwise},
      {"Div", Category::Elementwise},
      {"Pow", Category::Elementwise},
      {"Sqrt", Category::Elementwise},
      {"Where", Category::Elementwise},
      {"Cast", Category::Elementwise},
      {"Reshape", Category::Shape},
      {"Transpose", Category::Shape},
      {"Concat", Category::Shape},
      {"Slice", Category::Shape},
      {"Split", Category::Shape},
      {"Squeeze", Category::Shape},
      {"Unsqueeze", Category::Shape},
      {"Shape", Category::Shape},
      {"Expand", Category::Shape},
      {"Identity", Category::Shape},
  };
  const auto it = table.find(op);
  return it == table.end() ? Category::Other : it->second;
}

std::optional<std::int64_t> layer_from_node_name(const std::string& name) {
  // Accepts "<sep>layers.N<sep>", "<sep>layer.N<sep>", "<sep>h.N<sep>" with sep in "/._".
  static const char* const kPrefixes[] = {"layers", "layer", "h"};
  auto is_sep = [](char c) { return c == '/' || c == '.' || c == '_'; };
  for (const char* prefix : kPrefixes) {
    const std::string p(prefix);
    for (std::size_t pos = name.find(p); pos != std::string::npos; pos = name.find(p, pos + 1)) {
      if (pos > 0 && !is_sep(name[pos - 1])) continue;
      std::size_t i = pos + p.size();
      if (i >= name.size() || (name[i] != '.' && name[i] != '_')) continue;
      ++i;
      const std::size_t digits = i;
      while (i < name.size() && std::isdigit(static_cast<unsigned char>(name[i])) && i - digits < 6)
        ++i;
      if (i == digits || (i < name.size() && !is_sep(name[i]))) continue;
      return std::stoll(name.substr(digits, i - digits));
    }
  }
  return std::nullopt;
}

AdapterCapabilities OnnxProfileAdapter::capabilities() const {
  AdapterCapabilities c;
  c.operation_timing = true;
  c.tensor_metadata = true;  // shapes and dtypes only; ORT profiles carry no values
  return c;
}

LoadedTrace OnnxProfileAdapter::load(const std::string& path, const ReadLimits& limits) const {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec) throw TraceIoError("cannot open " + path + ": " + ec.message());
  if (size > limits.max_profile_bytes)
    throw TraceIoError(path + " is larger than the " + std::to_string(limits.max_profile_bytes) +
                       "-byte profile limit");
  std::ifstream in(path, std::ios::binary);
  if (!in) throw TraceIoError("cannot open " + path);
  return parse(in, std::filesystem::path(path).filename().string(), limits);
}

LoadedTrace OnnxProfileAdapter::parse(std::istream& in, const std::string& source_name,
                                      const ReadLimits& limits) const {
  const std::string text = read_bounded(in, limits.max_profile_bytes);
  if (!detail::json_depth_within(text, kMaxJsonDepth))
    throw TraceIoError("profile JSON nesting is too deep");
  const json doc = json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded()) throw TraceIoError("profile is not valid JSON");
  if (!doc.is_array()) throw TraceIoError("profile must be a JSON array of trace events");

  LoadedTrace trace;
  Parser p{limits};
  std::size_t fences = 0, skipped = 0, layered = 0;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;  // model_run [start, end)
  for (const auto& item : doc) {
    if (!item.is_object()) p.fail("must be an object");
    const json* ph = p.find(item, "ph");
    if (!ph || p.str(*ph, "ph") != "X") {
      ++skipped;  // only complete events ("X") carry durations
      ++p.entry;
      continue;
    }
    const json* cat_v = p.find(item, "cat");
    const json* name_v = p.find(item, "name");
    if (!cat_v || !name_v) p.fail("missing \"cat\" or \"name\"");
    const std::string cat = p.str(*cat_v, "cat");
    const std::string raw = p.str(*name_v, "name");
    if (raw.empty()) p.fail("empty \"name\"");
    if (cat == "Node" && (ends_with(raw, "_fence_before") || ends_with(raw, "_fence_after"))) {
      ++fences;
      ++p.entry;
      continue;
    }
    if (cat != "Node" && cat != "Session") {
      ++skipped;
      ++p.entry;
      continue;
    }
    if (trace.events.size() >= limits.max_events)
      p.fail("profile has more than " + std::to_string(limits.max_events) + " events");

    TraceEvent e;
    e.timestamp_ns = p.micros_to_ns(item, "ts");
    e.duration_ns = p.micros_to_ns(item, "dur");
    if (const json* tid = p.find(item, "tid"); tid && tid->is_number_integer())
      e.thread_id = tid->get<std::int64_t>();
    e.attributes.emplace_back("raw_name", raw);
    const json* args = p.find(item, "args");
    if (args && !args->is_object()) p.fail("\"args\" must be an object");

    if (cat == "Session") {
      e.name = raw;
      e.category = Category::Session;
      if (raw == "model_run") runs.emplace_back(e.timestamp_ns, e.timestamp_ns + *e.duration_ns);
    } else if (ends_with(raw, kKernelSuffix)) {
      e.name = raw.substr(0, raw.size() - std::string(kKernelSuffix).size());
      if (args) {
        if (const json* op = p.find(*args, "op_name")) e.op_type = p.str(*op, "op_name");
        if (const json* prov = p.find(*args, "provider")) e.device = p.str(*prov, "provider");
        e.inputs = p.tensors(*args, "input_type_shape", "input");
        e.outputs = p.tensors(*args, "output_type_shape", "output");
        for (const char* key : {"node_index", "parameter_size", "activation_size", "output_size"})
          if (const json* v = p.find(*args, key); v && v->is_string())
            e.attributes.emplace_back(key, p.str(*v, key));
      }
      e.category = e.op_type ? category_for_onnx_op(*e.op_type) : Category::Unknown;
      e.layer_index = layer_from_node_name(e.name);
      if (e.layer_index) ++layered;
    } else {
      ++skipped;  // other Node records (none in ORT 1.20.1) are not interpreted
      ++p.entry;
      continue;
    }
    if (const std::string reason = validate(e); !reason.empty()) p.fail(reason);
    trace.events.push_back(std::move(e));
    ++p.entry;
  }

  std::stable_sort(
      trace.events.begin(), trace.events.end(),
      [](const TraceEvent& a, const TraceEvent& b) { return a.timestamp_ns < b.timestamp_ns; });
  std::sort(runs.begin(), runs.end());
  for (std::size_t i = 0; i < trace.events.size(); ++i) {
    TraceEvent& e = trace.events[i];
    e.sequence = i;
    if (e.category == Category::Session) continue;
    for (std::size_t r = 0; r < runs.size(); ++r) {
      if (e.timestamp_ns >= runs[r].first && e.timestamp_ns < runs[r].second) {
        e.attributes.emplace_back("run_index", std::to_string(r));
        break;
      }
    }
  }

  TraceMetadata& m = trace.metadata;
  m.run_id = source_name;
  m.source = DataSource::OnnxProfile;
  m.origin = DataSource::OnnxProfile;
  m.adapter = name();
  m.runtime = "ONNX Runtime";
  m.capabilities = capabilities();
  m.notes.push_back(
      "Durations are ONNX Runtime kernel_time records (microsecond resolution). Session "
      "records (model_run, session_initialization) are listed but excluded from operation "
      "totals.");
  m.notes.push_back(
      "The profile does not record the ONNX Runtime version; this adapter is tested with "
      "1.20.1.");
  m.notes.push_back(
      "ONNX Runtime profiles contain no tensor values: attention weights and activation "
      "statistics are not available.");
  if (layered > 0)
    m.notes.push_back(
        "Layer numbers are parsed from node names (e.g. \"layers.3\"), not "
        "reported by ONNX Runtime.");
  if (fences > 0)
    m.notes.push_back(std::to_string(fences) + " fence_before/fence_after records were omitted.");
  if (skipped > 0)
    trace.warnings.push_back(std::to_string(skipped) +
                             " profile record(s) were not recognised "
                             "and were skipped.");
  if (runs.size() > 0)
    m.notes.push_back(std::to_string(runs.size()) + " model_run(s) in this profile.");
  return trace;
}

}  // namespace llm_trace
