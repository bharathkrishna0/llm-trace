#include "llm_trace/trace_io.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <istream>
#include <limits>
#include <nlohmann/json.hpp>
#include <ostream>
#include <system_error>

namespace llm_trace {

namespace {

using nlohmann::json;
// Writing keeps insertion order so "record" is the first key on every line.
using ojson = nlohmann::ordered_json;

constexpr int kMaxJsonDepth = 16;
constexpr const char* kMetadataRecord = "metadata";
constexpr const char* kEventRecord = "event";

// ---- writing ---------------------------------------------------------------

ojson tensor_json(const TensorMetadata& t) {
  ojson j = {{"name", t.name}};
  if (t.shape) j["shape"] = *t.shape;
  if (t.dtype) j["dtype"] = *t.dtype;
  if (t.element_count) j["element_count"] = *t.element_count;
  if (t.byte_count) j["byte_count"] = *t.byte_count;
  return j;
}

double finite(double v, const char* field) {
  if (!std::isfinite(v))
    throw TraceIoError(std::string("cannot serialize non-finite activation ") + field);
  return v;
}

ojson capabilities_json(const AdapterCapabilities& c) {
  return {
      {"operation_timing", c.operation_timing},         {"tensor_metadata", c.tensor_metadata},
      {"layer_hierarchy", c.layer_hierarchy},           {"token_index", c.token_index},
      {"activation_summaries", c.activation_summaries}, {"attention_values", c.attention_values}};
}

// ---- reading ---------------------------------------------------------------

struct Reader {
  const ReadLimits& limits;
  std::size_t line;

  [[noreturn]] void fail(const std::string& message) const { throw TraceIoError(message, line); }

  const json* find(const json& obj, const char* key) const {
    const auto it = obj.find(key);
    if (it == obj.end() || it->is_null()) return nullptr;
    return &*it;
  }

  const json& require(const json& obj, const char* key) const {
    const json* v = find(obj, key);
    if (!v) fail(std::string("missing required field \"") + key + "\"");
    return *v;
  }

  std::string str(const json& v, const char* key) const {
    if (!v.is_string()) fail(std::string("\"") + key + "\" must be a string");
    const auto& s = v.get_ref<const std::string&>();
    if (s.size() > limits.max_string_bytes)
      fail(std::string("\"") + key + "\" exceeds " + std::to_string(limits.max_string_bytes) +
           " bytes");
    return s;
  }

  std::uint64_t u64(const json& v, const char* key) const {
    if (!v.is_number_unsigned())
      fail(std::string("\"") + key + "\" must be a non-negative integer");
    return v.get<std::uint64_t>();
  }

  std::int64_t i64(const json& v, const char* key) const {
    if (v.is_number_unsigned()) {
      const auto u = v.get<std::uint64_t>();
      if (u > static_cast<std::uint64_t>(INT64_MAX))
        fail(std::string("\"") + key + "\" is too large");
      return static_cast<std::int64_t>(u);
    }
    if (!v.is_number_integer()) fail(std::string("\"") + key + "\" must be an integer");
    return v.get<std::int64_t>();
  }

  double num(const json& v, const char* key) const {
    if (!v.is_number()) fail(std::string("\"") + key + "\" must be a number");
    return v.get<double>();
  }

  bool boolean(const json& v, const char* key) const {
    if (!v.is_boolean()) fail(std::string("\"") + key + "\" must be true or false");
    return v.get<bool>();
  }

  std::optional<std::string> opt_str(const json& obj, const char* key) const {
    if (const json* v = find(obj, key)) return str(*v, key);
    return std::nullopt;
  }
  std::optional<std::uint64_t> opt_u64(const json& obj, const char* key) const {
    if (const json* v = find(obj, key)) return u64(*v, key);
    return std::nullopt;
  }
  std::optional<std::int64_t> opt_i64(const json& obj, const char* key) const {
    if (const json* v = find(obj, key)) return i64(*v, key);
    return std::nullopt;
  }

  TensorMetadata tensor(const json& j) const {
    if (!j.is_object()) fail("tensor entries must be objects");
    TensorMetadata t;
    t.name = str(require(j, "name"), "name");
    if (const json* shape = find(j, "shape")) {
      if (!shape->is_array()) fail("\"shape\" must be an array");
      if (shape->size() > limits.max_shape_rank)
        fail("tensor rank exceeds " + std::to_string(limits.max_shape_rank));
      std::vector<std::int64_t> dims;
      dims.reserve(shape->size());
      for (const auto& d : *shape) {
        const auto dim = i64(d, "shape");
        if (dim < -1) fail("shape dimensions must be >= -1");
        dims.push_back(dim);
      }
      t.shape = std::move(dims);
    }
    t.dtype = opt_str(j, "dtype");
    t.element_count = opt_u64(j, "element_count");
    t.byte_count = opt_u64(j, "byte_count");
    return t;
  }

  std::vector<TensorMetadata> tensors(const json& obj, const char* key) const {
    std::vector<TensorMetadata> out;
    const json* arr = find(obj, key);
    if (!arr) return out;
    if (!arr->is_array()) fail(std::string("\"") + key + "\" must be an array");
    if (arr->size() > limits.max_tensors_per_event)
      fail(std::string("\"") + key + "\" has more than " +
           std::to_string(limits.max_tensors_per_event) + " tensors");
    out.reserve(arr->size());
    for (const auto& t : *arr) out.push_back(tensor(t));
    return out;
  }

  ActivationSummary activation(const json& j) const {
    if (!j.is_object()) fail("\"activation\" must be an object");
    ActivationSummary a;
    a.mean = num(require(j, "mean"), "mean");
    a.min = num(require(j, "min"), "min");
    a.max = num(require(j, "max"), "max");
    a.stddev = num(require(j, "stddev"), "stddev");
    a.nonfinite_count = u64(require(j, "nonfinite_count"), "nonfinite_count");
    a.sparsity = num(require(j, "sparsity"), "sparsity");
    a.sparsity_epsilon = num(require(j, "sparsity_epsilon"), "sparsity_epsilon");
    a.sampled_elements = u64(require(j, "sampled_elements"), "sampled_elements");
    a.full_tensor = boolean(require(j, "full_tensor"), "full_tensor");
    return a;
  }

  AdapterCapabilities capabilities(const json& j) const {
    if (!j.is_object()) fail("\"capabilities\" must be an object");
    AdapterCapabilities c;
    auto flag = [&](const char* key, bool& out) {
      if (const json* v = find(j, key)) out = boolean(*v, key);
    };
    flag("operation_timing", c.operation_timing);
    flag("tensor_metadata", c.tensor_metadata);
    flag("layer_hierarchy", c.layer_hierarchy);
    flag("token_index", c.token_index);
    flag("activation_summaries", c.activation_summaries);
    flag("attention_values", c.attention_values);
    return c;
  }

  DataSource source(const json& obj, const char* key) const {
    const json* v = find(obj, key);
    if (!v) return DataSource::None;
    const auto parsed = parse_data_source(str(*v, key));
    if (!parsed) fail(std::string("unknown ") + key + " \"" + v->get<std::string>() + "\"");
    return *parsed;
  }

  TraceMetadata metadata(const json& j) const {
    TraceMetadata m;
    m.schema_version = static_cast<int>(i64(require(j, "schema_version"), "schema_version"));
    if (m.schema_version < 1) fail("invalid schema_version");
    if (m.schema_version > kSchemaVersion)
      fail("trace uses schema_version " + std::to_string(m.schema_version) +
           "; this build reads up to " + std::to_string(kSchemaVersion));
    m.run_id = opt_str(j, "run_id").value_or("");
    m.source = source(j, "source");
    m.origin = source(j, "origin");
    m.adapter = opt_str(j, "adapter").value_or("");
    m.runtime = opt_str(j, "runtime").value_or("");
    m.runtime_version = opt_str(j, "runtime_version").value_or("");
    m.model = opt_str(j, "model").value_or("");
    m.created_utc = opt_str(j, "created_utc").value_or("");
    if (const json* c = find(j, "capabilities")) m.capabilities = capabilities(*c);
    m.buffer_capacity = opt_u64(j, "buffer_capacity").value_or(0);
    m.overwritten_events = opt_u64(j, "overwritten_events").value_or(0);
    m.rejected_events = opt_u64(j, "rejected_events").value_or(0);
    if (const json* notes = find(j, "notes")) {
      if (!notes->is_array()) fail("\"notes\" must be an array");
      if (notes->size() > limits.max_attributes)
        fail("more than " + std::to_string(limits.max_attributes) + " notes");
      for (const auto& n : *notes) m.notes.push_back(str(n, "notes"));
    }
    return m;
  }

  TraceEvent event(const json& j, std::vector<std::string>& warnings) const {
    TraceEvent e;
    e.sequence = u64(require(j, "sequence"), "sequence");
    e.timestamp_ns = u64(require(j, "timestamp_ns"), "timestamp_ns");
    e.duration_ns = opt_u64(j, "duration_ns");
    const std::string type = str(require(j, "type"), "type");
    const auto parsed_type = parse_event_type(type);
    if (!parsed_type) fail("unknown event type \"" + type + "\"");
    e.type = *parsed_type;
    e.name = str(require(j, "name"), "name");
    if (const auto category = opt_str(j, "category")) {
      if (const auto c = parse_category(*category)) {
        e.category = *c;
      } else {
        warnings.push_back("line " + std::to_string(line) + ": unknown category \"" + *category +
                           "\" read as unknown");
      }
    }
    e.op_type = opt_str(j, "op_type");
    e.parent_id = opt_str(j, "parent_id");
    e.layer_index = opt_i64(j, "layer_index");
    e.token_index = opt_i64(j, "token_index");
    e.head_index = opt_i64(j, "head_index");
    e.device = opt_str(j, "device");
    e.thread_id = opt_i64(j, "thread_id");
    e.inputs = tensors(j, "inputs");
    e.outputs = tensors(j, "outputs");
    if (const json* a = find(j, "activation")) e.activation = activation(*a);
    e.message = opt_str(j, "message").value_or("");
    if (const json* attrs = find(j, "attributes")) {
      if (!attrs->is_array()) fail("\"attributes\" must be an array of [key, value] pairs");
      if (attrs->size() > limits.max_attributes)
        fail("more than " + std::to_string(limits.max_attributes) + " attributes");
      for (const auto& kv : *attrs) {
        if (!kv.is_array() || kv.size() != 2) fail("attributes must be [key, value] pairs");
        e.attributes.emplace_back(str(kv[0], "attributes"), str(kv[1], "attributes"));
      }
    }
    if (const std::string reason = validate(e); !reason.empty()) fail("invalid event: " + reason);
    return e;
  }
};

// Nesting depth check before parsing, so a hostile line cannot build a deep tree.
bool depth_ok(const std::string& text) {
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (const char ch : text) {
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
    } else if (ch == '{' || ch == '[') {
      if (++depth > kMaxJsonDepth) return false;
    } else if (ch == '}' || ch == ']') {
      --depth;
    }
  }
  return true;
}

// Reads one line without letting it grow past `max_bytes`. Returns false at EOF.
bool read_bounded_line(std::istream& in, std::string& out, std::size_t max_bytes, bool& too_long) {
  out.clear();
  too_long = false;
  char ch = 0;
  bool any = false;
  while (in.get(ch)) {
    any = true;
    if (ch == '\n') return true;
    if (out.size() >= max_bytes) {
      too_long = true;
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      return true;
    }
    out.push_back(ch);
  }
  return any;
}

bool is_blank(const std::string& s) {
  return s.find_first_not_of(" \t\r") == std::string::npos;
}

}  // namespace

std::string serialize_metadata(const TraceMetadata& m) {
  ojson j = {{"record", kMetadataRecord},
             {"schema_version", m.schema_version},
             {"run_id", m.run_id},
             {"source", to_string(m.source)},
             {"origin", to_string(m.origin)},
             {"adapter", m.adapter},
             {"runtime", m.runtime},
             {"runtime_version", m.runtime_version},
             {"model", m.model},
             {"created_utc", m.created_utc},
             {"capabilities", capabilities_json(m.capabilities)},
             {"buffer_capacity", m.buffer_capacity},
             {"overwritten_events", m.overwritten_events},
             {"rejected_events", m.rejected_events},
             {"notes", m.notes}};
  return j.dump();
}

std::string serialize_event(const TraceEvent& e) {
  ojson j = {
      {"record", kEventRecord},    {"sequence", e.sequence}, {"timestamp_ns", e.timestamp_ns},
      {"type", to_string(e.type)}, {"name", e.name},         {"category", to_string(e.category)}};
  if (e.duration_ns) j["duration_ns"] = *e.duration_ns;
  if (e.op_type) j["op_type"] = *e.op_type;
  if (e.parent_id) j["parent_id"] = *e.parent_id;
  if (e.layer_index) j["layer_index"] = *e.layer_index;
  if (e.token_index) j["token_index"] = *e.token_index;
  if (e.head_index) j["head_index"] = *e.head_index;
  if (e.device) j["device"] = *e.device;
  if (e.thread_id) j["thread_id"] = *e.thread_id;
  if (!e.inputs.empty()) {
    ojson arr = ojson::array();
    for (const auto& t : e.inputs) arr.push_back(tensor_json(t));
    j["inputs"] = std::move(arr);
  }
  if (!e.outputs.empty()) {
    ojson arr = ojson::array();
    for (const auto& t : e.outputs) arr.push_back(tensor_json(t));
    j["outputs"] = std::move(arr);
  }
  if (e.activation) {
    const auto& a = *e.activation;
    j["activation"] = {{"mean", finite(a.mean, "mean")},
                       {"min", finite(a.min, "min")},
                       {"max", finite(a.max, "max")},
                       {"stddev", finite(a.stddev, "stddev")},
                       {"nonfinite_count", a.nonfinite_count},
                       {"sparsity", finite(a.sparsity, "sparsity")},
                       {"sparsity_epsilon", finite(a.sparsity_epsilon, "sparsity_epsilon")},
                       {"sampled_elements", a.sampled_elements},
                       {"full_tensor", a.full_tensor}};
  }
  if (!e.message.empty()) j["message"] = e.message;
  if (!e.attributes.empty()) {
    ojson arr = ojson::array();
    for (const auto& [k, v] : e.attributes) arr.push_back(ojson::array({k, v}));
    j["attributes"] = std::move(arr);
  }
  return j.dump();
}

void write_jsonl(std::ostream& out, const TraceMetadata& metadata,
                 const std::vector<TraceEvent>& events) {
  out << serialize_metadata(metadata) << '\n';
  for (const auto& e : events) out << serialize_event(e) << '\n';
}

void write_jsonl(const std::string& path, const TraceMetadata& metadata,
                 const std::vector<TraceEvent>& events) {
  namespace fs = std::filesystem;
  const fs::path target(path);
  fs::path tmp = target;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) throw TraceIoError("cannot create " + tmp.string());
    try {
      write_jsonl(out, metadata, events);
    } catch (...) {
      out.close();
      std::error_code ignored;
      fs::remove(tmp, ignored);
      throw;
    }
    out.flush();
    if (!out) {
      out.close();
      std::error_code ignored;
      fs::remove(tmp, ignored);
      throw TraceIoError("write failed for " + tmp.string());
    }
  }
  std::error_code ec;
  fs::rename(tmp, target, ec);
  if (ec) {
    std::error_code ignored;
    fs::remove(tmp, ignored);
    throw TraceIoError("cannot rename " + tmp.string() + " to " + path + ": " + ec.message());
  }
}

LoadedTrace read_jsonl(std::istream& in, const ReadLimits& limits) {
  LoadedTrace trace;
  std::string text;
  std::size_t line = 0;
  bool have_metadata = false;
  bool too_long = false;
  std::optional<std::uint64_t> last_sequence;
  while (read_bounded_line(in, text, limits.max_line_bytes, too_long)) {
    ++line;
    if (too_long)
      throw TraceIoError("record exceeds " + std::to_string(limits.max_line_bytes) + " bytes",
                         line);
    if (is_blank(text)) continue;
    if (!depth_ok(text)) throw TraceIoError("JSON nesting is too deep", line);
    const json j = json::parse(text, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded()) throw TraceIoError("not valid JSON", line);
    if (!j.is_object()) throw TraceIoError("record must be a JSON object", line);
    const Reader r{limits, line};
    const std::string record = r.str(r.require(j, "record"), "record");
    if (!have_metadata) {
      if (record != kMetadataRecord)
        throw TraceIoError("first record must be \"metadata\", got \"" + record + "\"", line);
      trace.metadata = r.metadata(j);
      have_metadata = true;
      continue;
    }
    if (record == kMetadataRecord) throw TraceIoError("duplicate metadata record", line);
    if (record != kEventRecord) {
      trace.warnings.push_back("line " + std::to_string(line) + ": skipped unknown record \"" +
                               record + "\"");
      continue;
    }
    if (trace.events.size() >= limits.max_events)
      throw TraceIoError("trace has more than " + std::to_string(limits.max_events) + " events",
                         line);
    TraceEvent e = r.event(j, trace.warnings);
    if (last_sequence && e.sequence <= *last_sequence)
      throw TraceIoError("sequence numbers must increase (got " + std::to_string(e.sequence) +
                             " after " + std::to_string(*last_sequence) + ")",
                         line);
    last_sequence = e.sequence;
    trace.events.push_back(std::move(e));
  }
  if (in.bad()) throw TraceIoError("read error", line);
  if (!have_metadata) throw TraceIoError("empty trace file: no metadata record");
  return trace;
}

LoadedTrace read_jsonl(const std::string& path, const ReadLimits& limits) {
  namespace fs = std::filesystem;
  std::error_code ec;
  const auto size = fs::file_size(path, ec);
  if (ec) throw TraceIoError("cannot open " + path + ": " + ec.message());
  if (size > limits.max_file_bytes)
    throw TraceIoError(path + " is larger than the " + std::to_string(limits.max_file_bytes) +
                       "-byte limit");
  std::ifstream in(path, std::ios::binary);
  if (!in) throw TraceIoError("cannot open " + path);
  return read_jsonl(in, limits);
}

TraceFormat detect_format(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return TraceFormat::Unknown;
  char buf[256] = {};
  in.read(buf, sizeof buf);
  const std::string head(buf, static_cast<std::size_t>(in.gcount()));
  const auto first = head.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return TraceFormat::Unknown;
  if (head[first] == '[') return TraceFormat::OnnxProfile;
  if (head[first] == '{') return TraceFormat::LlmTraceJsonl;
  return TraceFormat::Unknown;
}

LoadedTrace load_trace(const std::string& path, const ReadLimits& limits) {
  switch (detect_format(path)) {
    case TraceFormat::LlmTraceJsonl: {
      LoadedTrace t = read_jsonl(path, limits);
      if (t.metadata.origin == DataSource::None) t.metadata.origin = t.metadata.source;
      t.metadata.source = DataSource::Replay;
      return t;
    }
    case TraceFormat::OnnxProfile:
      throw TraceIoError(path +
                         " looks like an ONNX Runtime profile; profile import is not "
                         "implemented yet");
    case TraceFormat::Unknown:
      break;
  }
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) throw TraceIoError("cannot open " + path);
  throw TraceIoError(path + " is not an llm-trace JSONL file");
}

}  // namespace llm_trace
