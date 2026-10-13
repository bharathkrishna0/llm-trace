#include "llm_trace/event_filter.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "llm_trace/metrics.hpp"

namespace llm_trace {

namespace {

constexpr std::size_t kMaxTerms = 16;

char lower(char ch) {
  return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
}

std::optional<std::int64_t> parse_int(const std::string& text) {
  if (text.empty()) return std::nullopt;
  std::size_t i = text[0] == '-' ? 1 : 0;
  if (i == text.size()) return std::nullopt;
  for (std::size_t j = i; j < text.size(); ++j)
    if (!std::isdigit(static_cast<unsigned char>(text[j]))) return std::nullopt;
  errno = 0;
  const long long value = std::strtoll(text.c_str(), nullptr, 10);
  if (errno == ERANGE) return std::nullopt;
  return static_cast<std::int64_t>(value);
}

// "50us", "1.5ms", "200ns", "2s" -> nanoseconds.
std::optional<std::int64_t> parse_duration(const std::string& text, std::string& error) {
  std::size_t split = 0;
  while (split < text.size() &&
         (std::isdigit(static_cast<unsigned char>(text[split])) || text[split] == '.'))
    ++split;
  const std::string number = text.substr(0, split);
  const std::string unit = text.substr(split);
  if (number.empty() || std::count(number.begin(), number.end(), '.') > 1) {
    error = "duration must be a number with a unit, e.g. dur>50us";
    return std::nullopt;
  }
  double scale = 0;
  if (unit == "ns")
    scale = 1;
  else if (unit == "us" || unit == "\xC2\xB5s")
    scale = 1e3;
  else if (unit == "ms")
    scale = 1e6;
  else if (unit == "s")
    scale = 1e9;
  else {
    error = unit.empty() ? "duration needs a unit: ns, us, ms or s"
                         : "unknown duration unit '" + unit + "' (use ns, us, ms or s)";
    return std::nullopt;
  }
  const double ns = std::strtod(number.c_str(), nullptr) * scale;
  if (!std::isfinite(ns) || ns > 9.0e18) {
    error = "duration is too large";
    return std::nullopt;
  }
  return static_cast<std::int64_t>(std::llround(ns));
}

std::optional<EventFilter::Field> field_named(const std::string& key) {
  using F = EventFilter::Field;
  if (key == "name") return F::Name;
  if (key == "op") return F::OpType;
  if (key == "device" || key == "dev") return F::Device;
  if (key == "cat" || key == "category") return F::Category;
  if (key == "type") return F::Type;
  if (key == "layer") return F::Layer;
  if (key == "token") return F::Token;
  if (key == "head") return F::Head;
  if (key == "dur" || key == "duration") return F::Duration;
  return std::nullopt;
}

bool is_numeric(EventFilter::Field f) {
  using F = EventFilter::Field;
  return f == F::Layer || f == F::Token || f == F::Head || f == F::Duration;
}

// Splits on whitespace, keeping "quoted text" as one token (quotes removed).
bool tokenize(const std::string& text, std::vector<std::string>& out, std::string& error) {
  std::string current;
  bool quoted = false;
  bool have = false;
  for (const char ch : text) {
    if (ch == '"') {
      quoted = !quoted;
      have = true;
    } else if (!quoted && std::isspace(static_cast<unsigned char>(ch))) {
      if (have) out.push_back(current);
      current.clear();
      have = false;
    } else {
      current.push_back(ch);
      have = true;
    }
  }
  if (quoted) {
    error = "unclosed quote";
    return false;
  }
  if (have) out.push_back(current);
  return true;
}

bool parse_term(std::string token, EventFilter::Term& term, std::string& error) {
  using F = EventFilter::Field;
  using C = EventFilter::Compare;
  if (token.size() > 1 && token[0] == '-') {
    term.negated = true;
    token.erase(0, 1);
  }

  // Find the first operator character after a field name.
  std::size_t op_pos = token.find_first_of(":<>=");
  std::optional<F> field;
  if (op_pos != std::string::npos && op_pos > 0)
    field = field_named(to_lower(token.substr(0, op_pos)));
  if (!field) {
    if (op_pos != std::string::npos && op_pos > 0 && token[op_pos] == ':') {
      error = "unknown filter key '" + token.substr(0, op_pos) +
              "' (use name, op, cat, type, layer, token, head, device or dur)";
      return false;
    }
    term.field = F::Any;
    term.compare = C::Contains;
    term.text = to_lower(token);
    if (const auto layer = parse_int(token)) term.any_layer = *layer;
    return true;
  }

  term.field = *field;
  std::string op;
  std::size_t value_pos = op_pos;
  while (value_pos < token.size() && (token[value_pos] == ':' || token[value_pos] == '<' ||
                                      token[value_pos] == '>' || token[value_pos] == '='))
    op.push_back(token[value_pos++]);
  const std::string value = token.substr(value_pos);
  const std::string key = token.substr(0, op_pos);
  if (value.empty()) {
    error = "'" + key + op + "' needs a value";
    return false;
  }

  if (!is_numeric(term.field)) {
    if (op != ":" && op != "=") {
      error = "'" + key + "' only supports ':' (e.g. " + key + ":value)";
      return false;
    }
    const std::string lowered = to_lower(value);
    if (term.field == F::Category) {
      const auto category = parse_category(lowered);
      if (!category) {
        error = "unknown category '" + value + "'";
        return false;
      }
      term.compare = C::Equal;
      term.enum_value = static_cast<int>(*category);
    } else if (term.field == F::Type) {
      const auto type = parse_event_type(lowered);
      if (!type) {
        error = "unknown event type '" + value + "'";
        return false;
      }
      term.compare = C::Equal;
      term.enum_value = static_cast<int>(*type);
    } else if (lowered == "missing" && term.field != F::Name) {
      term.compare = C::Missing;
    } else {
      term.compare = C::Contains;
      term.text = lowered;
    }
    return true;
  }

  auto number = [&](const std::string& text) -> std::optional<std::int64_t> {
    if (term.field == F::Duration) return parse_duration(to_lower(text), error);
    auto v = parse_int(text);
    if (!v) error = "'" + key + "' needs an integer, got '" + text + "'";
    return v;
  };

  if ((op == ":" || op == "=") && to_lower(value) == "missing") {
    term.compare = C::Missing;
    return true;
  }
  if (op == ":" || op == "=" || op == "==") {
    const std::size_t dots = value.find("..");
    if (dots != std::string::npos) {
      const auto lo = number(value.substr(0, dots));
      if (!lo) return false;
      const auto hi = number(value.substr(dots + 2));
      if (!hi) return false;
      if (*lo > *hi) {
        error = "range " + value + " is empty";
        return false;
      }
      term.compare = C::Range;
      term.low = *lo;
      term.high = *hi;
      return true;
    }
    const auto v = number(value);
    if (!v) return false;
    term.compare = C::Equal;
    term.low = *v;
    return true;
  }
  C compare;
  if (op == "<")
    compare = C::Less;
  else if (op == "<=")
    compare = C::LessEqual;
  else if (op == ">")
    compare = C::Greater;
  else if (op == ">=")
    compare = C::GreaterEqual;
  else {
    error = "unknown operator '" + op + "' (use :, <, <=, >, >=)";
    return false;
  }
  const auto v = number(value);
  if (!v) return false;
  term.compare = compare;
  term.low = *v;
  return true;
}

std::optional<std::int64_t> numeric_value(const TraceEvent& e, EventFilter::Field field) {
  using F = EventFilter::Field;
  switch (field) {
    case F::Layer:
      return e.layer_index;
    case F::Token:
      return e.token_index;
    case F::Head:
      return e.head_index;
    case F::Duration:
      if (!e.duration_ns ||
          *e.duration_ns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        return std::nullopt;
      return static_cast<std::int64_t>(*e.duration_ns);
    default:
      return std::nullopt;
  }
}

bool term_matches(const EventFilter::Term& t, const TraceEvent& e) {
  using F = EventFilter::Field;
  using C = EventFilter::Compare;
  switch (t.field) {
    case F::Any:
      if (t.any_layer && e.layer_index == t.any_layer) return true;
      return contains_ignore_case(e.name, t.text) ||
             contains_ignore_case(to_string(e.category), t.text) ||
             contains_ignore_case(to_string(e.type), t.text) ||
             (e.op_type && contains_ignore_case(*e.op_type, t.text));
    case F::Name:
      return contains_ignore_case(e.name, t.text);
    case F::OpType:
      if (t.compare == C::Missing) return !e.op_type;
      return e.op_type && contains_ignore_case(*e.op_type, t.text);
    case F::Device:
      if (t.compare == C::Missing) return !e.device;
      return e.device && contains_ignore_case(*e.device, t.text);
    case F::Category:
      return static_cast<int>(e.category) == t.enum_value;
    case F::Type:
      return static_cast<int>(e.type) == t.enum_value;
    case F::Layer:
    case F::Token:
    case F::Head:
    case F::Duration:
      break;
  }
  const auto v = numeric_value(e, t.field);
  if (t.compare == C::Missing) return !v;
  if (!v) return false;
  switch (t.compare) {
    case C::Equal:
      return *v == t.low;
    case C::Less:
      return *v < t.low;
    case C::LessEqual:
      return *v <= t.low;
    case C::Greater:
      return *v > t.low;
    case C::GreaterEqual:
      return *v >= t.low;
    case C::Range:
      return *v >= t.low && *v <= t.high;
    default:
      return false;
  }
}

}  // namespace

bool contains_ignore_case(const std::string& haystack, const std::string& lowered_needle) {
  if (lowered_needle.empty()) return true;
  if (lowered_needle.size() > haystack.size()) return false;
  const auto it = std::search(haystack.begin(), haystack.end(), lowered_needle.begin(),
                              lowered_needle.end(), [](char a, char b) { return lower(a) == b; });
  return it != haystack.end();
}

EventFilter EventFilter::parse(const std::string& text) {
  EventFilter filter;
  std::vector<std::string> tokens;
  if (!tokenize(text, tokens, filter.error_)) return filter;
  if (tokens.size() > kMaxTerms) {
    filter.error_ = "too many terms (max " + std::to_string(kMaxTerms) + ")";
    return filter;
  }
  for (const auto& token : tokens) {
    Term term;
    if (!parse_term(token, term, filter.error_)) {
      filter.terms_.clear();
      return filter;
    }
    filter.terms_.push_back(std::move(term));
  }
  return filter;
}

bool EventFilter::matches(const TraceEvent& event) const {
  if (!valid()) return false;
  for (const auto& term : terms_)
    if (term_matches(term, event) == term.negated) return false;
  return true;
}

}  // namespace llm_trace
