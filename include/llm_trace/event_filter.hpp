#pragma once

// Filter expressions for the operation list. Whitespace-separated terms are ANDed.
//
//   matmul              case-insensitive substring of name, category, type or op type;
//                       a bare integer also matches that layer index exactly
//   "layer.0 attn"      quotes keep spaces inside one term
//   name:qkv  op:MatMul  device:cpu      case-insensitive substring of that field
//   cat:attention  type:warning          exact category / event type name
//   layer:3  layer:2..5  layer>=10       integer comparisons (also token:, head:)
//   dur>50us  dur<=1.5ms  dur:missing    duration with a unit (ns, us, ms, s)
//   op:missing  device:missing           the event does not carry that field
//   -cat:session                         a leading '-' negates any term
//
// Fields an event does not carry never satisfy a comparison. See README "Filtering".

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "llm_trace/event.hpp"

namespace llm_trace {

class EventFilter {
 public:
  enum class Field { Any, Name, OpType, Device, Category, Type, Layer, Token, Head, Duration };
  enum class Compare { Contains, Equal, Less, LessEqual, Greater, GreaterEqual, Range, Missing };

  struct Term {
    Field field = Field::Any;
    Compare compare = Compare::Contains;
    bool negated = false;
    std::string text;  // lower-cased, for Contains
    std::int64_t low = 0;
    std::int64_t high = 0;                  // Range upper bound (inclusive)
    std::optional<std::int64_t> any_layer;  // free-text integer also matches this layer
    int enum_value = 0;                     // Category or EventType for Equal on those fields
  };

  // Never throws. On a syntax error valid() is false and error() explains it.
  static EventFilter parse(const std::string& text);

  bool empty() const { return terms_.empty(); }
  bool valid() const { return error_.empty(); }
  const std::string& error() const { return error_; }
  const std::vector<Term>& terms() const { return terms_; }

  // An invalid filter matches nothing.
  bool matches(const TraceEvent& event) const;

 private:
  std::vector<Term> terms_;
  std::string error_;
};

// Case-insensitive (ASCII) substring test; `lowered_needle` must already be lower case.
bool contains_ignore_case(const std::string& haystack, const std::string& lowered_needle);

}  // namespace llm_trace
