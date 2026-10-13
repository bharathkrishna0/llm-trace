#pragma once

#include <string>

namespace llm_trace::detail {

// Returns false if JSON text nests objects/arrays deeper than `max_depth`. Run before
// parsing untrusted input so a hostile file cannot build a very deep tree.
inline bool json_depth_within(const std::string& text, int max_depth) {
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
      if (++depth > max_depth) return false;
    } else if (ch == '}' || ch == ']') {
      --depth;
    }
  }
  return true;
}

}  // namespace llm_trace::detail
