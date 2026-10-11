#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llm_trace {

// Value formatting shared by the CLI summary and the TUI. Every value carries a unit.
std::string format_duration_ns(std::uint64_t ns, bool ascii = false);
std::string format_duration_ns(const std::optional<std::uint64_t>& ns, bool ascii = false);
std::string format_bytes(std::uint64_t bytes);
std::string format_count(std::uint64_t count);
std::string format_percent(double fraction);
std::string format_shape(const std::vector<std::int64_t>& shape);
std::string format_offset_ns(std::uint64_t ns);  // "00:00:02.418"

// Truncates to at most max_columns code points, appending "..." (or "…") when cut.
// Never splits a UTF-8 sequence.
std::string truncate_utf8(const std::string& text, std::size_t max_columns, bool ascii = false);
std::size_t utf8_length(const std::string& text);
std::string to_lower(std::string text);

}  // namespace llm_trace
