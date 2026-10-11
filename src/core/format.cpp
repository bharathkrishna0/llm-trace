#include <cctype>
#include <cstdio>
#include <string>

#include "llm_trace/metrics.hpp"

namespace llm_trace {

namespace {

std::string printf_string(const char* format, double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), format, value);
  return buffer;
}

}  // namespace

std::string format_duration_ns(std::uint64_t ns, bool ascii) {
  const double value = static_cast<double>(ns);
  if (ns < 1'000) return std::to_string(ns) + " ns";
  if (ns < 1'000'000) return printf_string("%.1f", value / 1e3) + (ascii ? " us" : " \xC2\xB5s");
  if (ns < 1'000'000'000) return printf_string("%.2f", value / 1e6) + " ms";
  return printf_string("%.3f", value / 1e9) + " s";
}

std::string format_duration_ns(const std::optional<std::uint64_t>& ns, bool ascii) {
  return ns ? format_duration_ns(*ns, ascii) : std::string("n/a");
}

std::string format_bytes(std::uint64_t bytes) {
  const double value = static_cast<double>(bytes);
  if (bytes < 1024) return std::to_string(bytes) + " B";
  if (bytes < 1024ull * 1024) return printf_string("%.1f", value / 1024.0) + " KiB";
  if (bytes < 1024ull * 1024 * 1024) return printf_string("%.1f", value / 1048576.0) + " MiB";
  return printf_string("%.2f", value / 1073741824.0) + " GiB";
}

std::string format_count(std::uint64_t count) {
  std::string digits = std::to_string(count);
  std::string out;
  int group = 0;
  for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
    if (group == 3) {
      out.push_back(',');
      group = 0;
    }
    out.push_back(*it);
    ++group;
  }
  return std::string(out.rbegin(), out.rend());
}

std::string format_percent(double fraction) {
  return printf_string("%.1f", fraction * 100.0) + "%";
}

std::string format_shape(const std::vector<std::int64_t>& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) out += ", ";
    out += shape[i] < 0 ? std::string("?") : std::to_string(shape[i]);
  }
  return out + "]";
}

std::string format_offset_ns(std::uint64_t ns) {
  const std::uint64_t ms_total = ns / 1'000'000;
  const unsigned long long hours = ms_total / 3'600'000;
  const unsigned long long minutes = (ms_total / 60'000) % 60;
  const unsigned long long seconds = (ms_total / 1000) % 60;
  const unsigned long long millis = ms_total % 1000;
  char buffer[48];
  std::snprintf(buffer, sizeof(buffer), "%02llu:%02llu:%02llu.%03llu", hours, minutes, seconds,
                millis);
  return buffer;
}

std::size_t utf8_length(const std::string& text) {
  std::size_t count = 0;
  for (unsigned char c : text)
    if ((c & 0xC0) != 0x80) ++count;
  return count;
}

std::string truncate_utf8(const std::string& text, std::size_t max_columns, bool ascii) {
  if (utf8_length(text) <= max_columns) return text;
  const std::string ellipsis = ascii ? "..." : "\xE2\x80\xA6";
  const std::size_t ellipsis_columns = ascii ? 3 : 1;
  if (max_columns <= ellipsis_columns) return ascii ? std::string(max_columns, '.') : ellipsis;
  const std::size_t keep = max_columns - ellipsis_columns;
  std::size_t columns = 0;
  std::size_t byte = 0;
  while (byte < text.size()) {
    const auto c = static_cast<unsigned char>(text[byte]);
    if ((c & 0xC0) != 0x80) {
      if (columns == keep) break;
      ++columns;
    }
    ++byte;
  }
  return text.substr(0, byte) + ellipsis;
}

std::string to_lower(std::string text) {
  for (auto& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return text;
}

}  // namespace llm_trace
