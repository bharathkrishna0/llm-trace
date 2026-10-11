#include "cli.hpp"

namespace llm_trace::cli {

std::string usage() {
  return "llm-trace " LLM_TRACE_VERSION
         "\n"
         "Local-first tracing and replay for transformer inference.\n"
         "\n"
         "Usage:\n"
         "  llm-trace [options]\n"
         "\n"
         "Options:\n"
         "  -h, --help      Show this help and exit\n"
         "      --version   Print the version and exit\n";
}

ParseResult parse(const std::vector<std::string>& args) {
  ParseResult result;
  for (const auto& arg : args) {
    if (arg == "-h" || arg == "--help") {
      result.options.help = true;
    } else if (arg == "--version") {
      result.options.version = true;
    } else {
      result.error = "unknown option: " + arg;
      return result;
    }
  }
  return result;
}

}  // namespace llm_trace::cli
