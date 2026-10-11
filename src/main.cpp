#include <cstdio>
#include <string>
#include <vector>

#include "cli.hpp"

int main(int argc, char** argv) {
  using namespace llm_trace;
  const std::vector<std::string> args(argv + 1, argv + argc);
  const cli::ParseResult parsed = cli::parse(args);
  if (!parsed.error.empty()) {
    std::fprintf(stderr, "llm-trace: %s\n\n%s", parsed.error.c_str(), cli::usage().c_str());
    return 2;
  }
  if (parsed.options.help) {
    std::fputs(cli::usage().c_str(), stdout);
    return 0;
  }
  if (parsed.options.version) {
    std::puts("llm-trace " LLM_TRACE_VERSION);
    return 0;
  }
  std::fputs(cli::usage().c_str(), stdout);
  return 0;
}
