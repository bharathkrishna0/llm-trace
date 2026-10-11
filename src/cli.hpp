#pragma once

#include <string>
#include <vector>

namespace llm_trace::cli {

struct Options {
  bool help = false;
  bool version = false;
};

struct ParseResult {
  Options options;
  std::string error;  // non-empty when parsing failed
};

ParseResult parse(const std::vector<std::string>& args);
std::string usage();

}  // namespace llm_trace::cli
