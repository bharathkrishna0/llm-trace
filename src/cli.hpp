#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llm_trace::cli {

struct FrameSize {
  int width = 0;
  int height = 0;
};

struct Options {
  bool help = false;
  bool version = false;
  bool demo = false;
  bool no_color = false;
  bool ascii = false;
  std::size_t buffer = 65536;
  int layers = 12;
  int tokens = 64;
  double rate = 8.0;
  std::uint64_t seed = 42;
  std::optional<FrameSize> print_frame;
};

struct ParseResult {
  Options options;
  std::string error;
};

ParseResult parse(const std::vector<std::string>& args);
std::string usage();

}  // namespace llm_trace::cli
