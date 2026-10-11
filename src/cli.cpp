#include "cli.hpp"

#include <cerrno>
#include <cstdlib>

namespace llm_trace::cli {

namespace {

bool parse_u64(const std::string& text, std::uint64_t min, std::uint64_t max, std::uint64_t& out) {
  if (text.empty() || text[0] == '-' || text[0] == '+') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end != text.c_str() + text.size() || v < min || v > max) return false;
  out = v;
  return true;
}

bool parse_double(const std::string& text, double min, double max, double& out) {
  if (text.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const double v = std::strtod(text.c_str(), &end);
  if (errno != 0 || end != text.c_str() + text.size() || !(v >= min && v <= max)) return false;
  out = v;
  return true;
}

bool parse_frame(const std::string& text, FrameSize& out) {
  const auto x = text.find('x');
  if (x == std::string::npos) return false;
  std::uint64_t w = 0, h = 0;
  if (!parse_u64(text.substr(0, x), 20, 1000, w) || !parse_u64(text.substr(x + 1), 5, 500, h))
    return false;
  out = {static_cast<int>(w), static_cast<int>(h)};
  return true;
}

}  // namespace

std::string usage() {
  return "llm-trace " LLM_TRACE_VERSION
         "\n"
         "Local-first tracing and replay for transformer inference.\n"
         "\n"
         "Usage:\n"
         "  llm-trace --demo [demo options] [display options]\n"
         "  llm-trace --help | --version\n"
         "\n"
         "Modes:\n"
         "      --demo             Interactive TUI over a deterministic synthetic trace.\n"
         "                         Values come from a cost model, not a real model.\n"
         "\n"
         "Demo options:\n"
         "      --layers N         Decoder layers (1-256, default 12)\n"
         "      --tokens N         Decode steps after the prompt (0-4096, default 64)\n"
         "      --seed N           PRNG seed (default 42)\n"
         "      --rate N           Decode steps per second, 0 = all at once (default 8)\n"
         "      --buffer N         Ring-buffer capacity in events (1-10000000, default 65536)\n"
         "\n"
         "Display options:\n"
         "      --no-color         Monochrome output (also honours NO_COLOR)\n"
         "      --ascii            ASCII-only symbols and separators\n"
         "      --print-frame WxH  Render one frame to stdout and exit (e.g. 120x40)\n"
         "\n"
         "  -h, --help             Show this help and exit\n"
         "      --version          Print the version and exit\n";
}

ParseResult parse(const std::vector<std::string>& args) {
  ParseResult result;
  Options& o = result.options;
  for (std::size_t i = 0; i < args.size(); ++i) {
    std::string arg = args[i];
    std::string value;
    bool has_value = false;
    if (const auto eq = arg.find('='); arg.rfind("--", 0) == 0 && eq != std::string::npos) {
      value = arg.substr(eq + 1);
      arg = arg.substr(0, eq);
      has_value = true;
    }
    auto take = [&]() -> bool {
      if (has_value) return true;
      if (i + 1 >= args.size()) return false;
      value = args[++i];
      return true;
    };
    auto fail = [&](const std::string& what) {
      result.error = what;
      return result;
    };
    std::uint64_t u = 0;
    if (arg == "-h" || arg == "--help") {
      o.help = true;
    } else if (arg == "--version") {
      o.version = true;
    } else if (arg == "--demo") {
      o.demo = true;
    } else if (arg == "--no-color") {
      o.no_color = true;
    } else if (arg == "--ascii") {
      o.ascii = true;
    } else if (arg == "--layers") {
      if (!take() || !parse_u64(value, 1, 256, u)) return fail("--layers expects 1-256");
      o.layers = static_cast<int>(u);
    } else if (arg == "--tokens") {
      if (!take() || !parse_u64(value, 0, 4096, u)) return fail("--tokens expects 0-4096");
      o.tokens = static_cast<int>(u);
    } else if (arg == "--seed") {
      if (!take() || !parse_u64(value, 0, UINT64_MAX, u)) return fail("--seed expects an integer");
      o.seed = u;
    } else if (arg == "--rate") {
      if (!take() || !parse_double(value, 0.0, 1000.0, o.rate))
        return fail("--rate expects 0-1000");
    } else if (arg == "--buffer") {
      if (!take() || !parse_u64(value, 1, 10'000'000, u))
        return fail("--buffer expects 1-10000000");
      o.buffer = static_cast<std::size_t>(u);
    } else if (arg == "--print-frame") {
      FrameSize frame;
      if (!take() || !parse_frame(value, frame))
        return fail("--print-frame expects WIDTHxHEIGHT (20-1000 x 5-500)");
      o.print_frame = frame;
    } else {
      return fail("unknown option: " + args[i]);
    }
  }
  return result;
}

}  // namespace llm_trace::cli
