#include <catch2/catch_test_macros.hpp>

#include "llm_trace/metrics.hpp"

using namespace llm_trace;

TEST_CASE("durations always carry a unit") {
  CHECK(format_duration_ns(std::uint64_t{0}) == "0 ns");
  CHECK(format_duration_ns(std::uint64_t{999}) == "999 ns");
  CHECK(format_duration_ns(std::uint64_t{1'500}, true) == "1.5 us");
  CHECK(format_duration_ns(std::uint64_t{1'240'000}) == "1.24 ms");
  CHECK(format_duration_ns(std::uint64_t{2'500'000'000}) == "2.500 s");
  CHECK(format_duration_ns(std::optional<std::uint64_t>{}) == "n/a");
}

TEST_CASE("bytes counts and shapes are formatted") {
  CHECK(format_bytes(512) == "512 B");
  CHECK(format_bytes(98304) == "96.0 KiB");
  CHECK(format_count(49152) == "49,152");
  CHECK(format_count(0) == "0");
  CHECK(format_count(1000000) == "1,000,000");
  CHECK(format_shape({1, 12, 4096}) == "[1, 12, 4096]");
  CHECK(format_shape({-1, 8}) == "[?, 8]");
  CHECK(format_offset_ns(2'418'000'000ull) == "00:00:02.418");
}

TEST_CASE("truncate_utf8 never splits a multi-byte sequence") {
  CHECK(truncate_utf8("short", 10) == "short");
  CHECK(truncate_utf8("decoder.layer.03.attention", 10, true) == "decoder...");
  const std::string greek = "\xCE\xB1\xCE\xB2\xCE\xB3\xCE\xB4\xCE\xB5";  // five 2-byte chars
  const std::string cut = truncate_utf8(greek, 3);
  CHECK(utf8_length(cut) == 3);
  CHECK(cut == "\xCE\xB1\xCE\xB2\xE2\x80\xA6");
  CHECK(truncate_utf8("abcdef", 2, true) == "..");
}
