#include <catch2/catch_test_macros.hpp>

#include "cli.hpp"

using llm_trace::cli::parse;

TEST_CASE("cli parses help and version flags") {
  CHECK(parse({"--help"}).options.help);
  CHECK(parse({"-h"}).options.help);
  CHECK(parse({"--version"}).options.version);
  CHECK(parse({}).error.empty());
}

TEST_CASE("cli rejects unknown options") {
  const auto result = parse({"--bogus"});
  CHECK(result.error == "unknown option: --bogus");
}
