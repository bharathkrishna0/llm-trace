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

TEST_CASE("cli parses open and output paths") {
  const auto open = parse({"--open", "trace.jsonl", "--output=copy.jsonl"});
  REQUIRE(open.error.empty());
  CHECK(*open.options.open == "trace.jsonl");
  CHECK(*open.options.output == "copy.jsonl");
  CHECK(parse({"--demo", "--output", "demo.jsonl"}).error.empty());
  CHECK(parse({"--open"}).error == "--open expects a file path");
  CHECK(parse({"--open="}).error == "--open expects a file path");
  CHECK(parse({"--open", "a", "--demo"}).error == "--open and --demo cannot be combined");
  CHECK(parse({"--output", "a"}).error == "--output needs --demo or --open");
  CHECK(parse({"--open", "a", "--output", "a"}).error == "--output must differ from --open");
}
