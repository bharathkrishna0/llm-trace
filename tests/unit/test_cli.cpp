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

TEST_CASE("cli parses compare options") {
  const auto r = parse({"--compare", "a.jsonl", "b.json", "--compare-by", "op", "--top", "5",
                        "--compare-json", "-"});
  REQUIRE(r.error.empty());
  REQUIRE(r.options.compare.size() == 2);
  CHECK(r.options.compare[1] == "b.json");
  CHECK(r.options.compare_by_op);
  CHECK(r.options.top == 5);
  CHECK(*r.options.compare_json == "-");
}

TEST_CASE("cli rejects invalid compare combinations") {
  CHECK_FALSE(parse({"--compare", "a.jsonl"}).error.empty());
  CHECK_FALSE(parse({"--compare=a.jsonl", "b.jsonl"}).error.empty());
  CHECK_FALSE(parse({"--compare", "a", "b", "--open", "c"}).error.empty());
  CHECK_FALSE(parse({"--compare", "a", "b", "--demo"}).error.empty());
  CHECK_FALSE(parse({"--compare", "a", "b", "--compare-by", "layer"}).error.empty());
  CHECK_FALSE(parse({"--compare", "a", "b", "--top", "0"}).error.empty());
  CHECK_FALSE(parse({"--top", "5"}).error.empty());
  CHECK_FALSE(parse({"--compare-json", "x.json"}).error.empty());
}
