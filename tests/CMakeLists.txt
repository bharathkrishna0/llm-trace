include(Catch)

add_executable(llm_trace_tests
  unit/test_format.cpp
  unit/test_cli.cpp
  unit/test_ring_buffer.cpp
  unit/test_collector.cpp
  unit/test_activation.cpp
  unit/test_anomaly.cpp
  unit/test_compare.cpp
  unit/test_event_filter.cpp
  unit/test_trace_session.cpp
  unit/test_synthetic.cpp
  unit/test_tui.cpp
  unit/test_trace_io.cpp
  unit/test_onnx_profile.cpp
  ${CMAKE_SOURCE_DIR}/src/cli.cpp)
target_include_directories(llm_trace_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(llm_trace_tests PRIVATE llm_trace_tui Catch2::Catch2WithMain)
target_compile_definitions(llm_trace_tests PRIVATE
  LLM_TRACE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
llm_trace_set_warnings(llm_trace_tests)
catch_discover_tests(llm_trace_tests)

add_test(NAME cli_help COMMAND llm-trace --help)
add_test(NAME cli_version COMMAND llm-trace --version)
set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "llm-trace 0\\.1\\.0")
add_test(NAME cli_rejects_unknown_option COMMAND llm-trace --definitely-not-an-option)
set_tests_properties(cli_rejects_unknown_option PROPERTIES WILL_FAIL TRUE)

add_test(NAME cli_print_frame_demo COMMAND llm-trace --demo --no-color --ascii --tokens 4 --print-frame 120x30)
set_tests_properties(cli_print_frame_demo PROPERTIES PASS_REGULAR_EXPRESSION "SYNTHETIC DEMO")
add_test(NAME cli_rejects_bad_buffer COMMAND llm-trace --demo --buffer 0)
set_tests_properties(cli_rejects_bad_buffer PROPERTIES WILL_FAIL TRUE)

# Save a demo trace, replay it, and reject a malformed file with its line number.
add_test(NAME cli_demo_output
  COMMAND llm-trace --demo --no-color --ascii --layers 1 --tokens 2 --output ${CMAKE_CURRENT_BINARY_DIR}/cli_demo.jsonl --print-frame 100x20)
add_test(NAME cli_open_replay
  COMMAND llm-trace --open ${CMAKE_CURRENT_BINARY_DIR}/cli_demo.jsonl --no-color --ascii --print-frame 120x30)
set_tests_properties(cli_open_replay PROPERTIES DEPENDS cli_demo_output
  PASS_REGULAR_EXPRESSION "SYNTHETIC DEMO.*REPLAY")
add_test(NAME cli_open_fixture
  COMMAND llm-trace --open ${CMAKE_SOURCE_DIR}/tests/fixtures/jsonl/synthetic_small.jsonl --no-color --ascii --print-frame 120x30)
set_tests_properties(cli_open_fixture PROPERTIES PASS_REGULAR_EXPRESSION "41 events")
add_test(NAME cli_open_malformed
  COMMAND llm-trace --open ${CMAKE_SOURCE_DIR}/tests/fixtures/jsonl/malformed_line3.jsonl --print-frame 80x20)
set_tests_properties(cli_open_malformed PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_open_missing
  COMMAND llm-trace --open ${CMAKE_CURRENT_BINARY_DIR}/does-not-exist.jsonl --print-frame 80x20)
set_tests_properties(cli_open_missing PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_open_onnx_profile
  COMMAND llm-trace --open ${CMAKE_SOURCE_DIR}/tests/fixtures/onnx_profiles/tiny_transformer_ort-1.20.1.json --no-color --ascii --print-frame 120x30)
set_tests_properties(cli_open_onnx_profile PROPERTIES PASS_REGULAR_EXPRESSION "ONNX")

# Compare two traces: a fixture against itself, the demo trace against the fixture, and JSON.
add_test(NAME cli_compare_same
  COMMAND llm-trace --compare ${CMAKE_SOURCE_DIR}/tests/fixtures/jsonl/synthetic_small.jsonl ${CMAKE_SOURCE_DIR}/tests/fixtures/jsonl/synthetic_small.jsonl --ascii)
set_tests_properties(cli_compare_same PROPERTIES PASS_REGULAR_EXPRESSION "By category.*\\+0\\.0%.*Only in A: 0")
add_test(NAME cli_compare_demo_vs_fixture
  COMMAND llm-trace --compare ${CMAKE_SOURCE_DIR}/tests/fixtures/jsonl/synthetic_small.jsonl ${CMAKE_CURRENT_BINARY_DIR}/cli_demo.jsonl --compare-by op --ascii)
set_tests_properties(cli_compare_demo_vs_fixture PROPERTIES DEPENDS cli_demo_output
  PASS_REGULAR_EXPRESSION "synthetic demo data")
add_test(NAME cli_compare_onnx_json
  COMMAND llm-trace --compare ${CMAKE_SOURCE_DIR}/tests/fixtures/jsonl/synthetic_small.jsonl ${CMAKE_SOURCE_DIR}/tests/fixtures/onnx_profiles/tiny_transformer_ort-1.20.1.json --compare-json -)
set_tests_properties(cli_compare_onnx_json PROPERTIES PASS_REGULAR_EXPRESSION "llm-trace-compare/1.*Different sources")
add_test(NAME cli_compare_missing
  COMMAND llm-trace --compare ${CMAKE_SOURCE_DIR}/tests/fixtures/jsonl/synthetic_small.jsonl ${CMAKE_CURRENT_BINARY_DIR}/does-not-exist.jsonl)
set_tests_properties(cli_compare_missing PROPERTIES WILL_FAIL TRUE)

# Heuristic rule checks: clean fixture exits 0, matches exit 1, bad rules exit 2.
set(LT_ORT_FIXTURE ${CMAKE_SOURCE_DIR}/tests/fixtures/onnx_profiles/tiny_transformer_ort-1.20.1.json)
add_test(NAME cli_check_clean COMMAND llm-trace --check ${LT_ORT_FIXTURE})
set_tests_properties(cli_check_clean PROPERTIES PASS_REGULAR_EXPRESSION "Flagged events: 0")
add_test(NAME cli_check_matches
  COMMAND llm-trace --check ${LT_ORT_FIXTURE} --rules ${CMAKE_SOURCE_DIR}/tests/fixtures/rules/slow_ops.json --top 2)
set_tests_properties(cli_check_matches PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_check_matches_output
  COMMAND llm-trace --check ${LT_ORT_FIXTURE} --rules ${CMAKE_SOURCE_DIR}/tests/fixtures/rules/slow_ops.json --top 2)
set_tests_properties(cli_check_matches_output PROPERTIES
  PASS_REGULAR_EXPRESSION "not a measured finding.*Flagged events: [1-9].*over-4us: matches")
add_test(NAME cli_check_bad_rules
  COMMAND llm-trace --check ${LT_ORT_FIXTURE} --rules ${CMAKE_SOURCE_DIR}/tests/fixtures/rules/bad_field.json)
set_tests_properties(cli_check_bad_rules PROPERTIES WILL_FAIL TRUE)
add_test(NAME cli_open_with_rules
  COMMAND llm-trace --open ${LT_ORT_FIXTURE} --rules ${CMAKE_SOURCE_DIR}/tests/fixtures/rules/slow_ops.json --no-color --ascii --print-frame 120x30)
set_tests_properties(cli_open_with_rules PROPERTIES PASS_REGULAR_EXPRESSION " !/layers")

add_test(NAME cli_open_activations
  COMMAND llm-trace --open ${CMAKE_SOURCE_DIR}/tests/fixtures/jsonl/tiny_activations_ort-1.20.1.jsonl --no-color --ascii --print-frame 160x45)
set_tests_properties(cli_open_activations PROPERTIES PASS_REGULAR_EXPRESSION "ACTIVATION SUMMARY")
