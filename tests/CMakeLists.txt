include(Catch)

add_executable(llm_trace_tests
  unit/test_format.cpp
  unit/test_cli.cpp
  unit/test_ring_buffer.cpp
  unit/test_collector.cpp
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
