include(Catch)

add_executable(llm_trace_tests
  unit/test_format.cpp
  unit/test_cli.cpp
  unit/test_ring_buffer.cpp
  unit/test_collector.cpp
  unit/test_trace_session.cpp
  unit/test_synthetic.cpp
  unit/test_tui.cpp
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
