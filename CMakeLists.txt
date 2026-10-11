cmake_minimum_required(VERSION 3.20)

project(llm_trace
  VERSION 0.1.0
  DESCRIPTION "Local-first tracing and replay for transformer inference"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LLM_TRACE_BUILD_TESTS "Build unit and integration tests" ON)
option(LLM_TRACE_BUILD_BENCH "Build the llm_trace_bench micro-benchmarks" ON)
option(LLM_TRACE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

function(llm_trace_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
    if(LLM_TRACE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
    if(LLM_TRACE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

# Core library: must not depend on any TUI library.
add_library(llm_trace_core
  src/core/event.cpp
  src/core/format.cpp
  src/core/collector.cpp
  src/core/synthetic.cpp
  src/core/trace_session.cpp
  src/core/trace_io.cpp
  src/adapters/onnx_profile_adapter.cpp)
target_include_directories(llm_trace_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(llm_trace_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
find_package(Threads REQUIRED)
target_link_libraries(llm_trace_core PUBLIC Threads::Threads)
# nlohmann_json is linked further down, after FetchContent makes it available.
target_compile_definitions(llm_trace_core PUBLIC LLM_TRACE_VERSION="${PROJECT_VERSION}")
llm_trace_set_warnings(llm_trace_core)

set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
FetchContent_MakeAvailable(nlohmann_json)
target_link_libraries(llm_trace_core PRIVATE nlohmann_json::nlohmann_json)

FetchContent_Declare(ftxui
  URL https://github.com/ArthurSonzogni/FTXUI/archive/refs/tags/v5.0.0.tar.gz
  URL_HASH SHA256=a2991cb222c944aee14397965d9f6b050245da849d8c5da7c72d112de2786b5b
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
FetchContent_MakeAvailable(ftxui)

# TUI library: FTXUI rendering and key handling over TraceSession.
add_library(llm_trace_tui
  src/tui/app.cpp
  src/tui/components.cpp
  src/tui/controller.cpp
  src/tui/key_bindings.cpp)
target_include_directories(llm_trace_tui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(llm_trace_tui PUBLIC llm_trace_core ftxui::component ftxui::dom ftxui::screen)
llm_trace_set_warnings(llm_trace_tui)

add_executable(llm-trace src/main.cpp src/cli.cpp)
target_include_directories(llm-trace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(llm-trace PRIVATE llm_trace_tui)
if(MSVC)
  # std::getenv is only read (NO_COLOR); silence C4996.
  target_compile_definitions(llm-trace PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
llm_trace_set_warnings(llm-trace)

if(LLM_TRACE_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
    URL_HASH SHA256=c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(Catch2)
  add_subdirectory(tests)
endif()

if(LLM_TRACE_BUILD_BENCH)
  add_executable(llm_trace_bench bench/llm_trace_bench.cpp)
  target_link_libraries(llm_trace_bench PRIVATE llm_trace_core llm_trace_tui)
  target_include_directories(llm_trace_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(llm_trace_bench PRIVATE LLM_TRACE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  llm_trace_set_warnings(llm_trace_bench)
endif()

# Optional: the ONNX Runtime sample that writes a real profile and measures profiling
# overhead. Needs an unpacked ONNX Runtime release, e.g.
#   cmake --preset release -DLLM_TRACE_BUILD_ORT_RUNNER=ON -DONNXRUNTIME_ROOT=~/deps/onnxruntime-linux-x64-1.20.1
option(LLM_TRACE_BUILD_ORT_RUNNER "Build examples/onnx/onnx_profile_runner" OFF)
if(LLM_TRACE_BUILD_ORT_RUNNER)
  set(ONNXRUNTIME_ROOT "" CACHE PATH "Unpacked ONNX Runtime release directory")
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS ${ONNXRUNTIME_ROOT}/include NO_DEFAULT_PATH REQUIRED)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH REQUIRED)
  add_executable(onnx_profile_runner examples/onnx/onnx_profile_runner.cpp)
  target_include_directories(onnx_profile_runner PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(onnx_profile_runner PRIVATE ${ONNXRUNTIME_LIBRARY})
  set_target_properties(onnx_profile_runner PROPERTIES BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib)
endif()
