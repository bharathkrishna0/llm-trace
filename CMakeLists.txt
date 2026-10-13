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
  src/core/trace_session.cpp)
target_include_directories(llm_trace_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(Threads REQUIRED)
target_link_libraries(llm_trace_core PUBLIC Threads::Threads)
target_compile_definitions(llm_trace_core PUBLIC LLM_TRACE_VERSION="${PROJECT_VERSION}")
llm_trace_set_warnings(llm_trace_core)

set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
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
