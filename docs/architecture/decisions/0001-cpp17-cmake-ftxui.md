# ADR 0001: C++17, CMake presets, FTXUI, Catch2, nlohmann/json

Status: accepted (M0)

**Context.** The spec asks for a portable, local-first tool that builds on Linux and Windows with no system packages beyond a compiler and CMake.

**Decision.**
- C++17 and CMake ≥ 3.20 with presets (`debug`, `release`, `msvc`).
- Every third-party dependency is fetched with `FetchContent` from a release URL pinned by SHA-256: FTXUI 5.0.0 for the TUI, Catch2 v3 for tests, nlohmann/json 3.11.3 for I/O.
- spdlog was dropped because nothing needed logging yet.

**Consequences.**
- The first configure needs network access, and later builds are reproducible.
- No package manager is required on Windows.
- A dependency bump means changing a URL and a hash.
