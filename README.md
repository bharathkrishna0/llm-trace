# llm-trace

A lightweight, local-first instrumentation, tracing, and replay platform for transformer inference, written in C++17 with a keyboard-driven terminal UI.

> **Status:** early development. This milestone provides the build, CLI skeleton, tests and CI only. Features are documented here as they land.

## Build

Requirements: CMake 3.20+, a C++17 compiler (GCC 11+, Clang 14+ or Visual Studio 2022), Ninja on Linux. Dependencies are fetched and pinned by CMake.

Linux:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
./build/debug/llm-trace --help
```

Windows (PowerShell, Visual Studio 2022):

```powershell
cmake --preset msvc
cmake --build --preset msvc-release
ctest --preset msvc-release
.\build\msvc\Release\llm-trace.exe --help
```

## License

MIT — see [LICENSE](LICENSE).
