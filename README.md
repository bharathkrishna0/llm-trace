# llm-trace

A lightweight, local-first instrumentation, tracing, and replay platform for transformer inference, written in C++17 with a keyboard-driven terminal UI.

> **Status:** early development. Available so far: the core event model, bounded ring-buffer storage, and a keyboard-driven TUI over a synthetic demo trace. saving traces to JSONL and replaying them offline. The ONNX Runtime adapter is not implemented yet.

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

## Synthetic demo

```bash
./build/debug/llm-trace --demo            # Windows: .\build\msvc\Debug\llm-trace.exe --demo
./build/debug/llm-trace --demo --no-color --ascii
./build/debug/llm-trace --demo --print-frame 120x40   # print one frame and exit
```

The demo trace is deterministic for a given `--seed`. Its timings come from a fixed cost model, **not from a real model**. The TUI marks it `[DEMO]` / `SYNTHETIC DEMO` at all times, and attention values and activation summaries are reported as "Not exposed by adapter".

Keys: `j`/`k` or arrows move, `Tab` switches panel, `Enter` inspects, `/` filters (by name, category, op type or layer), `t` cycles the sort order, `m` shows metrics, `p` pauses the view while capture continues, `?` shows help, `q` quits. Layouts: two panes at 110+ columns, a compact two-pane view at 80–109, and a single pane below 80 columns.

## Save and replay

```bash
./build/debug/llm-trace --demo --output demo.jsonl   # saves on exit; press s to save at any time
./build/debug/llm-trace --open demo.jsonl            # offline replay, no model or adapter needed
```

Traces are versioned JSON Lines ([format](docs/architecture/event-schema.md)). Files are written to a temporary file and then renamed, so an interrupted save never truncates an existing trace. Imported files are validated against size and structure limits. A malformed file is rejected with its line number, e.g. `cannot load bad.jsonl: line 3: not valid JSON`. A replayed synthetic trace is still labelled `[DEMO]`. Nothing is uploaded: files are only read from and written to the paths you give.

## License

MIT — see [LICENSE](LICENSE).
