# Contributing

## Build and test

```bash
cmake --preset debug && cmake --build --preset debug --parallel && ctest --preset debug
```

On Windows, use `cmake --preset msvc`, `cmake --build --preset msvc-release` and `ctest --preset msvc-release`. CI builds with GCC, Clang and MSVC, and treats warnings as errors.

## Format

CI checks every tracked `*.cpp`/`*.hpp` with clang-format **18.1.8** (`pip install clang-format==18.1.8`):

```bash
clang-format -i $(git ls-files '*.cpp' '*.hpp')
```

## Benchmarks

```bash
cmake --preset release && cmake --build --preset release --target llm_trace_bench
./build/release/llm_trace_bench
```

See [docs/benchmarks/](docs/benchmarks/) for the method. Report numbers together with the machine and build type, and never quote a number that was estimated rather than measured.

## Pull requests

- One purpose per PR, branched from `main`. Add a test for every core bug fix.
- Keep runtime-specific code in `src/adapters/`. The core must not depend on FTXUI or any runtime.
- Do not add data the source does not report. Use `std::nullopt` and set the capability flag to false.
- Record significant design decisions as a short ADR in `docs/architecture/decisions/`.
