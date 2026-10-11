# Benchmarks

There are two kinds of cost:

1. **llm-trace's own cost**: storing, saving, loading and drawing events. `bench/llm_trace_bench.cpp` measures it.
2. **The runtime's profiling cost**: what ONNX Runtime adds when profiling is on. `examples/onnx/onnx_profile_runner` measures it; see [../adapters/onnxruntime.md](../adapters/onnxruntime.md#measured-profiling-overhead).

Both numbers below come from runs, not estimates. They describe this machine only.

## Method

```bash
cmake --preset release && cmake --build --preset release --target llm_trace_bench
./build/release/llm_trace_bench            # --events N (default 200000), --reps N (default 7)
```

- **Workload**: 200,000 events copied from the deterministic synthetic trace (12 layers; each event has 1–2 tensors with shape, dtype and counts). The same events are used for every case.
- **Timing**: each case runs 7 times. The median wall-clock time (`steady_clock`) is divided by the number of events. There are no warm-up exclusions beyond the median.
- **Machine**: Linux VM, 8 vCPU Intel Xeon Platinum 8559C, GCC 11, `-O3` (CMake Release).

## Results (2026-10-11)

| Case | Per event | Throughput |
|---|---|---|
| `RingBuffer::push` (copy, capacity 65,536) | 168 ns | 5.9 M events/s |
| `Collector::submit`, 1 thread | 192 ns | 5.2 M events/s |
| `Collector::submit`, 4 threads contending (total) | 1,184 ns | 0.84 M events/s |
| `write_jsonl` to memory | 5.96 µs | 168 k events/s |
| `read_jsonl` from memory (all validation on) | 11.1 µs | 90 k events/s |
| ONNX Runtime profile parse (fixture) | 10.0 µs | 100 k events/s |

| Case | Per frame |
|---|---|
| Render a 160×45 frame, 10,000 events | 4.2 ms |
| Render a 160×45 frame, 65,536 events (full default buffer) | 26.2 ms |

Memory: `sizeof(TraceEvent)` is 448 bytes, plus heap for names and tensor shapes. A saved event is about 503 bytes of JSONL.

## What this means, and what it does not

- Storing an event costs about 0.2 µs, mostly the copy of its strings and tensors. That is small next to a transformer kernel lasting microseconds to milliseconds. However, it is **not** a measured end-to-end overhead on a real model, because live runtime capture is not implemented yet.
- The single `Collector` mutex is the bottleneck under contention: 4 threads together are about 6× slower per event than 1 thread. This is fine for the current single-producer demo. A lock-free queue is the planned fix if a multi-threaded runtime adapter needs it (ADR 0002).
- Frame time grows with the number of retained events, because each frame rebuilds the filtered and sorted view. At 65k events it is about 26 ms, so redraws in a full buffer are visibly slower. Caching the view between frames is a known improvement.
- Saving and loading are much slower than capture (JSON formatting and validation). They are user-triggered, never on the capture path.
