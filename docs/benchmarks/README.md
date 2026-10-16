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

## Results (2026-10-11, M6)

`main` (M5) and this branch were built in Release and run alternately on the same VM, twice each (main, M6, main, M6). The table shows both runs. Differences under about 10 % are within run-to-run noise on this VM.

| Case | main (M5) | M6 |
|---|---|---|
| `RingBuffer::push` (copy, capacity 65,536) | 179 / 190 ns | 197 / 197 ns |
| `Collector::submit`, 1 thread | 203 / 207 ns | 236 / 218 ns |
| `Collector::submit`, 4 threads contending (total, per event) | 1,290 / 1,284 ns | 861 / 854 ns |
| `Collector::submit_batch` (64 per call), 4 threads (total, per event) | n/a | 742 / 639 ns |
| `write_jsonl` to memory | 6.57 / 6.09 µs | 6.18 / 6.18 µs |
| `read_jsonl` from memory (all validation on) | 12.1 / 11.0 µs | 11.2 / 11.4 µs |
| ONNX Runtime profile parse (fixture) | 11.3 / 9.9 µs | 10.0 / 10.0 µs |

| Case | main (M5) | M6 |
|---|---|---|
| Open + render one 160×45 replay frame, 10,000 events | 4.65 / 4.30 ms | 4.47 / 4.43 ms |
| Open + render one 160×45 replay frame, 65,536 events | 29.1 / 27.4 ms | 28.3 / 28.4 ms |
| Live refresh, full 65,536 buffer + 1,000 new events: snapshot + new session (the M5 path) | — | 30.5 / 34.5 ms |
| Live refresh, same workload: `delta_since` + `TraceSession::apply` (the M6 path) | — | 2.99 / 2.95 ms |

The two live-refresh rows run in the same M6 binary, so they compare the old and new refresh paths on identical data. The M5 binary has no such case.

Memory: `sizeof(TraceEvent)` is 448 bytes, plus heap for names and tensor shapes. A saved event is about 503 bytes of JSONL.

## Heuristic rules (2026-10-13, M8)

The M8 binary was compared with `main` (M7) in the same way: alternately, twice each. This VM was noisy during these runs. main's first run was about 2× slower than its second on every frame case, so the table shows both runs.

| Case | main (M7) | M8 |
|---|---|---|
| Open + render one 160×45 replay frame, 65,536 events | 55.0 / 27.1 ms | 40.4 / 37.1 ms |
| Live refresh, `delta_since` path, 65,536 + 1,000 events, rules off | 6.0 / 2.8 ms | 2.8 / 2.9 ms |
| Live refresh, same, default rules on (throttled) | — | 2.9 / 2.8 ms |
| One pass of the default rules over 65,536 events | — | 9.9 / 10.3 ms |

- **One pass of the rules costs about 10 ms** at 65,536 events. Most of it is grouping every timed op by a hash of its name and input shapes, then a partial sort (`nth_element`) per group for the p95. A first version that built a key string per event took about 30 ms.
- **Opening a replay** now runs that pass once, so the open-and-draw case rose from about 27 ms to about 37–40 ms (comparing with main's quieter run).
- **Live view.** The TUI re-evaluates rules at most once per second. The live-refresh rows above therefore mostly measure refreshes that skip the pass. A refresh that includes the pass costs about 10 ms more, so about 13 ms; before throttling, every refresh measured 12.8 ms. Events that arrived since the last pass are not flagged until the next one, up to about 1 s later.

## What this means, and what it does not

- Storing an event costs about 0.2 µs, mostly the copy of its strings and tensors. That is small next to a transformer kernel lasting microseconds to milliseconds. However, it is **not** a measured end-to-end overhead on a real model, because live runtime capture is not implemented yet.
- **Contention (M6).** With 4 threads, the per-event cost dropped from about 1.29 µs to about 0.86 µs (1.5×). The gain comes from moving the overwritten event out of the ring and destroying it after the lock is released. `submit_batch` takes the lock once per batch and reaches about 0.64–0.74 µs. 4 threads are still slower per event than 1 thread: there is still one mutex. A lock-free queue remains deferred (ADR 0002).
- **Single-thread cost went up slightly** (about 205 → 225 ns per submit, ring push about 185 → 197 ns). This is the extra move into the "evicted" slot. It is close to the noise level, but it shows up in both runs, so it is reported.
- **Live refresh (M6).** In M5, every refresh in a live view copied the whole buffer and rebuilt the session, about 30–35 ms with a full buffer. Now the viewer asks the collector only for events newer than the last one it saw (`delta_since`). It drops overwritten events from the front and appends the new ones. The same refresh takes about 3 ms (about 10×). Aggregates are still recomputed over all retained events, which is most of the remaining time.
- **Replay frame.** The "open + render" case builds a session from a loaded trace and then draws one frame. It costs the same as before (about 28 ms at 65k events). In the interactive TUI this happens once when a file is opened. Moving the selection or redrawing does not rebuild the view; filtering and sorting re-run the query once per key press.
- Saving and loading are much slower than capture (JSON formatting and validation). They are user-triggered, never on the capture path.
