# Architecture overview

```
 runtime / generator            core (llm_trace_core)                 presentation
 ───────────────────            ─────────────────────                 ────────────
 SyntheticGenerator ──submit──▶ Collector ──▶ RingBuffer<TraceEvent>
                                   │ snapshot()                       TUI (llm_trace_tui, FTXUI)
 ORT profile file ──OnnxProfileAdapter──▶ LoadedTrace ──▶ TraceSession ──▶ Controller ─▶ components
 JSONL file ──────read_jsonl───────────▶ LoadedTrace ─┘   (filter, sort,     (keys → actions,
                                                            aggregate)         pure render)
 TraceSession ◀── write_jsonl (s / --output, temp + rename)
```

| Component | Responsibility | Does not |
|---|---|---|
| `event.hpp` | `TraceEvent`, `TensorMetadata`, `AdapterCapabilities`, `TraceMetadata`. Optional fields use `std::optional`: absent means unknown, zero is a real value | know about any runtime |
| `RingBuffer<T>` | fixed capacity, set at construction; overwrites the oldest event and counts overwrites | grow, or allocate after construction |
| `Collector` | thread-safe `submit()` / `submit_batch()` (one mutex; validation and destruction of overwritten events happen outside it), rejection and overwrite counters, `snapshot()`, incremental `delta_since(sequence)` | render or do I/O |
| `compare.hpp` | `compare_traces(A, B)`: per-category, per-layer and per-op mean-per-call deltas, one-sided ops, mismatch warnings; text and JSON output | decide whether a change is significant |
| `TraceSession` | a list of events plus queries: filter (`EventFilter`), sort, per-op/per-category aggregates; `apply(delta)` drops overwritten events and appends new ones for live views | mutate events |
| `SyntheticGenerator` | deterministic, seeded demo trace from a fixed cost model, labelled `synthetic` | claim to measure anything |
| `OnnxProfileAdapter` | maps an ONNX Runtime profile file to events; declares its capabilities | link ONNX Runtime (it parses the file only) |
| `trace_io` | versioned JSONL write (atomic) and bounded read; `load_trace` detects the format | trust input |
| `tui/` | key bindings → `Action`, `Controller` state, pure frame rendering, wide/medium/narrow layouts | block the producer: the demo producer runs on its own thread, and the UI only takes snapshots |

Data flow rules:
- The core never depends on FTXUI or ONNX Runtime. The TUI reads only `TraceSession` and `CollectorStats`.
- Each data source sets `TraceMetadata::source`, `origin` and `capabilities`. The UI shows "Not exposed by adapter" for any capability that is false. It does not show blanks or zeros.
- The application reads and writes only the paths given to it on the command line. It makes no network connections.

Decisions are recorded in [decisions/](decisions/).
