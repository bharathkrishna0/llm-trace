# ADR 0002: Fixed-capacity ring buffer that overwrites the oldest event

Status: accepted (M1)

**Context.** Tracing must not grow memory without bound or stall the producer when the viewer falls behind.

**Decision.**
- `RingBuffer<TraceEvent>` allocates its capacity up front. When it is full, it overwrites the oldest event and increments an `overwritten` counter.
- `Collector` protects it with a single mutex and counts rejected (invalid) events separately.
- Capacity, retained, accepted, overwritten and rejected counts are always visible in the status strip and saved with the trace.

**Consequences.**
- Long runs keep only the most recent window, and the UI says so.
- Saving writes only the retained events, and the metadata records how many were lost.
- One mutex is simple and adequate at the measured rates (see `docs/benchmarks/`). A lock-free queue is deferred until a runtime needs it.

**Update (M6).** The design still uses one mutex, with less work done while holding it. Validation happens before the lock. The overwritten event is moved out of the ring and destroyed after the lock is released. `submit_batch()` stores many events under a single lock acquisition. Viewers call `delta_since(sequence)`, which copies only the new events under the lock instead of the whole buffer. Sequences stay consecutive and every accepted event is pushed, so the retained window is always `[next - size, next)`.
