# ADR 0003: Capability flags instead of fabricated data

Status: accepted (M1–M4)

**Context.** Most runtimes do not expose attention weights, activations or token positions. A tool that fills those fields with plausible values would mislead users.

**Decision.**
- Every source declares `AdapterCapabilities`. Optional fields stay `std::nullopt` unless the source reports them.
- The TUI shows "Not exposed by adapter" (a capability is false) or "Not present in this trace" (the capability is true but this event lacks the field).
- Synthetic data is always labelled `[DEMO]`, including after it is saved and replayed (`origin` is kept).

**Consequences.**
- The ONNX Runtime view shows timings and shapes only.
- Layer numbers that come from node names are marked as derived from names.
