# Comparing two traces

```bash
llm-trace --compare A B [--compare-by name|op] [--top N] [--compare-json FILE|-] [--ascii]
```

`A` and `B` can each be an llm-trace JSONL file or an ONNX Runtime profile. Anything `--open` reads works here. The report shows B against A:

- **By category** and **By layer**: every category and layer, in a fixed order.
- **By operation**: ops that occur in both traces, with the largest absolute change in mean time first (`--top`, default 20).
- **Only in A / Only in B**: ops that appear in one trace only, largest total first.

`--compare-json FILE` also writes the comparison as JSON (`"schema": "llm-trace-compare/1"`, durations in ns). The JSON includes count, total, mean, median and share per row. `--compare-json -` prints only the JSON.

## What is compared

- **Timed operations only**: event type `operation`, category other than `session`, with a duration. Operations without a duration are counted and reported in a warning, never treated as zero. Run-level records such as ONNX Runtime's `model_run` are excluded, as they are in the TUI's aggregates.
- **Mean time per call.** If B has twice as many decode steps or runs, B's totals are twice as large, but the rows do not show that as a slowdown. `change` is `(B mean − A mean) / A mean`.
- **Matching is by key only.**
  - `--compare-by name` (default) matches the event name, which is the node name for ONNX Runtime. This is exact for two runs of the same model.
  - `--compare-by op` matches the op type, falling back to the name when there is no op type. Use it across models or exports where node names differ.
  - llm-trace cannot tell whether two ops with the same key did the same work, for example with different input shapes.

## Warnings

The report warns, rather than refusing, when:

- the traces come from different sources (for example synthetic vs ONNX Runtime);
- they come from different models, runtimes/versions or devices;
- a trace is synthetic (its timings come from a cost model);
- a trace lost events to ring-buffer overwrites or had events rejected;
- a trace has operations without a duration, or no timed operations at all.

It cannot detect settings that the trace does not record, such as ONNX Runtime thread counts, optimisation level, CPU frequency or machine load. Note these yourself.

## Statistics

No significance testing is done. Rows with fewer than 5 calls on either side are marked `*`. Repeat runs before trusting a change: the JSON has medians, which are less sensitive to outliers than the means shown in the text table.

## Example (measured)

Two profiles of the checked-in tiny model, made with the M4 runner on an 8 vCPU Linux VM (Release, ONNX Runtime 1.20.1, 200 runs + 20 warm-up each). The only difference is `--threads 1` versus `--threads 4`:

```bash
build/ort/onnx_profile_runner examples/onnx/tiny_transformer.onnx /tmp/t1 --runs 200 --warmup 20 --threads 1
build/ort/onnx_profile_runner examples/onnx/tiny_transformer.onnx /tmp/t4 --runs 200 --warmup 20 --threads 4
llm-trace --compare /tmp/t1_*.json /tmp/t4_*.json --compare-by op --top 5 --ascii
```

```
By category
  category                                  A mean      B mean    change   A calls   B calls
  normalization                             1.1 us      8.3 us   +626.0%       880       880
  activation                                1.4 us      3.1 us   +120.0%       880       880
  matmul                                    1.7 us      4.0 us   +141.7%     3,520     3,520
  elementwise                               1.1 us      2.4 us   +113.1%       880       880
  shape                                     1.7 us      3.3 us    +88.9%     3,520     3,520

By op type (largest change in mean first)
  op type                                   A mean      B mean    change   A calls   B calls
  LayerNormalization                        1.1 us      8.3 us   +626.0%       880       880
  FusedMatMul                               2.9 us      6.3 us   +121.0%       440       440
  Transpose                                 1.5 us      4.3 us   +184.7%     1,760     1,760
  MatMul                                    1.5 us      3.7 us   +147.3%     3,080     3,080
  Relu                                      1.0 us      2.8 us   +175.7%       440       440
```

On this tiny model (kernels of 1–3 µs), 4 intra-op threads made every kernel slower. Thread hand-off costs more than the work it splits. Expect a different result on a large model.

The report printed "No warnings", because both profiles have the same source, model and runtime. The thread count is not in the profile, so llm-trace could not flag it. This is the kind of difference you have to track yourself. This is one pair of runs; it was not repeated.
