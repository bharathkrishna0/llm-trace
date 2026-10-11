# ONNX Runtime profile adapter

`llm-trace --open <profile>.json` reads the profile ONNX Runtime writes when `Ort::SessionOptions::EnableProfiling(prefix)` is set. It is tested against **ONNX Runtime 1.20.1** (CPU execution provider, Linux x64). Other versions may work, but they are not tested. The profile file itself does not record the runtime version.

## What is read

| Profile record | Becomes |
|---|---|
| `cat: "Node"`, name ends in `_kernel_time` | one `operation` event. `name` is the node name with the suffix removed. `op_type` comes from `args.op_name`, `device` from `args.provider`, and the inputs/outputs from `input_type_shape`/`output_type_shape` (dtype + shape, plus element and byte counts when the shape is static) |
| `cat: "Session"` (`model_loading_uri`, `session_initialization`, `model_run`, `SequentialExecutor::Execute`) | `operation` events with category `session`. These are listed, but excluded from operation totals |
| `_fence_before` / `_fence_after` | omitted; the count is shown as a note |
| anything else (non-`X` phases, other categories) | skipped with a warning |

Timestamps and durations are in microseconds in the profile and are converted to nanoseconds. The raw record name is kept as attribute `raw_name`. `node_index`, `parameter_size`, `activation_size` and `output_size` are kept as attributes too. Each kernel gets a `run_index` attribute for the `model_run` it falls inside.

Category is assigned from a fixed op-type table (`category_for_onnx_op`). For example, `Softmax` is `activation`, not `attention`: the table does not guess model structure. A layer number is filled in only when a node name contains `layers.N`, `layer.N` or `h.N`, and a note says so.

## What is not available

ONNX Runtime profiles contain no tensor values. Attention weights, activation statistics and token positions are reported as **Not exposed by adapter**. The adapter's capabilities are `operation_timing` and `tensor_metadata` only.

## Producing a profile

```bash
python scripts/make_tiny_model.py examples/onnx/tiny_transformer.onnx   # needs: pip install onnx numpy
cmake -S . -B build/ort -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLM_TRACE_BUILD_ORT_RUNNER=ON -DONNXRUNTIME_ROOT=$HOME/deps/onnxruntime-linux-x64-1.20.1
cmake --build build/ort --target onnx_profile_runner
build/ort/onnx_profile_runner examples/onnx/tiny_transformer.onnx /tmp/tiny --runs 200 --warmup 20
llm-trace --open /tmp/tiny_<timestamp>.json
```

The ONNX Runtime package comes from the [v1.20.1 GitHub release](https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1). The tiny model is a 2-layer, 32-wide transformer-shaped graph with seeded random weights. It is not a trained model. The checked-in fixture `tests/fixtures/onnx_profiles/tiny_transformer_ort-1.20.1.json` was written by this runner with `--runs 2 --warmup 0`.

## Measured profiling overhead

The runner times the same workload in two sessions, one with profiling off and one with it on. Results on this tiny model, ORT 1.20.1, 1 intra-op thread, seq 16, 20 warm-up and 200 timed runs, Linux VM (8 vCPU, Intel Xeon Platinum 8559C), three repeats:

| | median per run |
|---|---|
| profiling off | 0.027 ms |
| profiling on | 0.184–0.187 ms |
| overhead | +577% to +589% |

The model's 44 kernels take about 27 µs in total. That makes ONNX Runtime's fixed per-node profiling cost dominate, so this is close to a worst case. It is **not** an estimate for real models, which have far longer kernels. I have not measured large models. Run the runner on your own model to get a number that applies to it.
