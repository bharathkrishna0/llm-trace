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

ONNX Runtime profiles contain no tensor values. Attention weights, activation statistics and token positions are reported as **Not exposed by adapter**. The adapter's capabilities are `operation_timing` and `tensor_metadata` only. Activation statistics can be added with the opt-in runner mode below; attention weights cannot.

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

## Real activation summaries (opt-in)

ONNX Runtime returns only graph outputs, so intermediate tensors must first be made outputs. `scripts/expose_activations.py` copies a model and does that. It never modifies the input file:

```bash
python scripts/expose_activations.py model.onnx model.act.onnx --op MatMul --op Relu   # or --name-regex mlp; default: every float32 node output
build/ort/onnx_profile_runner model.act.onnx /tmp/prof --runs 200 --warmup 20 --activations trace.jsonl [--epsilon 1e-6]
llm-trace --open trace.jsonl
```

The script records which node produced each exposed tensor in the model's metadata (`llm_trace.exposed_activations`). After every `Run` of the profiled session (warm-up runs included, matching the profile's `model_run` records), the runner reads those outputs and computes, over every element:

- mean, min, max and population standard deviation of the finite values;
- the count of NaN and ±Inf values;
- the share of finite values with |x| < ε (default 1e-6).

It then loads the profile with this adapter and attaches each summary to the kernel event with the same node name and run (`run_index`). The result is written as llm-trace JSONL with the `activation_summaries` capability set, and the tensor name is kept as attribute `activation_tensor`. Only nodes that were exposed get a summary. All other events, and attention weights, stay **Not exposed by adapter**. If a node was fused away so that no kernel carries its name, the record is dropped and counted in a warning. The input is a fixed synthetic pattern, so these are activations for that input, not for real text.

**Checked against ONNX Runtime's Python API.** I exposed all 46 float outputs of the tiny model and compared the runner's summaries with NumPy statistics of the same outputs from `onnxruntime` 1.20.1 (Python) on the same input. The worst relative difference was 2e-6, which is float rounding.

**Cost**, on the tiny model with the same setup as below, three alternating runs each:

| | median per run, profiling off |
|---|---|
| original model | 0.0307–0.0309 ms |
| all 46 node outputs exposed and fetched | 0.0415–0.0423 ms (about +35 %) |
| computing the 46 summaries (outside the timed region) | about 0.069 ms per run |

Both costs grow with tensor size: the statistics read every element once more. Expose only the nodes you need. Kernel timings in the activation trace come from the modified model, so compare them with a trace of the same modified model, not with the original.

## Measured profiling overhead

The runner times the same workload in two sessions, one with profiling off and one with it on. Results on this tiny model, ORT 1.20.1, 1 intra-op thread, seq 16, 20 warm-up and 200 timed runs, Linux VM (8 vCPU, Intel Xeon Platinum 8559C), three repeats:

| | median per run |
|---|---|
| profiling off | 0.027 ms |
| profiling on | 0.184–0.187 ms |
| overhead | +577% to +589% |

**Re-measured in M9.** The same command now gives 0.0307–0.0310 ms with profiling off and 0.367–0.434 ms with it on (+1,050 % to +1,170 %, three repeats). I don't know why profiling costs more than in the M4 runs above. While working on M9 I found and fixed a bug in the runner's input. `(i % 17) - 8` was computed unsigned, so about half of the input values were around 1.8e17 instead of −0.08…0. Old and fixed inputs measured alternately gave the same timings (+1,080/+1,169/+1,108 % vs +1,055/+1,124/+1,085 %), so the bug does not explain the change. The checked-in profile fixture was written with the old input. Its timings are still real ONNX Runtime timings, but they are for that input.

The model's 44 kernels take about 27 µs in total. That makes ONNX Runtime's fixed per-node profiling cost dominate, so this is close to a worst case. It is **not** an estimate for real models, which have far longer kernels. I have not measured large models. Run the runner on your own model to get a number that applies to it.
