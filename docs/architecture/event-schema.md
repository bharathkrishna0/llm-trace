# Trace file format (schema version 1)

`llm-trace` saves traces as JSON Lines: one JSON object per line, UTF-8, `\n`-terminated. The first record must be `metadata`. Every record after it is an `event`, in increasing `sequence` order.

Times are integer nanoseconds. `timestamp_ns` is a monotonic offset from the start of the run, and `duration_ns` is inclusive (it covers nested child work). An optional field that is **absent** means "not present in this trace". `0` is always a real value.

## `metadata` record

| field | type | notes |
|---|---|---|
| `record` | `"metadata"` | required, first line |
| `schema_version` | integer | required; files newer than this build are rejected |
| `run_id`, `adapter`, `runtime`, `runtime_version`, `model` | string | |
| `created_utc` | string | wall-clock time of the save, for display only; never used to compute durations |
| `source` | `none` \| `synthetic` \| `onnx_profile` \| `replay` | where the events came from when saved |
| `origin` | same values | original source. Replayed synthetic traces keep `origin: synthetic`, so they are still labelled DEMO |
| `capabilities` | object of booleans | `operation_timing`, `tensor_metadata`, `layer_hierarchy`, `token_index`, `activation_summaries`, `attention_values` |
| `buffer_capacity`, `overwritten_events`, `rejected_events` | unsigned integer | ring-buffer state at save time. `overwritten_events > 0` means the oldest events are missing |
| `notes` | string array | adapter and save notes |

## `event` record

| field | type | required |
|---|---|---|
| `record` | `"event"` | yes |
| `sequence` | unsigned integer, strictly increasing | yes |
| `timestamp_ns` | unsigned integer | yes |
| `type` | `run_start` `run_end` `operation` `tensor_metadata` `activation_summary` `attention_summary` `warning` `adapter_error` `dropped_events` | yes |
| `name` | non-empty string | yes |
| `category` | e.g. `attention`, `mlp`, `normalization`, ... (unknown values load as `unknown` with a warning) | no |
| `duration_ns` | unsigned integer | no |
| `op_type`, `parent_id`, `device`, `message` | string | no |
| `layer_index`, `token_index`, `head_index`, `thread_id` | integer | no |
| `inputs`, `outputs` | array of tensors: `{name, shape?, dtype?, element_count?, byte_count?}`; `-1` in `shape` marks a dynamic dimension | no |
| `activation` | `{mean, min, max, stddev, nonfinite_count, sparsity, sparsity_epsilon, sampled_elements, full_tensor}` | no; no current adapter writes it |
| `attributes` | array of `[key, value]` string pairs, order and duplicates preserved (e.g. raw runtime names) | no |

Records with an unknown `record` value are skipped with a warning.

## Reading untrusted files

Trace files are treated as untrusted input. These limits are checked before data is allocated (defaults in `ReadLimits`, `include/llm_trace/trace_io.hpp`):

| limit | default |
|---|---|
| file size | 1 GiB |
| line length | 1 MiB |
| events | 2,000,000 |
| string length | 4 KiB |
| tensors per event | 64 |
| tensor rank | 16 |
| attributes / notes | 64 |
| JSON nesting depth | 16 |

Any type error, missing required field, invalid value, or limit violation stops the import with `line N: <reason>`. Nothing is partially loaded.

## Writing

`write_jsonl(path, ...)` writes to `<path>.tmp`, flushes it, and only then renames it over `<path>`. If the write fails, an existing file at `<path>` is left untouched and the temporary file is removed. Non-finite activation statistics are rejected rather than written as `null`.
