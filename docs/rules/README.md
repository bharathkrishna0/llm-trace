# Heuristic anomaly rules

A rule flags events that look unusual, such as "this call took 17× longer than this op normally does". **A match is a heuristic, not a measured finding.** It does not say why the op was slow, or whether the slowness matters. The TUI and `--check` label every match this way.

```bash
llm-trace --open trace.json [--rules rules.json]   # flagged rows show "!", press n to step through them
llm-trace --demo [--rules rules.json]
llm-trace --check trace.json [--rules rules.json] [--top N]
```

`--check` prints the rules and the flagged events. It exits with **0** when nothing matched, **1** when something matched, and **2** on an error: the trace or rules could not be read, or the rules are invalid. You can use it in CI as a regression tripwire.

## Default rules

Without `--rules`, two rules apply:

| id | rule |
| --- | --- |
| `slow-vs-p95` | a call longer than 3× the p95 of the **same op name with the same input shapes**. Groups need 20+ timed calls, and calls under 10 µs are ignored |
| `large-share` | one call longer than 25% of the summed operation time |

Why these choices:

- **Same name and same input shapes.** A 16-token prefill MatMul is different work from a 1-token decode MatMul on the same node. On the default demo trace, grouping by op type flagged 114 calls and grouping by name alone flagged 51, mostly the first, 16-token call of each node. Adding input shapes brought that to 0. If the source records no shapes, the group is the name alone.
- **20 calls.** With fewer calls, a nearest-rank p95 is close to the maximum, so the rule would rarely fire and would mean little when it did.
- **10 µs floor.** ONNX Runtime profiles record whole microseconds. A 1 µs op with a p95 of 1 µs would be flagged at 4 µs, which is timer noise.

Pass a rules file with `"rules": []` to turn all rules off.

## Rules file

```json
{
  "rules": [
    {"id": "slow-vs-p95", "kind": "p95_multiple", "factor": 3, "min_calls": 20, "min_us": 10},
    {"id": "large-share", "kind": "share", "above": 0.25},
    {"id": "slow-attention-matmul", "kind": "match", "filter": "op:MatMul dur>200us",
     "scope": "cat:attention"}
  ]
}
```

This file is [`examples/rules/example.json`](../../examples/rules/example.json).

| kind | fields | flags an event when |
| --- | --- | --- |
| `p95_multiple` | `factor` (1–1000, default 3), `min_calls` (default 20), `group` (`name` default, or `op` for op type), `min_us` (default 10) | duration > factor × p95 of its group |
| `share` | `above` (0–1, default 0.25) | duration / summed op time > above |
| `match` | `filter` (required) | it matches the [filter syntax](../../README.md#filtering), e.g. `dur>5ms`, `layer:0 op:MatMul` |

Every rule needs a unique `id` and may have a `scope` filter, which limits the events the rule looks at, including the events used to compute the p95 and the share. Unknown fields, unknown kinds, out-of-range numbers, duplicate ids and filters that do not parse are rejected with the rule's number and id. A file may hold at most 64 rules and be at most 1 MiB.

## What the rules use

- `p95_multiple` and `share` only use timed operations: type `operation`, category other than `session`, with a duration. Operations without a duration are skipped and counted, never treated as zero. Run records such as `model_run` are not compared with kernels.
- The p95 is nearest-rank over the events the trace holds. If the ring buffer overwrote events before saving, `--check` says so: the statistics cover only the retained events.
- Synthetic demo traces get a note that their timings come from a cost model.
- In the live demo, rules are re-evaluated at most once per second, because a full pass over a 65,536-event buffer takes about 10 ms. Replays and `--check` evaluate once.

## Example

These are two real ONNX Runtime profiles of `examples/onnx/tiny_transformer.onnx`, 200 runs plus 20 warm-up runs each, on this VM:

```text
$ llm-trace --check t1.json --top 3
...
Flagged events: 29 (showing first 3)
  #1010  /layers.1/mlp/up_proj/MatMul  layer 1  52.0 µs
      slow-vs-p95: 52.0 µs = 17.3x p95 of /layers.1/mlp/up_proj/MatMul [1x16x32] (3.0 µs, 220 calls)
  #1038  /layers.1/input_layernorm/LayerNormalization  layer 1  14.0 µs
      slow-vs-p95: 14.0 µs = 14.0x p95 of /layers.1/input_layernorm/LayerNormalization [1x16x32, 32, 32] (1.0 µs, 220 calls)
  #1352  /layers.0/attn/context/Reshape  layer 0  1.50 ms
      slow-vs-p95: 1.50 ms = 1495.0x p95 of /layers.0/attn/context/Reshape [1x16x4x8, 3] (1.0 µs, 220 calls)
```

These are single slow calls on a tiny model, most likely scheduling or cache effects on a shared VM. The rule cannot tell which. The 4-thread profile had 62 flagged calls; the 1-thread profile had 29. That is one pair of runs, not a repeated measurement.

## Limits

- Rules only see what the trace recorded. Thread count, batch size and other unrecorded settings cannot be grouped on.
- p95 is per trace. A trace where every call is slow flags nothing; use `--compare` against a baseline for that.
- No rule adjusts for the number of comparisons. A long trace will contain some calls above 3× p95 by chance.
