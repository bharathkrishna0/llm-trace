#!/usr/bin/env python3
"""Copy an ONNX model and expose chosen intermediate tensors as extra graph outputs.

onnx_profile_runner --activations reads those outputs after every run and stores
mean/min/max/std, NaN/Inf count and near-zero share per node in an llm-trace trace.
The input model is not modified. Exposing a tensor can stop ONNX Runtime from fusing
the node that produces it, so timings from the copy may differ from the original.

Usage:
  python scripts/expose_activations.py in.onnx out.onnx [--op MatMul --op Relu]
                                                       [--name-regex 'mlp']
Without --op/--name-regex, the first output of every node whose output is float32
is exposed. Requires: pip install onnx
"""

import argparse
import json
import os
import re
import sys

import onnx
from onnx import TensorProto, helper

METADATA_KEY = "llm_trace.exposed_activations"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--op", action="append", default=[], help="op type to expose (repeatable)")
    ap.add_argument("--name-regex", help="expose nodes whose name matches this regex")
    args = ap.parse_args()

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("refusing to overwrite the input model; choose another output path", file=sys.stderr)
        return 2

    model = onnx.load(args.input)
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {vi.name: vi for vi in list(inferred.graph.value_info) + list(inferred.graph.output)}
    graph = model.graph
    existing = {o.name for o in graph.output}
    pattern = re.compile(args.name_regex) if args.name_regex else None

    exposed = {}
    skipped_type = 0
    for node in graph.node:
        if not node.output or not node.output[0] or not node.name:
            continue
        if args.op and node.op_type not in args.op:
            continue
        if pattern and not pattern.search(node.name):
            continue
        tensor = node.output[0]
        if tensor in existing:
            continue
        vi = types.get(tensor)
        if vi is None or vi.type.tensor_type.elem_type != TensorProto.FLOAT:
            skipped_type += 1
            continue
        graph.output.append(vi)
        exposed[tensor] = node.name

    if not exposed:
        print("no float32 node outputs matched; nothing exposed", file=sys.stderr)
        return 1
    for prop in list(model.metadata_props):
        if prop.key == METADATA_KEY:
            model.metadata_props.remove(prop)
    helper.set_model_props(
        model,
        {**{p.key: p.value for p in model.metadata_props}, METADATA_KEY: json.dumps(exposed)},
    )
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"exposed {len(exposed)} tensor(s); skipped {skipped_type} non-float32 output(s)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
