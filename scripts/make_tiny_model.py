#!/usr/bin/env python3
"""Generate a tiny 2-layer transformer-style ONNX model for profiling tests.

The weights are seeded random numbers, so the model is ours to redistribute (MIT).
It exists only to give ONNX Runtime a realistic mix of operators to profile; it is
not a trained language model.

Usage: python scripts/make_tiny_model.py examples/onnx/tiny_transformer.onnx
Requires: pip install onnx numpy
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

D_MODEL = 32
HEADS = 4
HEAD_DIM = D_MODEL // HEADS
FFN = 4 * D_MODEL
LAYERS = 2
OPSET = 17


def main(path: str) -> None:
    rng = np.random.default_rng(1234)
    nodes, inits = [], []

    def weight(name, shape, scale=0.05):
        inits.append(numpy_helper.from_array((rng.standard_normal(shape) * scale).astype(np.float32), name))
        return name

    def const(name, values, dtype=np.int64):
        inits.append(numpy_helper.from_array(np.array(values, dtype=dtype), name))
        return name

    shape_heads = const("shape_heads", [0, -1, HEADS, HEAD_DIM])
    shape_merge = const("shape_merge", [0, -1, D_MODEL])
    scale = const("attn_scale", [np.sqrt(HEAD_DIM)], np.float32)

    x = "input"
    for layer in range(LAYERS):
        p = f"/layers.{layer}"

        def n(op, inputs, outputs, name, **attrs):
            nodes.append(helper.make_node(op, inputs, outputs, name=f"{p}/{name}", **attrs))
            return outputs[0]

        ln1 = n("LayerNormalization", [x, weight(f"{p}.ln1.g", [D_MODEL], 1.0), weight(f"{p}.ln1.b", [D_MODEL])],
                [f"{p}/ln1_out"], "input_layernorm/LayerNormalization", axis=-1)
        heads = {}
        for proj in ("q", "k", "v"):
            mm = n("MatMul", [ln1, weight(f"{p}.{proj}.w", [D_MODEL, D_MODEL])], [f"{p}/{proj}_mm"],
                   f"attn/{proj}_proj/MatMul")
            rs = n("Reshape", [mm, shape_heads], [f"{p}/{proj}_rs"], f"attn/{proj}_proj/Reshape")
            perm = [0, 2, 3, 1] if proj == "k" else [0, 2, 1, 3]
            heads[proj] = n("Transpose", [rs], [f"{p}/{proj}_t"], f"attn/{proj}_proj/Transpose", perm=perm)
        scores = n("MatMul", [heads["q"], heads["k"]], [f"{p}/scores"], "attn/scores/MatMul")
        scaled = n("Div", [scores, scale], [f"{p}/scaled"], "attn/scores/Div")
        probs = n("Softmax", [scaled], [f"{p}/probs"], "attn/Softmax", axis=-1)
        ctx = n("MatMul", [probs, heads["v"]], [f"{p}/ctx"], "attn/context/MatMul")
        ctx_t = n("Transpose", [ctx], [f"{p}/ctx_t"], "attn/context/Transpose", perm=[0, 2, 1, 3])
        ctx_m = n("Reshape", [ctx_t, shape_merge], [f"{p}/ctx_m"], "attn/context/Reshape")
        attn_out = n("MatMul", [ctx_m, weight(f"{p}.o.w", [D_MODEL, D_MODEL])], [f"{p}/attn_out"],
                     "attn/o_proj/MatMul")
        res1 = n("Add", [x, attn_out], [f"{p}/res1"], "attn/residual/Add")
        ln2 = n("LayerNormalization", [res1, weight(f"{p}.ln2.g", [D_MODEL], 1.0), weight(f"{p}.ln2.b", [D_MODEL])],
                [f"{p}/ln2_out"], "post_attention_layernorm/LayerNormalization", axis=-1)
        up = n("MatMul", [ln2, weight(f"{p}.up.w", [D_MODEL, FFN])], [f"{p}/up"], "mlp/up_proj/MatMul")
        act = n("Relu", [up], [f"{p}/act"], "mlp/act/Relu")
        down = n("MatMul", [act, weight(f"{p}.down.w", [FFN, D_MODEL])], [f"{p}/down"], "mlp/down_proj/MatMul")
        x = n("Add", [res1, down], [f"{p}/out"], "mlp/residual/Add")

    nodes.append(helper.make_node("Identity", [x], ["output"], name="/output/Identity"))
    graph = helper.make_graph(
        nodes, "tiny_transformer",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", "seq", D_MODEL])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", "seq", D_MODEL])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="llm-trace/scripts/make_tiny_model.py")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"wrote {path}: {LAYERS} layers, d_model={D_MODEL}, {len(nodes)} nodes")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/onnx/tiny_transformer.onnx")
