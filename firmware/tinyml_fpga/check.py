# check.py - PC side: run a model through tinyml_fpga and through LiteRT (the TFLite interpreter)
# and compare every intermediate tensor. int8 tensors must match bit for bit; the final softmax is
# compared after dequantisation (TFLite uses an int8 lookup-table softmax, we use float).

import numpy as np

from .converter import Converter, convert_bytes
from .model import Model
from .engine import CpuEngine, HwEngine, SimBus
from . import kernels as kn


# Create a LiteRT interpreter that keeps all intermediate tensors.
def _interpreter(data):
    # Prefer the standalone LiteRT runtime.
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        # Fall back to the one inside TensorFlow.
        from tensorflow.lite.python.interpreter import Interpreter
    # Keep every tensor so intermediates can be compared.
    it = Interpreter(model_content=data, experimental_preserve_all_tensors=True)
    # Allocate buffers.
    it.allocate_tensors()
    # Ready.
    return it


# Compare tinyml_fpga with LiteRT on a set of int8 input samples. Returns a result dict.
def compare(tflite_path, samples, backend="cpu", verbose=True):
    # Source model.
    with open(tflite_path, "rb") as f:
        data = f.read()
    # Converter (kept for the tensor index map).
    conv = Converter(data)
    conv.run()
    # Our tensor id -> tflite tensor index.
    back = {v: k for k, v in conv.tmap.items()}
    # Our runtime on the chosen engine.
    eng = {"cpu": CpuEngine, "sim": lambda: HwEngine(SimBus())}[backend]() if isinstance(backend, str) else backend
    model = Model(convert_bytes(data), eng)
    # LiteRT.
    it = _interpreter(data)
    # Input index of LiteRT.
    inp = it.get_input_details()[0]
    # Statistics per op: [mismatching elements, max abs diff, elements].
    stats = [[0, 0, 0] for _ in model.ops]
    # Top-1 agreement count.
    agree = 0
    # Each sample.
    for s in samples:
        # int8 sample, shaped for LiteRT.
        s = np.asarray(s, dtype=np.int8).reshape(inp["shape"])
        # Reference run.
        it.set_tensor(inp["index"], s)
        it.invoke()
        # Our run, keeping intermediates.
        out = model.invoke(bytearray(s.tobytes()), keep=True)
        # Each op output.
        for i, op in enumerate(model.ops):
            # Reference tensor.
            ref = it.get_tensor(back[op["out"]]).reshape(-1)
            # Our tensor.
            ours = model.values[op["out"]]
            # Softmax / dequantize outputs are floats: compare in real units.
            if isinstance(ours, list):
                # Reference in real units.
                td = model.tensors[op["out"]]
                r = (ref.astype(np.float64) - td["zp"]) * td["scale"] if ref.dtype == np.int8 else ref.astype(np.float64)
                # Difference.
                d = np.abs(np.asarray(ours) - r)
                # Tolerance: one output step of the reference (or 1e-5 for float outputs).
                tol = td["scale"] if ref.dtype == np.int8 else 1e-5
                bad = int((d > tol * 1.001).sum())
            else:
                # int8 bytes vs int8 reference.
                d = np.abs(np.asarray(kn.to_i8_list(ours)) - ref.astype(np.int64))
                bad = int((d != 0).sum())
            # Accumulate.
            stats[i][0] += bad
            stats[i][1] = max(stats[i][1], float(d.max()) if d.size else 0)
            stats[i][2] += d.size
        # Top-1 of both.
        ref_out = it.get_tensor(it.get_output_details()[0]["index"]).reshape(-1)
        ours_out = out if isinstance(out, list) else kn.to_i8_list(out)
        agree += int(np.argmax(ref_out) == int(np.argmax(ours_out)))
    # Report.
    exact = all(st[0] == 0 for st in stats)
    if verbose:
        print("%s on %s: %d samples, top-1 agreement %d/%d, %s" % (tflite_path, model.engine.name, len(samples),
              agree, len(samples), "ALL TENSORS MATCH" if exact else "MISMATCHES"))
        for op, st in zip(model.ops, stats):
            print("  %-10s mismatches %6d / %-7d  max|diff| %g" % (op["op"], st[0], st[2], st[1]))
    # Machine-readable result.
    return {"exact": exact, "agree": agree, "n": len(samples), "per_op": stats}
