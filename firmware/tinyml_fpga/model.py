# model.py - loads a .tmf file and runs it. MicroPython-safe (no numpy, no typing, no dataclasses).
# Engine ops (conv, depthwise, fully connected) go to the engine (FPGA or CpuEngine); pooling,
# reshape, softmax, relu, quantise/dequantise always run here on the CPU.
# int8 tensors are kept as bytearrays (two's complement), float tensors as lists of floats.

import json
import struct

from . import kernels as kn
from .engine import CpuEngine, MODE_DW

# File magic.
MAGIC = b"TMF1"
# Ops that run on the engine.
ENGINE_OPS = ("conv", "dw", "fc")


# Microsecond timer that works on MicroPython (ticks_us) and CPython (perf_counter).
def _now_us():
    # Import here so the module has no import-time dependencies.
    import time
    # MicroPython.
    if hasattr(time, "ticks_us"):
        return time.ticks_us()
    # CPython.
    return int(time.perf_counter() * 1000000)


# Elapsed microseconds between two _now_us() values.
def _elapsed(t0, t1):
    # Same module again.
    import time
    # MicroPython ticks wrap around.
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(t1, t0)
    # Plain subtraction on CPython.
    return t1 - t0


# Turn whatever the caller passed (list, tuple, numpy array, bytes) into a flat Python list.
def _flat(x):
    # numpy arrays (and anything else with tolist) on the PC.
    if hasattr(x, "tolist"):
        x = x.tolist()
    # Flatten nested lists.
    out = []
    # Iterative stack walk (no recursion limit worries on MicroPython).
    stack = [x]
    # Until everything is unpacked.
    while stack:
        # Next item.
        v = stack.pop()
        # Containers get expanded (in reverse so order is preserved).
        if isinstance(v, (list, tuple)):
            stack.extend(reversed(v))
        else:
            out.append(v)
    # Flat values.
    return out


# A loaded model.
class Model:

    # Read the .tmf header; path may be a file name or the file's bytes.
    def __init__(self, src, engine=None):
        # In-memory model.
        if isinstance(src, (bytes, bytearray)):
            # Keep the bytes.
            self._data, self._f = src, None
            # Header bytes.
            head = src[:8]
        else:
            # Keep the file open so weights can be streamed on demand.
            self._data, self._f = None, open(src, "rb")
            # Header bytes.
            head = self._f.read(8)
        # Check the magic.
        if head[:4] != MAGIC:
            raise ValueError("not a .tmf file (bad magic)")
        # JSON header length.
        hl = struct.unpack("<I", head[4:8])[0]
        # JSON header.
        hj = self._data[8:8 + hl] if self._f is None else self._f.read(hl)
        # Parsed header.
        self.h = json.loads(hj)
        # Start of the binary blob.
        self._blob0 = 8 + hl
        # Tensor table.
        self.tensors = self.h["tensors"]
        # Ops in execution order.
        self.ops = self.h["ops"]
        # Engine (software unless told otherwise).
        self.engine = engine or CpuEngine()
        # Per-op timing of the last run: list of (op name, where, microseconds).
        self.timing = []
        # Index of the last op that reads each tensor (so it can be freed right after).
        self._last = {}
        for i, op in enumerate(self.ops):
            self._last[op["in"]] = i
        # Place weights on the engine.
        self._prepare()

    # Read a [offset, length] blob reference.
    def _blob(self, ref):
        # In memory.
        if self._f is None:
            return self._data[self._blob0 + ref[0]:self._blob0 + ref[0] + ref[1]]
        # From the file.
        self._f.seek(self._blob0 + ref[0])
        # The bytes.
        return self._f.read(ref[1])

    # Decide where every engine op's weights live; upload them now if they all fit.
    def _prepare(self):
        # Engine shortcut.
        e = self.engine
        # Engine ops.
        eops = [op for op in self.ops if op["op"] in ENGINE_OPS]
        # Total weight bytes and parameter channels.
        tw = sum(op["N"] * op["K"] for op in eops)
        tp = sum(op["N"] for op in eops)
        # Everything fits: weights stay resident on the engine for every inference.
        self.resident = tw <= e.w_cap and tp <= e.p_cap
        # Running offsets.
        wo, po = 0, 0
        # Each engine op.
        for op in eops:
            # Resident placement.
            if self.resident:
                # Weight and parameter offsets.
                op["_wb"], op["_pb"] = wo, po
                # Upload once.
                e.upload_w(wo, self._blob(op["w"]))
                e.upload_p(po, self._blob(op["p"]))
                # Advance.
                wo += op["N"] * op["K"]
                po += op["N"]
            # Channels per job: weights, parameters and one vector's outputs must fit, and for depthwise
            # one vector holds K bytes per channel, so X limits the chunk too.
            c = min(op["N"], e.w_cap // op["K"], e.p_cap, e.y_cap)
            # Depthwise: K * chunk bytes per vector in X.
            if op["op"] == "dw":
                c = min(c, e.x_cap // op["K"])
            # Others: one K-byte vector must fit in X.
            elif op["K"] > e.x_cap:
                c = 0
            # 0 means "never fits": this layer runs in software.
            op["_chunk"] = c

    # Input tensor description.
    @property
    def input(self):
        # Tensor of the graph input.
        return self.tensors[self.h["input"]]

    # Output tensor description.
    @property
    def output(self):
        # Tensor of the graph output.
        return self.tensors[self.h["output"]]

    # Run on real-valued input; returns a list of floats (dequantised output, or probabilities after softmax).
    def predict(self, x):
        # Flat input values.
        v = _flat(x)
        # Input tensor.
        ti = self.input
        # int8 models: quantise with the input's scale and zero point.
        if ti["dtype"] == "int8":
            v = bytearray(q & 0xFF for q in kn.quantize(v, ti["scale"], ti["zp"]))
        # Run.
        y = self.invoke(v)
        # Float output already (softmax or dequantize as last op).
        if isinstance(y, list):
            return y
        # Dequantise int8 output.
        to = self.output
        return kn.dequantize(kn.to_i8_list(y), to["scale"], to["zp"])

    # Index of the largest output and its value.
    def classify(self, x):
        # Outputs.
        y = self.predict(x)
        # Best index.
        best = 0
        for i in range(1, len(y)):
            if y[i] > y[best]:
                best = i
        # Index and score.
        return best, y[best]

    # Run on an already-quantised input (bytes of int8, or floats for float-input models).
    # keep=True keeps every intermediate tensor in self.values (used by the LiteRT checker).
    def invoke(self, x, keep=False):
        # Tensor values by id.
        vals = {self.h["input"]: x if not isinstance(x, (bytes, bytearray)) else bytearray(x)}
        # Fresh timing.
        self.timing = []
        # Ops in order.
        for i, op in enumerate(self.ops):
            # Start time.
            t0 = _now_us()
            # Input of the op.
            a = vals[op["in"]]
            # Run it and note where.
            vals[op["out"]], where = self._run_op(op, a)
            # Elapsed time.
            self.timing.append((op["op"], where, _elapsed(t0, _now_us())))
            # Free tensors nobody reads any more (keeps MCU RAM low).
            if not keep and self._last[op["in"]] == i and op["in"] != self.h["output"] and op["in"] != op["out"]:
                del vals[op["in"]]
        # Intermediates for debugging.
        self.values = vals if keep else None
        # Graph output.
        return vals[self.h["output"]]

    # Dispatch one op; returns (output tensor, "fpga"/"cpu").
    def _run_op(self, op, a):
        # Op tag.
        k = op["op"]
        # Engine ops.
        if k in ENGINE_OPS:
            return self._engine_op(op, a)
        # Input / output tensors.
        ti, to = self.tensors[op["in"]], self.tensors[op["out"]]
        # Pure reshape: same bytes.
        if k == "reshape":
            return a, "cpu"
        # Pooling.
        if k in ("maxpool", "avgpool"):
            # Pool on int8 values.
            y, _, _ = kn.pool(kn.to_i8_list(a), op["H"], op["W"], op["C"], op["kh"], op["kw"], op["sh"], op["sw"],
                              op["same"], k == "maxpool", op["act"][0], op["act"][1])
            # Back to bytes.
            return bytearray(v & 0xFF for v in y), "cpu"
        # Softmax on dequantised values -> float probabilities.
        if k == "softmax":
            return kn.softmax(kn.dequantize(kn.to_i8_list(a), ti["scale"], ti["zp"]), op["beta"]), "cpu"
        # Standalone relu / relu6.
        if k == "relu":
            # Requantised clamp per element.
            y = [kn.requant(v - ti["zp"], op["mult"], op["shift"], to["zp"], op["act"][0], op["act"][1])
                 for v in kn.to_i8_list(a)]
            # Back to bytes.
            return bytearray(v & 0xFF for v in y), "cpu"
        # Float -> int8.
        if k == "quantize":
            return bytearray(q & 0xFF for q in kn.quantize(a, to["scale"], to["zp"])), "cpu"
        # int8 -> float.
        if k == "dequantize":
            return kn.dequantize(kn.to_i8_list(a), ti["scale"], ti["zp"]), "cpu"
        # Converter and runtime out of sync.
        raise ValueError("unknown op " + k)

    # Build the input vectors of an engine op: returns (number of vectors, function giving vector i as bytes).
    def _vectors(self, op, a):
        # Input zero point as a byte (the padding value).
        pad = self.tensors[op["in"]]["zp"] & 0xFF
        # Fully connected: the input rows themselves.
        if op["op"] == "fc":
            # Dot length.
            K = op["K"]
            # Row i.
            return op["nvec"], lambda i: a[i * K:(i + 1) * K]
        # Output width (pixels are numbered row-major).
        OW = op["OW"]
        # Convolution patch of pixel i.
        if op["op"] == "conv":
            # Gather into a fresh buffer.
            def vec(i):
                # Patch bytes.
                out = bytearray()
                # im2col for pixel (i // OW, i % OW).
                kn.gather_conv(a, op["H"], op["W"], op["C"], i // OW, i % OW, op["kh"], op["kw"], op["sh"], op["sw"],
                               op["dh"], op["dw"], op["ph"], op["pw"], pad, out)
                # Done.
                return out
        else:
            # Depthwise patch (all channels) of pixel i.
            def vec(i):
                # Patch bytes.
                out = bytearray()
                # Per-channel windows for pixel (i // OW, i % OW).
                kn.gather_dw(a, op["H"], op["W"], op["C"], op["mult"], i // OW, i % OW, op["kh"], op["kw"], op["sh"],
                             op["sw"], op["dh"], op["dw"], op["ph"], op["pw"], pad, out)
                # Done.
                return out
        # One vector per output pixel.
        return op["OH"] * OW, vec

    # Run conv / dw / fc on the engine (falls back to software if the layer can never fit).
    def _engine_op(self, op, a):
        # Engine and sizes.
        e, K, N = self.engine, op["K"], op["N"]
        # Quantisation of the op.
        ti, to = self.tensors[op["in"]], self.tensors[op["out"]]
        # Common job arguments.
        in_off, zp, amin, amax = -ti["zp"], to["zp"], op["act"][0], op["act"][1]
        # Depthwise mode bit.
        dw = op["op"] == "dw"
        # Vector count and generator.
        nv, vec = self._vectors(op, a)
        # Output tensor.
        y = bytearray(nv * N)
        # Engine to use: the configured one, or software when this layer cannot fit at all.
        eng, where = (e, e.name) if op["_chunk"] else (CpuEngine(), "cpu")
        # Channels per chunk.
        cs = op["_chunk"] or N
        # Software fallback needs its own copy of the weights.
        if not op["_chunk"]:
            eng.upload_w(0, self._blob(op["w"]))
            eng.upload_p(0, self._blob(op["p"]))
        # Each channel chunk.
        for n0 in range(0, N, cs):
            # Chunk end and size.
            n1 = min(n0 + cs, N)
            nc = n1 - n0
            # Weight/param location of this chunk.
            if eng is e and self.resident:
                # Already resident.
                wb, pb = op["_wb"] + n0 * K, op["_pb"] + n0
            elif eng is e:
                # Stream this chunk's weights and params in.
                w, p = op["w"], op["p"]
                e.upload_w(0, self._blob([w[0] + n0 * K, nc * K]))
                e.upload_p(0, self._blob([p[0] + n0 * kn.PARAM_BYTES, nc * kn.PARAM_BYTES]))
                wb, pb = 0, 0
            else:
                # Software copy holds the whole layer.
                wb, pb = n0 * K, n0
            # Bytes per vector in X (depthwise: this chunk's channel slices).
            xs = K * nc if dw else K
            # Vectors per job: limited by X, Y and the 16-bit NVEC register.
            nb = min(eng.x_cap // xs, eng.y_cap // nc, 65535, nv)
            # Each batch of vectors.
            for v0 in range(0, nv, nb):
                # Batch end.
                v1 = min(v0 + nb, nv)
                # Pack the batch.
                xb = bytearray()
                for i in range(v0, v1):
                    # Whole vector, or this chunk's slice for depthwise.
                    xb += vec(i)[n0 * K:n1 * K] if dw else vec(i)
                # Run the job.
                r = eng.run(K, nc, v1 - v0, xs, wb, pb, in_off, zp, amin, amax, MODE_DW if dw else 0, xb)
                # Scatter into NHWC order.
                j = 0
                for i in range(v0, v1):
                    # Row start of pixel i plus the chunk offset.
                    base = i * N + n0
                    for c in range(nc):
                        y[base + c] = r[j] & 0xFF
                        j += 1
        # Output tensor and where it ran.
        return y, where

    # Human-readable summary of the ops, their sizes and where they run.
    def summary(self):
        # Lines to print.
        lines = ["engine: %s  (W %d B, X %d B, Y %d B, P %d ch)  weights %s" % (
            self.engine.name, self.engine.w_cap, self.engine.x_cap, self.engine.y_cap, self.engine.p_cap,
            "resident" if self.resident else "streamed per inference")]
        # Each op.
        for op in self.ops:
            # Output shape.
            sh = self.tensors[op["out"]]["shape"]
            # Engine ops show their GEMV size.
            if op["op"] in ENGINE_OPS:
                info = "K=%d N=%d  MACs=%d  %s" % (op["K"], op["N"], op["K"] * op["N"] * (op.get("nvec") or op["OH"] * op["OW"]),
                                                  "engine" if op["_chunk"] else "cpu (too big for engine)")
            else:
                info = "cpu"
            # One line.
            lines.append("  %-10s -> %-18s %s" % (op["op"], str(sh), info))
        # Print and return.
        s = "\n".join(lines)
        print(s)
        return s

    # Close the model file.
    def close(self):
        # Only when streaming from a file.
        if self._f:
            self._f.close()
