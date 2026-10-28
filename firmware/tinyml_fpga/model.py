# model.py - loads a .tmf file and runs it. MicroPython-safe (no numpy, no typing, no dataclasses).
# Engine ops (conv, depthwise, fully connected) go to the engine (FPGA or CpuEngine); pooling,
# reshape, softmax, relu, quantise/dequantise always run here on the CPU.
# int8 tensors are kept as bytearrays (two's complement), float tensors as lists of floats.

import json
import struct

from . import kernels as kn
from .engine import CpuEngine, MODE_DW, MODE_WIN, MODE_OUTX, MODE_W4, MODE_CB, window_block

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
        # Runs of layers the engine can execute with one start.
        self._plan_chains()

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
        # Engine shortcut and its lane count (channels are spread over `lanes` weight banks).
        e = self.engine
        L = e.lanes
        # Engine ops.
        eops = [op for op in self.ops if op["op"] in ENGINE_OPS]
        # Rows per bank of a layer: ceil(N / lanes). Weight bytes per row: K, or ceil(K / 2) for 4-bit weights.
        for op in eops:
            op["_rows"] = (op["N"] + L - 1) // L
            w4 = op.get("wbits", 8) == 4
            # A codebook layer stores 4-bit indices and carries one extra parameter record (the table).
            op["_cb"] = 1 if op.get("cb") else 0
            op["_rb"] = kn.row_bytes(op["K"], w4 or op["_cb"])
            op["_wm"] = MODE_CB if op["_cb"] else (MODE_W4 if w4 else 0)
        # Total weight bytes and parameter records per bank.
        tw = sum(op["_rows"] * op["_rb"] for op in eops)
        tp = sum(op["_rows"] + op["_cb"] for op in eops)
        # Everything fits: weights stay resident on the engine for every inference.
        self.resident = tw <= e.w_cap and tp <= e.p_cap
        # Running local offsets (per bank).
        wo, po = 0, 0
        # Each engine op.
        for op in eops:
            # Resident placement.
            if self.resident:
                # Weight and parameter offsets.
                op["_wb"], op["_pb"] = wo, po
                # Upload once.
                e.upload_w(wo, self._blob(op["w"]), op["_rb"])
                e.upload_p(po, self._blob(op["p"]))
                # Codebook table right after the layer's records.
                if op["_cb"]:
                    e.upload_cb(po + op["_rows"], self._blob(op["cb"]))
                # Advance.
                wo += op["_rows"] * op["_rb"]
                po += op["_rows"] + op["_cb"]
            # Channels per job: weights, parameters and one vector's outputs must fit, and for depthwise
            # one vector holds K bytes per channel, so X limits the chunk too.
            c = min(op["N"], L * (e.w_cap // op["_rb"]), L * (e.p_cap - op["_cb"]), e.y_cap)
            # Depthwise: K * chunk bytes per vector in X.
            if op["op"] == "dw":
                c = min(c, e.x_cap // op["K"])
            # Others: one K-byte vector must fit in X.
            elif op["K"] > e.x_cap:
                c = 0
            # Chunks that do not cover the whole layer must start on a lane boundary.
            if c < op["N"]:
                c -= c % L
            # An engine without 4-bit weights cannot run a 4-bit layer: software.
            if (op["_wm"] == MODE_W4 and not getattr(e, "has_w4", False)) or (op["_cb"] and not getattr(e, "has_cb", False)):
                c = 0
            # 0 means "never fits": this layer runs in software.
            op["_chunk"] = c
            # Window mode (the engine walks the convolution windows itself): channel chunk limited by one output row.
            op["_wchunk"] = 0
            if getattr(e, "has_win", False) and op["op"] in ("conv", "dw"):
                # The image must fit X and one row of outputs of the chunk must fit Y.
                img = op["H"] * op["W"] * op["C"]
                cw = min(op["N"], L * (e.w_cap // op["_rb"]), L * (e.p_cap - op["_cb"]), e.y_cap // op["OW"])
                # Chunks that do not cover the whole layer must start on a lane boundary.
                if cw < op["N"]:
                    cw -= cw % L
                # Depthwise channels cannot be chunked (the window generator starts at input channel 0).
                if op["op"] == "dw" and cw < op["N"]:
                    cw = 0
                # Descriptor fields must hold the geometry.
                if img <= e.x_cap and cw > 0 and self._win(op, 0, 0) is not None and (op["_wm"] != MODE_W4 or getattr(e, "has_w4", False)) and (not op["_cb"] or getattr(e, "has_cb", False)):
                    op["_wchunk"] = cw

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
        # Ops in order (a planned chain of layers counts as one step; debugging runs keep every tensor, so no chains).
        i = 0
        while i < len(self.ops):
            op = self.ops[i]
            # Start time.
            t0 = _now_us()
            # Input of the op.
            a = vals[op["in"]]
            # A run of layers with one start.
            ch = None if keep else self._chains.get(i)
            if ch:
                # The last layer's output tensor.
                last = self.ops[ch["end"]]
                vals[last["out"]] = self._run_chain(ch, a)
                # Timing entry for the whole run.
                self.timing.append(("+".join(self.ops[m]["op"] for m in ch["ops"]), self.engine.name + "-fused", _elapsed(t0, _now_us())))
                # Free the chain input if nobody else reads it.
                if self._last[op["in"]] == i and op["in"] != self.h["output"]:
                    del vals[op["in"]]
                i = ch["end"] + 1
                continue
            # Run it and note where.
            vals[op["out"]], where = self._run_op(op, a)
            # Elapsed time.
            self.timing.append((op["op"], where, _elapsed(t0, _now_us())))
            # Free tensors nobody reads any more (keeps MCU RAM low).
            if not keep and self._last[op["in"]] == i and op["in"] != self.h["output"] and op["in"] != op["out"]:
                del vals[op["in"]]
            i += 1
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

    # Window block of a conv / dw op for output rows r0.. with the image at xbase in X (None if it does not fit).
    def _win(self, op, r0, xbase):
        return window_block(op["H"], op["W"], op["C"], op["kh"], op["kw"], op["dh"], op["dw"], op["sh"], op["sw"],
                            op["ph"], op["pw"], op["mult"] if op["op"] == "dw" else 1, op["OW"], r0, xbase)

    # Weight and parameter offsets of the channel chunk n0..n0+nc of an op on engine eng (uploads when streaming).
    def _place(self, op, eng, n0, nc):
        # Engine and sizes.
        e, K = self.engine, op["_rb"]
        # Already resident (n0 is a multiple of the lane count, so its rows start n0 // lanes into the layer).
        if eng is e and self.resident:
            return op["_wb"] + (n0 // e.lanes) * K, op["_pb"] + n0 // e.lanes
        # Stream this chunk's weights and params in.
        if eng is e:
            w, p = op["w"], op["p"]
            e.upload_w(0, self._blob([w[0] + n0 * K, nc * K]), K)
            e.upload_p(0, self._blob([p[0] + n0 * kn.PARAM_BYTES, nc * kn.PARAM_BYTES]))
            if op["_cb"]:
                e.upload_cb((nc + e.lanes - 1) // e.lanes, self._blob(op["cb"]))
            return 0, 0
        # Software copy holds the whole layer.
        return n0 * K, n0

    # Convolution / depthwise on the engine's window generator: the image goes to X once and the engine walks the
    # windows (padding, stride, dilation) itself, so no patches are built or sent.
    def _engine_op_win(self, op, a):
        # Engine and sizes.
        e, K, N, OH, OW = self.engine, op["K"], op["N"], op["OH"], op["OW"]
        # Quantisation of the op.
        ti, to = self.tensors[op["in"]], self.tensors[op["out"]]
        in_off, zp, amin, amax = -ti["zp"], to["zp"], op["act"][0], op["act"][1]
        # Depthwise mode bit.
        dw = op["op"] == "dw"
        # The image, once.
        e.write_x(0, a)
        # Output tensor.
        y = bytearray(OH * OW * N)
        # Channels per chunk.
        cs = op["_wchunk"]
        # Each channel chunk.
        for n0 in range(0, N, cs):
            # Chunk end and size.
            nc = min(n0 + cs, N) - n0
            # Weights of this chunk.
            wb, pb = self._place(op, e, n0, nc)
            # Output rows per job: one job's outputs must fit Y and the 16-bit vector count.
            rb = max(1, min(OH, e.y_cap // (OW * nc), 65535 // OW))
            # Each batch of output rows.
            for r0 in range(0, OH, rb):
                # Rows r0 .. r1-1.
                r1 = min(r0 + rb, OH)
                # Run the job (the image is already in X).
                r = e.run(K, nc, (r1 - r0) * OW, 0, wb, pb, in_off, zp, amin, amax, MODE_WIN | op["_wm"] | (MODE_DW if dw else 0),
                          None, 0, 0, self._win(op, r0, 0))
                # Scatter into NHWC order.
                j = 0
                for i in range(r0 * OW, r1 * OW):
                    # Row start of pixel i plus the chunk offset.
                    base = i * N + n0
                    for c in range(nc):
                        y[base + c] = r[j] & 0xFF
                        j += 1
        # Output tensor and where it ran.
        return y, e.name

    # Find runs of consecutive engine layers that can run as one queue: each layer writes its output into X where the
    # next one reads it, so the whole run costs one descriptor upload and one start.
    def _plan_chains(self):
        # Chain start index -> plan.
        self._chains = {}
        e = self.engine
        # Needs queue + fusion, resident weights and a single job per layer.
        if not (getattr(e, "has_queue", False) and getattr(e, "has_fuse", False) and self.resident):
            return
        ops = self.ops
        # How many ops read each tensor.
        cons = {}
        for op in ops:
            cons[op["in"]] = cons.get(op["in"], 0) + 1
        i, n = 0, len(ops)
        while i < n:
            ch = self._chain_from(i, cons)
            if ch:
                self._chains[i] = ch
                i = ch["end"] + 1
            else:
                i += 1

    # Can this op be a link of a chain? Returns (input bytes, output bytes) or None.
    def _link(self, op):
        e = self.engine
        # Window convolution / depthwise in a single job.
        if op["op"] in ("conv", "dw"):
            if op["_wchunk"] != op["N"] or op["OH"] * op["OW"] > 65535:
                return None
            return op["H"] * op["W"] * op["C"], op["OH"] * op["OW"] * op["N"]
        # Fully connected over one vector, in a single job.
        if op["op"] == "fc" and op["_chunk"] == op["N"] and op["nvec"] == 1 and op["K"] <= e.x_cap:
            return op["K"], op["N"]
        return None

    # Plan the longest chain starting at op i (None if shorter than two layers).
    def _chain_from(self, i, cons):
        e, ops = self.engine, self.ops
        # Member op indices and their (in, out) sizes.
        mem, sizes = [], []
        j = i
        while j < len(ops):
            op = ops[j]
            if op["op"] not in ENGINE_OPS:
                break
            sz = self._link(op)
            # Must be linkable and, after the first, take exactly the previous output.
            if sz is None or (mem and sz[0] != sizes[-1][1]):
                break
            mem.append(j)
            sizes.append(sz)
            # Follow the output through reshapes; it must have a single reader (the next link).
            t, k = op["out"], j + 1
            while k < len(ops) and ops[k]["op"] == "reshape" and ops[k]["in"] == t and cons.get(t, 0) == 1:
                t = ops[k]["out"]
                k += 1
            # Next link must read this tensor and nobody else may.
            if k >= len(ops) or ops[k]["in"] != t or cons.get(t, 0) != 1 or ops[k]["op"] not in ENGINE_OPS:
                break
            j = k
        # Fit the chain into X (two regions used alternately), Y (last output) and the descriptor slots.
        while len(mem) >= 2:
            # Region A holds the chain input and every second output; region B the others.
            sa = max([sizes[0][0]] + [sizes[m][1] for m in range(1, len(mem) - 1, 2)])
            sb = max([sizes[m][1] for m in range(0, len(mem) - 1, 2)])
            if sa + sb <= e.x_cap and sizes[-1][1] <= e.y_cap and len(mem) <= min(e.queue_slots, 255):
                break
            mem.pop()
            sizes.pop()
        if len(mem) < 2:
            return None
        # Region bases and the job list.
        base = (0, sa)
        jobs = []
        for m, oi in enumerate(mem):
            op = ops[oi]
            ti, to = self.tensors[op["in"]], self.tensors[op["out"]]
            xin = base[m & 1]
            last = m == len(mem) - 1
            # Output into the other region, or Y for the last link.
            yout = 0 if last else base[(m + 1) & 1]
            mode = (0 if last else MODE_OUTX)
            if op["op"] == "fc":
                job = [op["K"], op["N"], 1, op["K"], op["_wb"], op["_pb"], -ti["zp"], to["zp"], op["act"][0], op["act"][1],
                       mode | op["_wm"], xin, yout]
            else:
                win = self._win(op, 0, xin)
                if win is None:
                    return None
                job = [op["K"], op["N"], op["OH"] * op["OW"], 0, op["_wb"], op["_pb"], -ti["zp"], to["zp"], op["act"][0],
                       op["act"][1], mode | op["_wm"] | MODE_WIN | (MODE_DW if op["op"] == "dw" else 0), xin, yout, win]
            jobs.append(tuple(job))
        return {"end": mem[-1], "jobs": jobs, "ops": mem, "out": sizes[-1][1]}

    # Run a planned chain: input image to X, one queue start, final output from Y.
    def _run_chain(self, ch, a):
        e = self.engine
        # Chain input at the start of X.
        e.write_x(0, a)
        # All layers, one start.
        e.run_queue(ch["jobs"])
        # Output of the last layer.
        return bytearray(e.read_y(0, ch["out"]))

    # Run conv / dw / fc on the engine (falls back to software if the layer can never fit).
    def _engine_op(self, op, a):
        # Window generator available for this layer: no host-side patches.
        if op.get("_wchunk"):
            return self._engine_op_win(op, a)
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
            eng.upload_w(0, self._blob(op["w"]), op["_rb"])
            eng.upload_p(0, self._blob(op["p"]))
            if op["_cb"]:
                eng.upload_cb(N, self._blob(op["cb"]))
        # Each channel chunk.
        for n0 in range(0, N, cs):
            # Chunk end and size.
            n1 = min(n0 + cs, N)
            nc = n1 - n0
            # Weight/param location of this chunk.
            wb, pb = self._place(op, eng, n0, nc)
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
                r = eng.run(K, nc, v1 - v0, xs, wb, pb, in_off, zp, amin, amax, (MODE_DW if dw else 0) | op["_wm"], xb)
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
        e = self.engine
        lines = ["engine: %s  (%d lane%s, W %d B/bank, X %d B, Y %d B, P %d ch/bank)  weights %s" % (
            e.name, e.lanes, "" if e.lanes == 1 else "s", e.w_cap, e.x_cap, e.y_cap, e.p_cap,
            "resident" if self.resident else "streamed per inference")]
        # Each op.
        for op in self.ops:
            # Output shape.
            sh = self.tensors[op["out"]]["shape"]
            # Engine ops show their GEMV size.
            if op["op"] in ENGINE_OPS:
                info = "K=%d N=%d  MACs=%d  %s%s" % (op["K"], op["N"], op["K"] * op["N"] * (op.get("nvec") or op["OH"] * op["OW"]),
                                                    "engine" if (op["_chunk"] or op.get("_wchunk")) else "cpu (too big for engine)",
                                                    "  int4" if op["_wm"] else "")
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
