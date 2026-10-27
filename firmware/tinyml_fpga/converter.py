# converter.py - PC side: turn a fully int8-quantised .tflite model into a .tmf file.
# (Needs numpy and the 'tflite' flatbuffer schema package; never imported on a microcontroller.)
#
# .tmf layout:  b"TMF1" | u32 json_len | json header | binary blob
#   The json header lists the tensors and the ops in execution order. Engine ops (conv, dw, fc)
#   point into the blob for their weights (int8, one row of K bytes per output channel) and
#   their parameters (16 bytes per channel: bias i32, multiplier i32, shift i8, pad).
#   Everything the runtime needs is precomputed here, so a microcontroller never parses a flatbuffer.

import json
import math
import struct

import numpy as np

from .kernels import PARAM_BYTES, pack_param, round_away, out_and_pad

# File magic.
MAGIC = b"TMF1"
# Format version written into the header.
VERSION = 1


# TFLite QuantizeMultiplier: real multiplier -> (int32 fixed-point multiplier, shift).
def quantize_multiplier(m):
    # Zero stays zero.
    if m == 0.0:
        return 0, 0
    # m = q * 2^shift with q in [0.5, 1).
    q, shift = math.frexp(m)
    # q as a Q31 fixed-point number, rounded half away from zero.
    q_fixed = round_away(q * (1 << 31))
    # Rounding can reach exactly 1.0: renormalise.
    if q_fixed == (1 << 31):
        # Halve the mantissa.
        q_fixed //= 2
        # Bump the exponent.
        shift += 1
    # Too small to represent: flush to zero.
    if shift < -31:
        return 0, 0
    # Too large: saturate like newer TFLite releases.
    if shift > 30:
        return (1 << 31) - 1, 30
    # Fixed-point multiplier and shift.
    return q_fixed, shift


# TFLite CalculateActivationRangeQuantized for int8 outputs.
def act_range(fused, scale, zp):
    # Float32 maths, as in the C++ (scale is a float).
    q = lambda f: zp + round_away(float(np.float32(f) / np.float32(scale)))
    # NONE.
    if fused == 0:
        return -128, 127
    # RELU.
    if fused == 1:
        return max(-128, q(0.0)), 127
    # RELU_N1_TO_1.
    if fused == 2:
        return max(-128, q(-1.0)), min(127, q(1.0))
    # RELU6.
    if fused == 3:
        return max(-128, q(0.0)), min(127, q(6.0))
    # TANH / SIGN_BIT are not fused activations of int8 conv/fc.
    raise ValueError("unsupported fused activation %d" % fused)


# Builder that walks the flatbuffer and produces the .tmf header and blob.
class Converter:

    # Load the flatbuffer and index operators by name.
    def __init__(self, data):
        # Import here so 'import tinyml_fpga' never needs the schema package.
        import tflite
        # Keep the module for option classes.
        self.tfl = tflite
        # Root model object.
        self.m = tflite.Model.GetRootAsModel(data, 0)
        # Only single-subgraph models are supported.
        if self.m.SubgraphsLength() != 1:
            raise ValueError("model has %d subgraphs, only 1 is supported" % self.m.SubgraphsLength())
        # The graph.
        self.g = self.m.Subgraphs(0)
        # Builtin operator code -> name.
        self.opname = {v: k for k, v in tflite.BuiltinOperator.__dict__.items() if not k.startswith("_")}
        # Binary blob being built.
        self.blob = bytearray()
        # Tensor index (tflite) -> index in our tensor list.
        self.tmap = {}
        # Our tensor list.
        self.tensors = []

    # Name of operator i.
    def op_name(self, op):
        # Operator code entry.
        oc = self.m.OperatorCodes(op.OpcodeIndex())
        # Newer files store large codes in BuiltinCode, older ones in DeprecatedBuiltinCode.
        return self.opname[max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())]

    # Tensor object by index.
    def T(self, i):
        # Plain accessor.
        return self.g.Tensors(i)

    # Constant data of a tensor as a numpy array, or None for activations.
    def const(self, i, dtype):
        # Buffer of the tensor.
        b = self.m.Buffers(self.T(i).Buffer())
        # Activations have empty buffers.
        if b is None or b.DataLength() == 0:
            return None
        # Reinterpret the raw bytes.
        return np.frombuffer(b.DataAsNumpy().tobytes(), dtype=dtype).reshape(self.T(i).ShapeAsNumpy())

    # Quantisation (scales array, zero points array) of a tensor.
    def quant(self, i):
        # Quantisation table.
        q = self.T(i).Quantization()
        # Unquantised tensors have no table.
        if q is None or q.ScaleLength() == 0:
            return None, None
        # Scales and zero points as numpy arrays.
        return q.ScaleAsNumpy(), q.ZeroPointAsNumpy()

    # Register an activation tensor and return its id in our list.
    def tensor(self, i):
        # Already known.
        if i in self.tmap:
            return self.tmap[i]
        # Tensor object.
        t = self.T(i)
        # Scale / zero point (per-tensor for activations).
        s, z = self.quant(i)
        # Record shape, dtype and quantisation.
        self.tensors.append({
            "shape": [int(d) for d in t.ShapeAsNumpy()],
            "dtype": {9: "int8", 0: "float32"}.get(t.Type(), "type%d" % t.Type()),
            "scale": float(s[0]) if s is not None else 1.0,
            "zp": int(z[0]) if z is not None else 0,
        })
        # Map tflite index -> ours.
        self.tmap[i] = len(self.tensors) - 1
        # New id.
        return self.tmap[i]

    # Append bytes to the blob, return [offset, length].
    def put(self, data):
        # Current end of the blob.
        off = len(self.blob)
        # Append.
        self.blob += data
        # Reference used in the header.
        return [off, len(data)]

    # Operators that actually produce the graph outputs (drops Keras' SHAPE/STRIDED_SLICE/PACK side chains).
    def live_ops(self):
        # Tensors we need, starting from the graph outputs.
        need = set(int(t) for t in self.g.OutputsAsNumpy())
        # Live op indices.
        live = []
        # Walk backwards (ops are stored in execution order).
        for i in reversed(range(self.g.OperatorsLength())):
            # Operator.
            op = self.g.Operators(i)
            # Skip ops whose outputs nobody needs.
            if not any(int(t) in need for t in op.OutputsAsNumpy()):
                continue
            # Keep it.
            live.append(i)
            # Only the data input (input 0) is a runtime dependency; shapes come from the tensor itself.
            need.add(int(op.InputsAsNumpy()[0]))
        # Execution order.
        return list(reversed(live))

    # Engine parameters for conv / dw / fc: weights rows and the 16-byte channel params.
    def engine_params(self, op, w_rows, in_scale, out_scale, per_tensor_fc):
        # Input indices.
        ins = op.InputsAsNumpy()
        # Weight scales (per channel or one).
        w_s, _ = self.quant(int(ins[1]))
        # Number of output channels.
        N = w_rows.shape[0]
        # Bias (int32) or zeros.
        bias = self.const(int(ins[2]), np.int32).reshape(-1) if len(ins) > 2 and ins[2] >= 0 else np.zeros(N, np.int32)
        # Parameter records.
        prm = bytearray()
        # One record per output channel.
        for o in range(N):
            # Per-channel weight scale, or the single one.
            ws = w_s[o] if len(w_s) > 1 else w_s[0]
            # FC per-tensor: TFLite multiplies input and filter scale in float32 first.
            if per_tensor_fc and len(w_s) == 1:
                # float32 product, then double division.
                real = float(np.float32(in_scale) * np.float32(ws)) / float(out_scale)
            else:
                # Conv path: everything in double.
                real = float(in_scale) * float(ws) / float(out_scale)
            # Fixed-point multiplier.
            mult, shift = quantize_multiplier(real)
            # Record.
            prm += pack_param(int(bias[o]), mult, shift)
        # Blob references for weights and parameters.
        return self.put(w_rows.astype(np.int8).tobytes()), self.put(bytes(prm))

    # Convert the whole graph; returns the header dict.
    def run(self):
        # Option classes from the schema package.
        tfl = self.tfl
        # Header ops.
        ops = []
        # Graph input.
        gin = int(self.g.InputsAsNumpy()[0])
        # Graph output.
        gout = int(self.g.OutputsAsNumpy()[0])
        # Each live operator.
        for i in self.live_ops():
            # Operator and its name.
            op = self.g.Operators(i)
            name = self.op_name(op)
            # Input / output tensor indices.
            ins, outs = op.InputsAsNumpy(), op.OutputsAsNumpy()
            # Data input and output.
            ti, to = int(ins[0]), int(outs[0])
            # Their ids in our list.
            a, b = self.tensor(ti), self.tensor(to)
            # Their descriptions.
            ta, tb = self.tensors[a], self.tensors[b]
            # Common fields.
            d = {"in": a, "out": b}
            # Convolution and depthwise convolution.
            if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                # Options table.
                o = (tfl.Conv2DOptions() if name == "CONV_2D" else tfl.DepthwiseConv2DOptions())
                # Bind it to the operator's options.
                o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
                # Input NHWC.
                _, H, W, C = ta["shape"]
                # Output NHWC.
                _, OH, OW, OC = tb["shape"]
                # Filter tensor.
                wt = self.const(int(ins[1]), np.int8)
                # Geometry.
                sh, sw = o.StrideH(), o.StrideW()
                dh, dw = o.DilationHFactor(), o.DilationWFactor()
                same = o.Padding() == tfl.Padding.SAME
                # Filter size: conv [OC,kh,kw,C], depthwise [1,kh,kw,OC].
                kh, kw = int(wt.shape[1]), int(wt.shape[2])
                # Padding exactly as TFLite computes it.
                oh2, ph = out_and_pad(H, kh, sh, dh, same)
                ow2, pw = out_and_pad(W, kw, sw, dw, same)
                # Sanity check against the stored output shape.
                assert (oh2, ow2) == (OH, OW), "output shape mismatch in %s" % name
                # Weight rows: conv is already [OC][kh*kw*C]; depthwise needs [OC][kh*kw].
                if name == "CONV_2D":
                    # Grouped convolutions are not supported.
                    if wt.shape[3] != C:
                        raise ValueError("grouped CONV_2D is not supported")
                    # One row per filter.
                    rows = wt.reshape(OC, -1)
                    # Op tag and dot length.
                    d["op"], K = "conv", kh * kw * C
                else:
                    # Channel multiplier.
                    d["mult"] = OC // C
                    # Transpose [1,kh,kw,OC] -> [OC][kh*kw].
                    rows = wt.reshape(kh * kw, OC).T
                    # Op tag and dot length.
                    d["op"], K = "dw", kh * kw
                # Weight and parameter blob references.
                d["w"], d["p"] = self.engine_params(op, rows, ta["scale"], tb["scale"], False)
                # Fused activation clamp.
                d["act"] = list(act_range(o.FusedActivationFunction(), tb["scale"], tb["zp"]))
                # Geometry for the patch gather.
                d.update(K=K, N=OC, H=H, W=W, C=C, OH=OH, OW=OW, kh=kh, kw=kw, sh=sh, sw=sw, dh=dh, dw=dw, ph=ph, pw=pw)
            # Fully connected.
            elif name == "FULLY_CONNECTED":
                # Options table.
                o = tfl.FullyConnectedOptions()
                o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
                # Weights [N][K].
                wt = self.const(int(ins[1]), np.int8)
                # Sizes.
                N, K = int(wt.shape[0]), int(wt.shape[1])
                # Batch = elements / K.
                nb = 1
                for s in ta["shape"]:
                    nb *= s
                nb //= K
                # Op fields.
                d.update(op="fc", K=K, N=N, nvec=nb)
                # Weight and parameter blob references (per-tensor scale product in float32).
                d["w"], d["p"] = self.engine_params(op, wt, ta["scale"], tb["scale"], True)
                # Fused activation clamp.
                d["act"] = list(act_range(o.FusedActivationFunction(), tb["scale"], tb["zp"]))
            # Pooling.
            elif name in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
                # Options table.
                o = tfl.Pool2DOptions()
                o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
                # Input NHWC.
                _, H, W, C = ta["shape"]
                # Op fields.
                d.update(op="maxpool" if name == "MAX_POOL_2D" else "avgpool", H=H, W=W, C=C,
                         kh=o.FilterHeight(), kw=o.FilterWidth(), sh=o.StrideH(), sw=o.StrideW(),
                         same=o.Padding() == tfl.Padding.SAME,
                         act=list(act_range(o.FusedActivationFunction(), tb["scale"], tb["zp"])))
            # Pure shape changes (NHWC data is already in the right order).
            elif name in ("RESHAPE", "SQUEEZE", "EXPAND_DIMS"):
                # Data is untouched.
                d["op"] = "reshape"
            # Softmax (float on the CPU).
            elif name == "SOFTMAX":
                # Options table.
                o = tfl.SoftmaxOptions()
                o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
                # Op fields.
                d.update(op="softmax", beta=float(o.Beta()))
            # Standalone RELU / RELU6 (re-quantised clamp).
            elif name in ("RELU", "RELU6"):
                # Real ratio between input and output scales.
                mult, shift = quantize_multiplier(float(ta["scale"]) / float(tb["scale"]))
                # Op fields.
                d.update(op="relu", mult=mult, shift=shift,
                         act=list(act_range(1 if name == "RELU" else 3, tb["scale"], tb["zp"])))
            # Float model input.
            elif name == "QUANTIZE" and ta["dtype"] == "float32":
                # Quantise floats with the output's scale and zero point.
                d["op"] = "quantize"
            # Float model output.
            elif name == "DEQUANTIZE":
                # Convert int8 back to float.
                d["op"] = "dequantize"
            # Everything else is reported clearly.
            else:
                raise ValueError("operator %s is not supported (supported: CONV_2D, DEPTHWISE_CONV_2D, "
                                 "FULLY_CONNECTED, MAX/AVERAGE_POOL_2D, RESHAPE, SQUEEZE, SOFTMAX, RELU, RELU6, "
                                 "QUANTIZE, DEQUANTIZE)" % name)
            # Only int8 activations may feed engine and pooling ops.
            if d["op"] in ("conv", "dw", "fc", "maxpool", "avgpool", "relu") and ta["dtype"] != "int8":
                raise ValueError("%s input must be int8 (fully quantise the model)" % name)
            # Keep the op.
            ops.append(d)
        # Header.
        return {"version": VERSION, "input": self.tensor(gin), "output": self.tensor(gout),
                "tensors": self.tensors, "ops": ops}


# Convert .tflite bytes to .tmf bytes.
def convert_bytes(data):
    # Build the header and blob.
    c = Converter(data)
    header = c.run()
    # Compact JSON (MicroPython's json reads it directly).
    hj = json.dumps(header, separators=(",", ":")).encode()
    # Magic, header length, header, blob.
    return MAGIC + struct.pack("<I", len(hj)) + hj + bytes(c.blob)


# Convert a .tflite file to a .tmf file; returns the header for reporting.
def convert_file(src, dst):
    # Read the source model.
    with open(src, "rb") as f:
        data = f.read()
    # Convert.
    out = convert_bytes(data)
    # Write the result.
    with open(dst, "wb") as f:
        f.write(out)
    # Header back for the caller.
    return json.loads(out[8:8 + struct.unpack_from("<I", out, 4)[0]])
