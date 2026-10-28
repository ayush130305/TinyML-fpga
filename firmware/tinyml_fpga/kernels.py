# kernels.py - bit-exact int8 kernels with TensorFlow Lite semantics, in plain Python.
# Runs unchanged on CPython and on MicroPython (no numpy, no f-string formatting tricks,
# no int.to_bytes(signed=...)), so a microcontroller and the PC compute identical numbers.
#
# The one compute primitive the FPGA engine implements is dot_requant(): an int8 dot
# product with an input offset, an int32 bias and TFLite's fixed-point requantisation.
# Conv, depthwise conv and fully-connected layers are all reduced to it by gathering
# "patches" (gather_conv / gather_dw); everything else here (pooling, softmax, quantise)
# always runs on the CPU.

import struct

# Smallest int32 value.
INT32_MIN = -2147483648
# Largest int32 value.
INT32_MAX = 2147483647
# Bytes of per-channel parameters (bias i32, multiplier i32, shift i8, 7 pad bytes).
PARAM_BYTES = 16


# Wrap a Python int to a signed 32-bit value (two's complement), like C int32 arithmetic.
def wrap32(v):
    # Keep the low 32 bits.
    v &= 0xFFFFFFFF
    # Re-interpret bit 31 as the sign.
    return v - 0x100000000 if v & 0x80000000 else v


# TFLite SaturatingRoundingDoublingHighMul(a, b): high 32 bits of 2*a*b with round-to-nearest.
def srdhm(a, b):
    # The only overflowing input pair saturates to INT32_MAX.
    if a == INT32_MIN and b == INT32_MIN:
        return INT32_MAX
    # Full 64-bit product (exact in Python).
    ab = a * b
    # Nudge so the truncating division rounds to nearest.
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    # Sum to divide by 2^31.
    s = ab + nudge
    # C integer division truncates toward zero, which >> does not do for negatives.
    q = (s >> 31) if s >= 0 else -((-s) >> 31)
    # The result always fits in int32.
    return wrap32(q)


# TFLite RoundingDivideByPOT(x, e): x / 2^e rounded to nearest, ties away from zero.
def rdbpot(x, e):
    # Mask of the bits shifted out.
    mask = (1 << e) - 1
    # Value of the bits shifted out.
    rem = x & mask
    # Threshold: half, plus one for negative x (ties away from zero).
    thr = (mask >> 1) + (1 if x < 0 else 0)
    # Arithmetic shift plus the rounding increment.
    return (x >> e) + (1 if rem > thr else 0)


# TFLite MultiplyByQuantizedMultiplier(acc, mult, shift), single-rounding form (TFLITE_SINGLE_ROUNDING):
#   (acc * mult + 2^(30 - shift)) >> (31 - shift)       (64-bit product, arithmetic shift = round half up)
# This is the variant LiteRT uses: checked bit-exact against it on conv, depthwise and FC layers
# (the older double-rounding form, srdhm + rdbpot above, differs by 1 LSB on ~0.3% of outputs).
def mbqm(acc, mult, shift):
    # Total right shift, 1..62 for shift in [-31, 30].
    ts = 31 - shift
    # Product plus half an output step, shifted down (Python >> is arithmetic).
    return (acc * mult + (1 << (ts - 1))) >> ts


# The older double-rounding MultiplyByQuantizedMultiplier, kept for TFLM builds that use it.
def mbqm_double(acc, mult, shift):
    # Positive shift is applied before the multiply.
    left = shift if shift > 0 else 0
    # Negative shift is applied after the multiply, with rounding.
    right = -shift if shift < 0 else 0
    # Left shift wraps in int32 exactly like the C code.
    return rdbpot(srdhm(wrap32(acc * (1 << left)), mult), right)


# Requantise an int32 accumulator to int8: multiply, add output zero point, clamp to the activation range.
def requant(acc, mult, shift, out_zp, act_min, act_max):
    # Fixed-point rescale.
    v = mbqm(acc, mult, shift) + out_zp
    # Clamp low.
    if v < act_min:
        v = act_min
    # Clamp high.
    if v > act_max:
        v = act_max
    # int8 result.
    return v


# Split a 16-byte parameter record into (bias, multiplier, shift).
def unpack_param(buf, off):
    # bias i32, multiplier i32, shift i8, little endian.
    return struct.unpack_from("<iib", buf, off)


# Pack (bias, multiplier, shift) into a 16-byte parameter record.
def pack_param(bias, mult, shift):
    # Seven pad bytes keep every record 16-byte aligned for the RTL address math.
    return struct.pack("<iib", bias, mult, shift) + bytes(7)


# Convert an unsigned byte (0..255) to signed int8.
def s8(b):
    # Bytes >= 128 are negative.
    return b - 256 if b > 127 else b


# Table so bytes -> int8 lists are a fast lookup instead of a branch per element.
_S8 = [i - 256 if i > 127 else i for i in range(256)]


# Weight bytes per output channel row: K bytes for int8 weights, ceil(K / 2) for packed 4-bit weights.
def row_bytes(K, w4):
    return (K + 1) // 2 if w4 else K


# Unpack N rows of K 4-bit weights (two per byte, low nibble = even index, rows start on a byte) to int8 values:
# signed nibbles, or with a 16-entry codebook cb (int8 values) the nibbles are indices into it.
def unpack_w4(buf, K, N, cb=None):
    # Bytes per row.
    rb = (K + 1) // 2
    # Output list.
    out = []
    for o in range(N):
        # Row start.
        b0 = o * rb
        for k in range(K):
            # Nibble k of the row.
            v = (buf[b0 + (k >> 1)] >> ((k & 1) * 4)) & 0xF
            # Codebook lookup, or sign extend.
            out.append(cb[v] if cb is not None else (v - 16 if v > 7 else v))
    return out


# Convert a bytes-like object of int8 data to a list of Python ints.
def to_i8_list(buf):
    # Table lookup per byte.
    t = _S8
    # Build the list.
    return [t[b] for b in buf]


# The engine primitive, in software. For each vector v and output channel o:
#   acc = bias[o] + sum_k (x[v*xstride + (o*K if dw else 0) + k] + in_off) * w[o*K + k]
#   y[v*N + o] = requant(acc)          (or the raw int32 accumulators when raw=True)
# x and w are int8 lists, params is the packed 16-byte-per-channel parameter block.
def dot_requant(x, w, params, K, N, nvec, xstride, in_off, out_zp, act_min, act_max, dw=False, raw=False):
    # Output list.
    y = []
    # Decode the per-channel parameters once.
    prm = [unpack_param(params, o * PARAM_BYTES) for o in range(N)]
    # Loop over the input vectors.
    for v in range(nvec):
        # Start of this vector in x.
        xb = v * xstride
        # Loop over the output channels.
        for o in range(N):
            # Parameters of this channel.
            bias, mult, shift = prm[o]
            # Start of this channel's weight row.
            wo = o * K
            # Start of the x slice this channel uses (depthwise: its own slice).
            xo = xb + (wo if dw else 0)
            # Accumulator starts at the bias.
            acc = bias
            # Dot product.
            for k in range(K):
                # One multiply-accumulate.
                acc += (x[xo + k] + in_off) * w[wo + k]
            # int32 wrap, as on the hardware.
            acc = wrap32(acc)
            # Keep raw accumulators or requantise.
            y.append(acc if raw else requant(acc, mult, shift, out_zp, act_min, act_max))
    # Flat list of NVEC*N outputs.
    return y


# Output size and leading padding for one spatial axis (TFLite ComputePaddingHeightWidth).
def out_and_pad(in_size, k, stride, dil, same):
    # Filter extent with dilation.
    eff = (k - 1) * dil + 1
    # SAME keeps ceil(in/stride) outputs, VALID only full windows.
    out = (in_size + stride - 1) // stride if same else (in_size - eff + stride) // stride
    # Total padding needed (SAME only).
    total = max((out - 1) * stride + eff - in_size, 0) if same else 0
    # TFLite puts the smaller half before.
    return out, total // 2


# Gather the im2col patch of one conv output pixel: order (ky, kx, ci), padding = pad_val.
def gather_conv(x, H, W, C, oy, ox, kh, kw, sh, sw, dh, dw_, ph, pw, pad_val, out):
    # Top-left input row of the window.
    y0 = oy * sh - ph
    # Top-left input column of the window.
    x0 = ox * sw - pw
    # Window rows.
    for ky in range(kh):
        # Input row of this filter row.
        iy = y0 + ky * dh
        # Window columns.
        for kx in range(kw):
            # Input column of this filter column.
            ix = x0 + kx * dw_
            # Inside the image: copy C channels.
            if 0 <= iy < H and 0 <= ix < W:
                # Offset of pixel (iy, ix) in NHWC.
                b = (iy * W + ix) * C
                # Append its channels.
                out.extend(x[b:b + C])
            else:
                # Padding: zero point, so (x + in_off) == 0 and padding adds nothing
                # (bytes, not a list: MicroPython's bytearray.extend needs a buffer).
                out.extend(bytes([pad_val]) * C)


# Gather the depthwise patch of one output pixel: order (oc, ky, kx), input channel oc // mult.
def gather_dw(x, H, W, C, mult, oy, ox, kh, kw, sh, sw, dh, dw_, ph, pw, pad_val, out):
    # Top-left input row of the window.
    y0 = oy * sh - ph
    # Top-left input column of the window.
    x0 = ox * sw - pw
    # One slice per output channel.
    for oc in range(C * mult):
        # Input channel feeding this output channel.
        ic = oc // mult
        # Window rows.
        for ky in range(kh):
            # Input row.
            iy = y0 + ky * dh
            # Window columns.
            for kx in range(kw):
                # Input column.
                ix = x0 + kx * dw_
                # Real pixel or padding.
                out.append(x[(iy * W + ix) * C + ic] if (0 <= iy < H and 0 <= ix < W) else pad_val)


# int8 max / average pooling (TFLite reference): padding is excluded from the window.
def pool(x, H, W, C, kh, kw, sh, sw, same, is_max, act_min, act_max):
    # Output height and top padding.
    OH, ph = out_and_pad(H, kh, sh, 1, same)
    # Output width and left padding.
    OW, pw = out_and_pad(W, kw, sw, 1, same)
    # Output list.
    y = []
    # Output rows.
    for oy in range(OH):
        # Clip the window to the image (rows).
        ys, ye = max(oy * sh - ph, 0), min(oy * sh - ph + kh, H)
        # Output columns.
        for ox in range(OW):
            # Clip the window to the image (columns).
            xs, xe = max(ox * sw - pw, 0), min(ox * sw - pw + kw, W)
            # Channels.
            for c in range(C):
                # Values in the window.
                vals = [x[(iy * W + ix) * C + c] for iy in range(ys, ye) for ix in range(xs, xe)]
                # Max pooling.
                if is_max:
                    # Largest value.
                    r = max(vals)
                else:
                    # Sum and count of the window.
                    acc, n = sum(vals), len(vals)
                    # Round half away from zero with C truncating division.
                    r = (acc + n // 2) // n if acc > 0 else -((-(acc - n // 2)) // n)
                # Clamp to the fused activation range.
                y.append(min(max(r, act_min), act_max))
    # Flat NHWC list and output shape.
    return y, OH, OW


# Round half away from zero (TfLiteRound / std::round).
def round_away(f):
    # Positive values.
    if f >= 0:
        return int(f + 0.5)
    # Negative values.
    return -int(-f + 0.5)


# Quantise floats to int8 with (scale, zero point).
def quantize(vals, scale, zp):
    # Clamp each rounded value to the int8 range.
    return [min(max(round_away(v / scale) + zp, -128), 127) for v in vals]


# Dequantise int8 values to floats.
def dequantize(vals, scale, zp):
    # Affine map back to real values.
    return [(v - zp) * scale for v in vals]


# Float softmax of dequantised logits (not bit-exact with TFLite's int8 LUT softmax; argmax always matches).
def softmax(vals, beta=1.0):
    # Import here so the module loads even where math is trimmed.
    import math
    # Subtract the max for numerical safety.
    m = max(vals)
    # Exponentials.
    e = [math.exp((v - m) * beta) for v in vals]
    # Normaliser.
    s = sum(e)
    # Probabilities.
    return [v / s for v in e]
