# transform.py - PC side: rewrite the weights of a converted .tmf model.
#   to_int4(tmf, layers)      re-quantise chosen layers to signed 4-bit weights (engine builds with HAS_W4)
#   to_codebook(tmf, layers)  replace chosen layers' weights by 4-bit indices into a 16-value table found by k-means
#                             (engine builds with HAS_CB; the same size as int4 but a better fit to the weights)
# The result is a normal .tmf file with "wbits": 4 on the changed ops; the runtime and the engine take care of the rest.
# (Not bit-exact with the int8 model by nature: compare() measures how far the outputs move.)

import json
import math
import struct

from . import kernels as kn
from .converter import quantize_multiplier

# Ops that carry weights.
WEIGHT_OPS = ("conv", "dw", "fc")


# Split .tmf bytes into (header dict, blob bytes).
def read_tmf(data):
    # Header length.
    hl = struct.unpack_from("<I", data, 4)[0]
    # JSON header and the binary blob behind it.
    return json.loads(data[8:8 + hl]), bytes(data[8 + hl:])


# Join a header and a blob into .tmf bytes.
def write_tmf(header, blob):
    # Compact JSON.
    hj = json.dumps(header, separators=(",", ":")).encode()
    return b"TMF1" + struct.pack("<I", len(hj)) + hj + bytes(blob)


# Pick the engine ops a layer selection names: "all", "auto" (all but the first and last weight layer), or a list of
# weight-layer ordinals (0 = first conv / dw / fc).
def select_layers(header, layers):
    # Ordinals of the weight layers.
    idx = [i for i, op in enumerate(header["ops"]) if op["op"] in WEIGHT_OPS]
    if layers == "all":
        return idx
    if layers == "auto":
        return idx[1:-1]
    return [idx[int(n)] for n in layers]


# int8 rows -> int4 rows plus the per-row scale r (w8 ~ r * w4), as (packed bytes, list of r).
def _quantize_rows(w8, K, N):
    out, scales = bytearray(), []
    for o in range(N):
        # Row and its largest magnitude.
        row = [w8[o * K + k] for k in range(K)]
        m = max(abs(v) for v in row)
        # All-zero rows stay zero with unit scale.
        r = m / 7.0 if m else 1.0
        scales.append(r)
        # Values in -7 .. 7, two per byte (low nibble = even index); odd K leaves the last high nibble zero.
        q = [max(-8, min(7, kn.round_away(v / r))) for v in row]
        if K & 1:
            q.append(0)
        for k in range(0, len(q), 2):
            out.append((q[k] & 0xF) | ((q[k + 1] & 0xF) << 4))
    return bytes(out), scales


# Re-quantise the selected layers (see select_layers) to 4-bit weights. Returns (tmf bytes, report); the report has
# one dict per changed layer with the weight RMS error relative to the weight RMS and the bytes saved.
def to_int4(tmf, layers="auto"):
    header, blob = read_tmf(tmf)
    chosen = set(select_layers(header, layers))
    new, report = bytearray(), []
    for i, op in enumerate(header["ops"]):
        if op["op"] not in WEIGHT_OPS:
            continue
        # Existing weight and parameter blobs.
        w = blob[op["w"][0]:op["w"][0] + op["w"][1]]
        p = blob[op["p"][0]:op["p"][0] + op["p"][1]]
        if i in chosen and op.get("wbits", 8) == 8 and not op.get("cb"):
            K, N = op["K"], op["N"]
            w8 = kn.to_i8_list(w)
            w4, scales = _quantize_rows(w8, K, N)
            # Fold the row scales into bias and multiplier: acc8 ~ r * acc4.
            p2 = bytearray()
            for o in range(N):
                bias, mult, shift = kn.unpack_param(p, o * kn.PARAM_BYTES)
                r = scales[o]
                real = mult * 2.0 ** (shift - 31) * r
                m2, s2 = quantize_multiplier(real) if mult else (0, 0)
                b2 = max(-2 ** 31, min(2 ** 31 - 1, kn.round_away(bias / r)))
                p2 += kn.pack_param(b2, m2, s2)
            # Weight error relative to the signal.
            un = kn.unpack_w4(w4, K, N)
            err = sum((w8[o * K + k] - scales[o] * un[o * K + k]) ** 2 for o in range(N) for k in range(K))
            sig = sum(v * v for v in w8) or 1
            report.append({"op": i, "kind": op["op"], "K": K, "N": N, "rel_rms_error": math.sqrt(err / sig),
                           "bytes_saved": len(w) - len(w4)})
            w, p = w4, bytes(p2)
            op["wbits"] = 4
        # Re-pack both blobs.
        op["w"] = [len(new), len(w)]
        new += w
        op["p"] = [len(new), len(p)]
        new += p
        # A layer that already has a codebook keeps it.
        if op.get("cb"):
            old = op["cb"]
            op["cb"] = [len(new), 16]
            new += blob[old[0]:old[0] + 16]
    return write_tmf(header, new), report


# Best 16-value int8 table for a layer's weights (1-D k-means / Lloyd on the value histogram), sorted.
def _kmeans16(w8):
    # Histogram of the 256 possible values.
    hist = [0] * 256
    for v in w8:
        hist[v + 128] += 1
    vals = [i - 128 for i in range(256) if hist[i]]
    # Fewer than 17 distinct values: the table holds them all exactly.
    if len(vals) <= 16:
        return sorted(vals + [vals[-1]] * (16 - len(vals)))
    # Start from evenly spaced quantiles of the weights.
    srt = sorted(w8)
    cent = [srt[int((i + 0.5) * len(srt) / 16)] for i in range(16)]
    for _ in range(60):
        # Assign each value to its nearest centroid and average.
        sums, cnt = [0] * 16, [0] * 16
        for v in vals:
            j = min(range(16), key=lambda c: (abs(v - cent[c]), c))
            sums[j] += v * hist[v + 128]
            cnt[j] += hist[v + 128]
        new = [kn.round_away(sums[j] / cnt[j]) if cnt[j] else cent[j] for j in range(16)]
        if new == cent:
            break
        cent = new
    return sorted(cent)


# Replace the weights of the selected layers (see select_layers) by 4-bit indices into a per-layer 16-entry table.
# Returns (tmf bytes, report) like to_int4; the biases and multipliers stay as they are.
def to_codebook(tmf, layers="auto"):
    header, blob = read_tmf(tmf)
    chosen = set(select_layers(header, layers))
    new, report = bytearray(), []
    for i, op in enumerate(header["ops"]):
        if op["op"] not in WEIGHT_OPS:
            continue
        w = blob[op["w"][0]:op["w"][0] + op["w"][1]]
        p = blob[op["p"][0]:op["p"][0] + op["p"][1]]
        cbb = None
        if i in chosen and op.get("wbits", 8) == 8 and not op.get("cb"):
            K, N = op["K"], op["N"]
            w8 = kn.to_i8_list(w)
            cb = _kmeans16(w8)
            # Index of the nearest table entry for every weight.
            idx = [min(range(16), key=lambda c: (abs(v - cb[c]), c)) for v in w8]
            packed = bytearray()
            for o in range(N):
                q = idx[o * K:(o + 1) * K] + ([0] if K & 1 else [])
                for k in range(0, len(q), 2):
                    packed.append(q[k] | (q[k + 1] << 4))
            err = sum((v - cb[j]) ** 2 for v, j in zip(w8, idx))
            sig = sum(v * v for v in w8) or 1
            report.append({"op": i, "kind": op["kind"] if "kind" in op else op["op"], "K": K, "N": N,
                           "rel_rms_error": math.sqrt(err / sig), "bytes_saved": len(w) - len(packed) - 16})
            cbb = bytes(v & 0xFF for v in cb)
            w = bytes(packed)
        op["w"] = [len(new), len(w)]
        new += w
        op["p"] = [len(new), len(p)]
        new += p
        if cbb is not None:
            op["cb"] = [len(new), 16]
            new += cbb
        elif op.get("cb"):
            old = op["cb"]
            op["cb"] = [len(new), 16]
            new += blob[old[0]:old[0] + 16]
    return write_tmf(header, new), report


# How far do two models' outputs differ on the given inputs? Runs both in software.
# Returns {"max_abs": largest output difference (int8 steps or probability), "mean_abs": mean, "same_top1": fraction}.
def compare(tmf_a, tmf_b, inputs):
    from .model import Model
    a, b = Model(tmf_a), Model(tmf_b)
    worst, total, n, same = 0.0, 0.0, 0, 0
    for x in inputs:
        ya, yb = a.invoke(bytearray(x)), b.invoke(bytearray(x))
        if isinstance(ya, (bytes, bytearray)):
            ya, yb = kn.to_i8_list(ya), kn.to_i8_list(yb)
        d = [abs(u - v) for u, v in zip(ya, yb)]
        worst, total, n = max(worst, max(d)), total + sum(d), n + len(d)
        same += max(range(len(ya)), key=lambda i: ya[i]) == max(range(len(yb)), key=lambda i: yb[i])
    return {"max_abs": worst, "mean_abs": total / max(n, 1), "same_top1": same / max(len(inputs), 1)}
