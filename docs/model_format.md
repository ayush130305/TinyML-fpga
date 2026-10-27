# .tmf model format

```
"TMF1" | u32 header_len (LE) | header (compact JSON, UTF-8) | blob
```

The board reads the JSON header with MicroPython's `json` and seeks into the blob only when it uploads weights, so a model never has to fit in RAM twice.

## Header

```json
{
  "version": 1,
  "input": 0, "output": 7,
  "tensors": [ {"shape": [1,8,8,1], "dtype": "int8", "scale": 0.0039, "zp": -128}, ... ],
  "ops": [ {"op": "conv", "in": 0, "out": 1, "K": 9, "N": 8, "w": [0, 72], "p": [72, 128],
            "act": [-128, 127], "H": 8, "W": 8, "C": 1, "OH": 8, "OW": 8,
            "kh": 3, "kw": 3, "sh": 1, "sw": 1, "dh": 1, "dw": 1, "ph": 1, "pw": 1}, ... ]
}
```

| op | fields |
|---|---|
| `conv` | K = kh·kw·C, N = output channels, geometry, padding offsets `ph`/`pw`, `w`/`p` blob refs, `act` |
| `dw` | as conv with K = kh·kw and `mult` (depth multiplier) |
| `fc` | K, N, `nvec` (batch rows), `w`, `p`, `act` |
| `maxpool` / `avgpool` | H, W, C, kh, kw, sh, sw, `same`, `act` |
| `reshape` | – (data unchanged) |
| `softmax` | `beta` |
| `relu` | `mult`, `shift`, `act` (requantised clamp) |
| `quantize` / `dequantize` | – (use the tensors' scale / zp) |

`w` and `p` are `[offset, length]` into the blob:

- **weights:** int8, one row of K bytes per output channel (depthwise filters are transposed from TFLite's `[1, kh, kw, OC]` to `[OC][kh·kw]`)
- **params:** 16 bytes per output channel: bias i32, multiplier i32, shift i8, 7 zero bytes (little endian). The multiplier and shift come from TFLite's `QuantizeMultiplier` on `in_scale · w_scale[c] / out_scale`, computed the way the TFLite kernels compute it.

Ops that only compute shapes (Keras' `SHAPE` / `STRIDED_SLICE` / `PACK` before a flatten) are pruned by the converter.
