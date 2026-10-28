# __main__.py - PC command line:
#   python -m tinyml_fpga convert model.tflite [model.tmf]    convert, then check it against LiteRT
#   python -m tinyml_fpga info    model.tmf [--sizes W X Y P] ops, sizes, where each op runs
#   python -m tinyml_fpga check   model.tflite [--n 50]       bit-exact comparison with LiteRT
#   python -m tinyml_fpga deploy  [model.tmf ...] [--port P]  copy the runtime (and models) to a MicroPython MCU
#   python -m tinyml_fpga configure --budget 8k [--suggest | --lanes N --features a,b] [--bus B] [--out pkg.zip]

import argparse
import os
import subprocess
import sys

# Files a MicroPython MCU needs (the PC-only converter/checker stay on the PC).
BOARD_FILES = ("__init__.py", "model.py", "kernels.py", "engine.py", "buses.py")
# Default engine RAM address widths (W, X, Y, P) = the defaults of rtl/tmf_engine.v.
DEFAULT_SIZES = (12, 10, 8, 10)


# Random int8 samples shaped like the model input (used when no data is given).
def _random_samples(tflite_path, n):
    # numpy only on the PC.
    import numpy as np
    from .check import _interpreter
    # Input shape from LiteRT.
    with open(tflite_path, "rb") as f:
        shape = _interpreter(f.read()).get_input_details()[0]["shape"]
    # Reproducible random data.
    rng = np.random.default_rng(0)
    # n samples.
    return [rng.integers(-128, 128, size=shape, dtype=np.int8) for _ in range(n)]


# Convert and verify.
def cmd_convert(a):
    # Output name next to the input by default.
    dst = a.output or os.path.splitext(a.model)[0] + ".tmf"
    # Convert.
    from .converter import convert_file
    convert_file(a.model, dst)
    # Size report.
    print("wrote %s (%d bytes)" % (dst, os.path.getsize(dst)))
    # Verify unless told not to (always the int8 conversion, which is bit-exact with LiteRT).
    if not a.no_check:
        a.n = 20
        cmd_check(a)
    # Optional 4-bit weights (HAS_W4 builds) or codebook weights (HAS_CB builds).
    if a.int4 or a.codebook:
        from .transform import to_int4, to_codebook, compare
        arg = a.int4 or a.codebook
        sel = arg if arg in ("all", "auto") else [int(v) for v in arg.split(",")]
        with open(dst, "rb") as f:
            t8 = f.read()
        t4, rep = (to_int4 if a.int4 else to_codebook)(t8, sel)
        with open(dst, "wb") as f:
            f.write(t4)
        print("wrote %s with %s weights (%d bytes, %d layers):" % (dst, "4-bit" if a.int4 else "codebook", len(t4), len(rep)))
        for r in rep:
            print("  op %d %-4s K=%-5d N=%-4d weight error %.1f%% of RMS, %d weight bytes saved" % (
                r["op"], r["kind"], r["K"], r["N"], 100 * r["rel_rms_error"], r["bytes_saved"]))
        # How far do the outputs move? (random inputs unless --samples gives real ones)
        xs = [bytearray(int(v) & 0xFF for v in s.reshape(-1)) for s in _random_samples(a.model, 20)]
        c = compare(t8, t4, xs)
        print("  vs int8 on %d random inputs: mean |diff| %.4f, max |diff| %.4f, same top-1 %.0f%%  (random inputs are a harsh "
              "test; check with real data before deploying)" % (len(xs), c["mean_abs"], c["max_abs"], 100 * c["same_top1"]))


# Summary on an engine with the given RAM sizes.
def cmd_info(a):
    # Runtime pieces.
    from . import Model, HwEngine, SimBus
    # Engine model with these address widths.
    w, x, y, p = a.sizes
    # Load and print.
    Model(a.model, HwEngine(SimBus(w, x, y, p))).summary()


# Bit-exact check against LiteRT.
def cmd_check(a):
    # Comparison helper.
    from .check import compare
    # Random int8 inputs (every int8 tensor must match regardless of the data).
    res = compare(a.model, _random_samples(a.model, a.n), backend="sim")
    # Exit code for scripts.
    if not res["exact"]:
        sys.exit(1)


# Copy the runtime (and optional .tmf files) to a MicroPython board with mpremote.
def cmd_deploy(a):
    # Package directory.
    here = os.path.dirname(os.path.abspath(__file__))
    # mpremote prefix (optionally with a port).
    mp = [sys.executable, "-m", "mpremote"] + (["connect", a.port] if a.port else [])
    # Make the package folder (ignore "exists").
    subprocess.run(mp + ["mkdir", ":tinyml_fpga"], check=False)
    # Copy the runtime files.
    for f in BOARD_FILES:
        subprocess.run(mp + ["cp", os.path.join(here, f), ":tinyml_fpga/" + f], check=True)
    # Copy the models to the board root.
    for m in a.models:
        subprocess.run(mp + ["cp", m, ":" + os.path.basename(m)], check=True)
    # Done.
    print("copied tinyml_fpga runtime%s to the board" % (" and " + ", ".join(a.models) if a.models else ""))


# Argument parsing and dispatch.
def main():
    # Top-level parser.
    ap = argparse.ArgumentParser(prog="python -m tinyml_fpga")
    sub = ap.add_subparsers(dest="cmd", required=True)
    # convert.
    c = sub.add_parser("convert", help=".tflite -> .tmf, then check against LiteRT")
    c.add_argument("model"); c.add_argument("output", nargs="?")
    c.add_argument("--no-check", action="store_true")
    c.add_argument("--codebook", metavar="LAYERS", help="codebook-compressed weights for engine builds with HAS_CB (same "
                   "choices as --int4)")
    c.add_argument("--int4", metavar="LAYERS", help="4-bit weights for engine builds with HAS_W4: auto (all but first and last "
                   "layer), all, or weight-layer numbers like 1,2")
    c.set_defaults(fn=cmd_convert)
    # info.
    i = sub.add_parser("info", help="ops, sizes and placement on an engine build")
    i.add_argument("model")
    i.add_argument("--sizes", type=int, nargs=4, default=list(DEFAULT_SIZES), metavar=("W_AW", "X_AW", "Y_AW", "P_AW"))
    i.set_defaults(fn=cmd_info)
    # check.
    k = sub.add_parser("check", help="bit-exact comparison with LiteRT")
    k.add_argument("model"); k.add_argument("--n", type=int, default=50)
    k.set_defaults(fn=cmd_check)
    # deploy.
    d = sub.add_parser("deploy", help="copy the runtime (and models) to a MicroPython board with mpremote")
    d.add_argument("models", nargs="*"); d.add_argument("--port")
    d.set_defaults(fn=cmd_deploy)
    # configure.
    g = sub.add_parser("configure", help="check features against a LUT budget and generate an RTL package")
    g.add_argument("--budget", required=True, help="logic elements available, e.g. 4k or 8k")
    g.add_argument("--bus", default="uart", choices=("uart", "spi", "i2c", "axil"))
    g.add_argument("--lanes", type=int, default=1, choices=(1, 2, 4, 8))
    g.add_argument("--pipelined", action="store_true", help="pipelined requantiser (4 multipliers)")
    g.add_argument("--features", help="comma list: queue,perf,win,fuse,w4,cb")
    g.add_argument("--suggest", action="store_true", help="pick the fastest configuration that fits (honours --features as required)")
    g.add_argument("--no-dsp", action="store_true", help="the device has no multiplier blocks: count them as LUTs")
    g.add_argument("--out", help="write the package zip here")
    g.set_defaults(fn=lambda a: sys.exit(__import__("tinyml_fpga.configure", fromlist=["main"]).main(a)))
    # Run.
    a = ap.parse_args()
    a.fn(a)


# Script entry.
if __name__ == "__main__":
    main()
