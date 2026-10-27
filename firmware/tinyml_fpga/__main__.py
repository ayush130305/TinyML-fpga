# __main__.py - PC command line:
#   python -m tinyml_fpga convert model.tflite [model.tmf]    convert, then check it against LiteRT
#   python -m tinyml_fpga info    model.tmf [--sizes W X Y P] ops, sizes, where each op runs
#   python -m tinyml_fpga check   model.tflite [--n 50]       bit-exact comparison with LiteRT
#   python -m tinyml_fpga deploy  [model.tmf ...] [--port P]  copy the runtime (and models) to a MicroPython MCU

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
    # Verify unless told not to.
    if not a.no_check:
        a.n = 20
        cmd_check(a)


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
    # Run.
    a = ap.parse_args()
    a.fn(a)


# Script entry.
if __name__ == "__main__":
    main()
