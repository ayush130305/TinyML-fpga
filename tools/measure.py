# measure.py - resource measurement of the RTL with Yosys (the numbers behind the configurator).
#
# synth(top, params) runs Yosys' iCE40 flow on the Verilog in rtl/ for a given top module and parameter
# set, and returns the cell counts. Multipliers are counted before mapping (each is one DSP block on a
# device that has them); lut4 is the LUT count with those multipliers removed, lut4_nodsp the count with
# them built from LUTs. (Yosys' own DSP mapping fails on some of our multiplier chains, so it is not used.)
#
#   lut4      4-input LUTs (adder sum bits included)              ff     flip-flops
#   dsp       multiplier blocks                                   lut4_nodsp  LUTs when no DSP is available
#   ram4k     4-kbit RAM blocks                                   carry  carry-chain cells
#
# Yosys' LUT counts are not any vendor's tool's counts. tools/calibration.json holds the scale factors that
# map them onto Efinity Trion numbers; they come from one measured design (see that file).
#
# usage: python3 tools/measure.py [top] [PARAM=VALUE ...]      e.g.  python3 tools/measure.py tmf_engine LANES=4 RQ_MODE=1

import json
import os
import re
import subprocess
import sys
import tempfile

# Repository root and RTL folder.
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RTL = os.path.join(ROOT, "rtl")


# All synthesisable RTL files (testbenches live in rtl/sim and are not included).
def rtl_files():
    # Every .v directly in rtl/.
    return sorted(os.path.join(RTL, f) for f in os.listdir(RTL) if f.endswith(".v"))


# Parse the "stat" report of Yosys into a dict of cell counts.
def _parse_stat(text):
    # Last block of cell lines.
    cells = {}
    for m in re.finditer(r"^\s+(\$?\w+)\s+(\d+)\s*$", text, re.M):
        cells[m.group(1)] = int(m.group(2))
    return cells


# Run Yosys on a script text; returns the log text.
def _run_script(script):
    # Temp script file.
    with tempfile.NamedTemporaryFile("w", suffix=".ys", delete=False) as f:
        f.write("\n".join(script) + "\n")
        name = f.name
    try:
        out = subprocess.run(["yosys", "-q", "-l", name + ".log", name], capture_output=True, text=True)
        log = open(name + ".log").read() if os.path.exists(name + ".log") else ""
        if out.returncode != 0:
            raise RuntimeError("yosys failed (exit %d):\n%s" % (out.returncode, out.stderr[-1500:] + out.stdout[-1500:] + log[-1500:]))
    finally:
        for n in (name, name + ".log"):
            if os.path.exists(n):
                os.remove(n)
    return log


# Script lines that read the design with parameter overrides and build the hierarchy under `top`.
def _read(top, params, files):
    # Read, parameters (before the hierarchy), hierarchy.
    script = ["read_verilog -sv " + " ".join(files)]
    for k, v in params.items():
        script.append("chparam -set %s %s %s" % (k, v, top))
    script.append("hierarchy -top %s" % top)
    return script


# Cell counts of the final "stat" report of a log.
def _final_stat(log):
    return _parse_stat(log[log.rindex("Number of cells"):])


# LUT4 cost of a stand-alone multiplier with the given output width (cache of unit costs).
_MUL_COST = {}


def mul_lut_cost(ywidth):
    # Operand widths by output width: 17 = the 9x8 MAC multiplier, 32 = 16x16, otherwise two equal halves.
    a, b = {17: (9, 8), 32: (16, 16)}.get(ywidth, (ywidth // 2, ywidth - ywidth // 2))
    if ywidth not in _MUL_COST:
        # A bare multiplier module.
        src = tempfile.NamedTemporaryFile("w", suffix=".v", delete=False)
        src.write("module m(input [%d:0] x, input [%d:0] y, output [%d:0] p); assign p = x * y; endmodule\n" % (a - 1, b - 1, a + b - 1))
        src.close()
        try:
            log = _run_script(["read_verilog %s" % src.name, "synth_ice40 -top m -flatten", "stat"])
        finally:
            os.remove(src.name)
        _MUL_COST[ywidth] = _final_stat(log).get("SB_LUT4", 0)
    return _MUL_COST[ywidth]


# Synthesise `top` with `params` and return the resource summary.
def synth(top="tmf_engine", params=None, files=None):
    # Defaults.
    params, files = params or {}, files or rtl_files()
    # Multipliers before technology mapping: "$mul_<output width>" cells.
    log = _run_script(_read(top, params, files) + ["proc", "flatten", "opt_expr", "opt_clean", "wreduce", "stat -width"])
    muls = {}
    for m in re.finditer(r"^\s+\$mul_(\d+)\s+(\d+)\s*$", log, re.M):
        muls[int(m.group(1))] = int(m.group(2))
    # Full flow with the multipliers built from LUTs.
    log = _run_script(_read(top, params, files) + ["synth_ice40 -top %s -flatten" % top, "stat"])
    cells = _final_stat(log)
    # Multipliers that a DSP block would take over.
    n_mul = sum(muls.values())
    lut_mul = sum(n * mul_lut_cost(w) for w, n in muls.items())
    return {
        "lut4": max(0, cells.get("SB_LUT4", 0) - lut_mul),
        "lut4_nodsp": cells.get("SB_LUT4", 0),
        "carry": cells.get("SB_CARRY", 0),
        "ff": sum(v for k, v in cells.items() if k.startswith("SB_DFF")),
        "dsp": n_mul,
        "ram4k": cells.get("SB_RAM40_4K", 0),
    }


# Command line.
if __name__ == "__main__":
    top = sys.argv[1] if len(sys.argv) > 1 and "=" not in sys.argv[1] else "tmf_engine"
    params = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
    print(json.dumps(synth(top, params)))
