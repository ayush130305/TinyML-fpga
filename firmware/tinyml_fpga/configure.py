# configure.py - pick engine features for a LUT budget and generate the RTL package (PC only).
#   python -m tinyml_fpga configure --budget 8k --bus spi --features win,fuse,w4 --out build.zip
#   python -m tinyml_fpga configure --budget 4k --suggest
# Everything is driven by configurator/spec.json (built by tools/build_spec.py from the Yosys sweep and the RTL), and
# the web page (docs/index.html) runs the same algorithm in JavaScript from the same spec; tools/crosscheck.py compares
# the two. Cost numbers are Yosys iCE40 LUT4 counts scaled to Efinity logic elements (tools/calibration.json): an
# estimate to plan with, not a place-and-route result.

import json
import os
import re
import zipfile

# Repository root (this file is firmware/tinyml_fpga/configure.py).
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# Default spec location.
SPEC_PATH = os.path.join(ROOT, "configurator", "spec.json")
# Cost metrics in the spec.
METRICS = ("lut4", "lut4_nodsp", "ff", "dsp", "ram4k")
# Fraction of the budget a configuration may use and still be called "fits" (routing and the rest of the design need room).
FIT_LIMIT = 0.85
# RAM address-width defaults and limits (name, default, low, high).
RAM = (("W_AW", 12, 8, 15), ("X_AW", 10, 6, 12), ("Y_AW", 8, 4, 12), ("P_AW", 10, 6, 12), ("D_AW", 9, 7, 12))


# Load the spec.
def load_spec(path=SPEC_PATH):
    with open(path) as f:
        return json.load(f)


# Round half up (the same in Python and JavaScript).
def _round(x):
    return int(x + 0.5)


# Fill in defaults and check a configuration; returns the cleaned dict or raises ValueError.
#   lanes int, rq 0/1, features list of ids, bus, dsp bool, budget int (LEs), options dict
def normalize(spec, cfg):
    c = {
        "lanes": int(cfg.get("lanes", 1)),
        "rq": int(cfg.get("rq", 0)),
        "features": sorted(set(cfg.get("features", []))),
        "bus": cfg.get("bus", "uart"),
        "dsp": bool(cfg.get("dsp", True)),
        "budget": int(cfg.get("budget", 8000)),
        "options": dict(cfg.get("options", {})),
    }
    if c["lanes"] not in spec["lanes"]:
        raise ValueError("lanes must be one of %s" % spec["lanes"])
    if c["rq"] not in (0, 1):
        raise ValueError("rq must be 0 (serial) or 1 (pipelined)")
    ids = [f["id"] for f in spec["features"]]
    for f in c["features"]:
        if f not in ids:
            raise ValueError("unknown feature %s" % f)
    if c["bus"] not in spec["buses"]:
        raise ValueError("bus must be one of %s" % list(spec["buses"]))
    o = c["options"]
    for k, d, lo, hi in RAM:
        o[k] = int(o.get(k, d))
        if not lo <= o[k] <= hi:
            raise ValueError("%s must be %d..%d" % (k, lo, hi))
    o["clk_hz"] = int(o.get("clk_hz", 100000000))
    o["baud"] = int(o.get("baud", 921600))
    o["i2c_addr"] = int(o.get("i2c_addr", 0x2A))
    if not 8 <= o["i2c_addr"] <= 119:
        raise ValueError("i2c_addr must be 8..119")
    return c


# Cost of a configuration: dict of the metrics plus "le" (estimated logic elements) and "ff_est".
def estimate(spec, cfg):
    c = normalize(spec, cfg)
    l, r = str(c["lanes"]), str(c["rq"])
    tot = {m: spec["base"][l + "," + r][m] for m in METRICS}
    # Feature deltas at this lane count.
    for f in c["features"]:
        for m in METRICS:
            tot[m] += spec["delta"][l][f][m]
    # Pair corrections (measured at 1 and 4 lanes; 2 lanes use the 1-lane ones, 8 lanes the 4-lane ones).
    pl = "1" if c["lanes"] <= 2 else "4"
    fs = c["features"]
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            corr = spec["pair"][pl].get(fs[i] + "+" + fs[j])
            if corr:
                for m in METRICS:
                    tot[m] += corr[m]
    # The bridge.
    for m in METRICS:
        tot[m] += spec["buses"][c["bus"]]["cost"][m]
    lut = tot["lut4"] if c["dsp"] else tot["lut4_nodsp"]
    out = {m: max(0, tot[m]) for m in METRICS}
    out["lut"] = max(0, lut)
    out["le"] = _round(max(0, lut) * spec["le_per_lut4"])
    out["ff_est"] = _round(max(0, tot["ff"]) * spec["ff_per_ff"])
    return out


# "fits" up to FIT_LIMIT of the budget, "tight" up to the budget, else "over".
def verdict(est, budget):
    if est["le"] <= FIT_LIMIT * budget:
        return "fits"
    return "tight" if est["le"] <= budget else "over"


# Warnings about the choices that cost nothing in LUTs but matter on the board.
def warnings(spec, cfg):
    c = normalize(spec, cfg)
    o, w = c["options"], []
    if (1 << o["P_AW"]) // 16 < c["lanes"]:
        w.append("P_AW=%d leaves less than one parameter record per lane" % o["P_AW"])
    if "HAS_QUEUE" in c["features"] and (1 << o["D_AW"]) < 64:
        w.append("D_AW too small for one descriptor")
    if "HAS_CB" in c["features"] and "HAS_W4" not in c["features"]:
        w.append("HAS_CB also stores 4-bit indices; HAS_W4 is not needed with it")
    if c["bus"] == "spi":
        w.append("SPI clock must be at most clk/6")
    if c["bus"] == "i2c":
        w.append("I2C clock must be at most clk/16; no clock stretching")
    return w


# Set the value of `parameter NAME = value` in Verilog source text.
def _set_param(text, name, value):
    return re.sub(r"(parameter\s+" + name + r"\s*=\s*)([^,\s/]+)", lambda m: m.group(1) + str(value), text, count=1)


# Names of the files in the package, in order: (source name, folder).
def package_files(spec, cfg):
    c = normalize(spec, cfg)
    b = spec["buses"][c["bus"]]
    names = list(spec["core"]) + list(spec["rq_files"][str(c["rq"])]) + list(b["files"]) + [b["top"]]
    return names


# The generated top (module tinyml_*_top) with the chosen parameters as defaults.
def top_source(spec, cfg):
    c = normalize(spec, cfg)
    b, o = spec["buses"][c["bus"]], c["options"]
    t = spec["sources"][b["top"]]
    vals = {"LANES": c["lanes"], "RQ_MODE": c["rq"], "W_AW": o["W_AW"], "X_AW": o["X_AW"], "Y_AW": o["Y_AW"], "P_AW": o["P_AW"], "D_AW": o["D_AW"]}
    for f in spec["features"]:
        vals[f["id"]] = 1 if f["id"] in c["features"] else 0
    if c["bus"] == "uart":
        vals["CLK_HZ"], vals["BAUD"] = o["clk_hz"], o["baud"]
    if c["bus"] == "i2c":
        vals["I2C_ADDR"] = "7'h%02X" % o["i2c_addr"]
    for k in sorted(vals):
        t = _set_param(t, k, vals[k])
    return t


# Python line that opens the chosen bus.
def python_line(c):
    o = c["options"]
    return {"uart": 'tinyml_fpga.load("model.tmf", backend="uart:COM5", baud=%d)' % o["baud"],
            "spi": 'tinyml_fpga.load("model.tmf", backend="spi:0.0")',
            "i2c": 'tinyml_fpga.load("model.tmf", backend="i2c:1:0x%02X")' % o["i2c_addr"],
            "axil": 'tinyml_fpga.load("model.tmf", backend="axi:0x40000000")'}[c["bus"]]


# README of the package.
def readme(spec, cfg):
    c = normalize(spec, cfg)
    e = estimate(spec, c)
    b = spec["buses"][c["bus"]]
    o = c["options"]
    feats = ", ".join(c["features"]) or "none"
    ram_bytes = {"W": 1 << o["W_AW"], "X": 1 << o["X_AW"], "Y": 1 << o["Y_AW"], "P": 1 << o["P_AW"]}
    lines = [
        "# tinyml-fpga configured engine",
        "",
        "Generated by the tinyml-fpga configurator (version %s)." % spec["version"],
        "",
        "## Configuration",
        "",
        "- lanes: %d" % c["lanes"],
        "- requantiser: %s" % ("pipelined" if c["rq"] else "serial"),
        "- features: %s" % feats,
        "- bus: %s (top module `%s`)" % (c["bus"], b["top"][:-2]),
        "- RAM: W %d B, X %d B, Y %d B, P %d B" % (ram_bytes["W"], ram_bytes["X"], ram_bytes["Y"], ram_bytes["P"]),
        "",
        "## Estimated size",
        "",
        "- about %d logic elements (budget %d, %s)" % (e["le"], c["budget"], verdict(e, c["budget"])),
        "- about %d flip-flops, %d multiplier blocks, %d 4-kbit RAM blocks" % (e["ff_est"], e["dsp"], e["ram4k"]),
        "- %s" % ("multipliers on DSP blocks" if c["dsp"] else "multipliers built from LUTs"),
        "",
        "Estimate from Yosys iCE40 counts scaled to Efinity Trion (calibrated on one design). Run your own synthesis for the real number.",
        "",
        "## Use",
        "",
        "1. Add every file in `rtl/` to your FPGA project and make `%s` the top module." % b["top"][:-2],
        "2. Connect its ports in your pin constraints (clock, reset, the bus pins, optional status LEDs).",
        "3. On the host: `pip install -e .` from the tinyml-fpga repository, convert a model with `python -m tinyml_fpga convert`, then",
        "",
        "```python",
        "import tinyml_fpga",
        "m = " + python_line(c),
        "```",
        "",
    ]
    for w in warnings(spec, c):
        lines.append("- note: " + w)
    return "\n".join(lines) + "\n"


# All files of the package as a list of (path, text).
def build_files(spec, cfg):
    c = normalize(spec, cfg)
    b = spec["buses"][c["bus"]]
    out = []
    for n in package_files(spec, c):
        out.append(("rtl/" + n, top_source(spec, c) if n == b["top"] else spec["sources"][n]))
    out.append(("config.json", json.dumps(c, indent=2, sort_keys=True) + "\n"))
    out.append(("README.md", readme(spec, c)))
    return out


# Write the package to a zip file (fixed timestamps so the same configuration gives the same bytes).
def write_zip(spec, cfg, path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in build_files(spec, cfg):
            zi = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, text)


# Every configuration (lanes, rq, feature subset) that fits the budget, best first by a simple speed score:
# lanes times 1.5 with the pipelined requantiser, then more features. `must` lists features that are required.
def suggest(spec, budget, bus="uart", dsp=True, must=(), options=None):
    ids = [f["id"] for f in spec["features"]]
    best, best_key = None, None
    for lanes in spec["lanes"]:
        for rq in (0, 1):
            for mask in range(1 << len(ids)):
                fs = [ids[i] for i in range(len(ids)) if mask >> i & 1]
                if any(m not in fs for m in must):
                    continue
                cfg = {"lanes": lanes, "rq": rq, "features": fs, "bus": bus, "dsp": dsp, "budget": budget, "options": options or {}}
                e = estimate(spec, cfg)
                if verdict(e, budget) != "fits":
                    continue
                key = (lanes * (1.5 if rq else 1.0), len(fs), -e["le"])
                if best_key is None or key > best_key:
                    best, best_key = cfg, key
    return best


# Parse "8k", "20000", "1.5k" to an integer.
def parse_budget(s):
    s = str(s).strip().lower()
    return int(float(s[:-1]) * 1000) if s.endswith("k") else int(s)


# Short feature names for the command line (win -> HAS_WIN).
def parse_features(spec, s):
    ids = [f["id"] for f in spec["features"]]
    out = []
    for p in [x.strip() for x in s.split(",") if x.strip()]:
        full = p.upper() if p.upper().startswith("HAS_") else "HAS_" + p.upper()
        if full not in ids:
            raise ValueError("unknown feature %r (choose from %s)" % (p, ", ".join(i[4:].lower() for i in ids)))
        out.append(full)
    return out


# Command line entry (called from __main__).
def main(a):
    spec = load_spec()
    budget = parse_budget(a.budget)
    feats = parse_features(spec, a.features or "")
    if a.suggest:
        cfg = suggest(spec, budget, a.bus, not a.no_dsp, feats)
        if cfg is None:
            print("nothing fits %d LEs with those features" % budget)
            return 1
    else:
        cfg = {"lanes": a.lanes, "rq": 1 if a.pipelined else 0, "features": feats, "bus": a.bus, "dsp": not a.no_dsp, "budget": budget}
    cfg["budget"] = budget
    c = normalize(spec, cfg)
    e = estimate(spec, c)
    print("lanes %d, %s requantiser, bus %s, features: %s" % (c["lanes"], "pipelined" if c["rq"] else "serial", c["bus"], ", ".join(f[4:].lower() for f in c["features"]) or "none"))
    print("about %d LEs of %d (%.0f%%): %s; %d FF, %d multipliers, %d RAM blocks" % (
        e["le"], budget, 100.0 * e["le"] / budget, verdict(e, budget), e["ff_est"], e["dsp"], e["ram4k"]))
    for w in warnings(spec, c):
        print("note:", w)
    if a.out:
        write_zip(spec, c, a.out)
        print("wrote", a.out)
    return 0 if verdict(e, budget) != "over" else 2
