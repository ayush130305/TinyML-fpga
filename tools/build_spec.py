# build_spec.py - fit the cost model from tools/sweep.json and write configurator/spec.json (cost tables, feature and
# bus descriptions, and the RTL sources the package generator copies). Prints how well the model predicts the random
# validation configurations, which are not used in the fit.
#   python3 tools/build_spec.py

import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "firmware"))
from tinyml_fpga import configure, __version__

# Feature descriptions for the page and the README.
FEATURES = [
    ("HAS_QUEUE", "Command queue", "Run a list of layer descriptors with one start; fewer round trips on slow links."),
    ("HAS_PERF", "Performance counters", "MAC, stall, output and job counters."),
    ("HAS_WIN", "Window generator", "The engine walks convolution windows itself (padding, stride, dilation); no host-side im2col."),
    ("HAS_FUSE", "Layer fusion", "Layer outputs stay in input RAM; a chain of layers runs from one start."),
    ("HAS_W4", "4-bit weights", "Two weights per byte; half the weight memory."),
    ("HAS_CB", "Codebook weights", "4-bit indices into a 16-entry table fitted to the weights; hardware decompression."),
]
# Buses: id -> (label, top, bridge files, sweep module).
BUSES = {
    "uart": ("UART", "tinyml_uart_top.v", ["uart_bus.v", "uart_rx.v", "uart_tx.v"], "uart_bus"),
    "spi": ("SPI slave", "tinyml_spi_top.v", ["spi_bus.v"], "spi_bus"),
    "i2c": ("I2C slave", "tinyml_i2c_top.v", ["i2c_bus.v"], "i2c_bus"),
    "axil": ("AXI4-Lite slave", "tinyml_axil_top.v", ["axil_bus.v"], "axil_bus"),
}
METRICS = configure.METRICS


# The metrics of a measurement as a dict.
def met(r):
    return {m: r[m] for m in METRICS}


# a - b for metric dicts.
def sub(a, b):
    return {m: a[m] - b[m] for m in METRICS}


def main():
    rows = json.load(open(os.path.join(HERE, "sweep.json")))
    ids = [f[0] for f in FEATURES]

    def key(r):
        p = r["params"]
        return (r["top"], p.get("LANES", 1), p.get("RQ_MODE", 0), tuple(sorted(k for k in p if k.startswith("HAS_"))))

    by = {key(r): r for r in rows}
    spec = {"version": __version__, "lanes": [1, 2, 4, 8], "budgets": [4000, 8000]}
    cal = json.load(open(os.path.join(HERE, "calibration.json")))
    spec["le_per_lut4"], spec["ff_per_ff"] = cal["le_per_yosys_lut4"], cal["ff_per_yosys_ff"]
    spec["features"] = [{"id": i, "label": l, "help": h} for i, l, h in FEATURES]
    spec["core"] = ["tmf_engine.v", "tmf_lane.v", "tmf_ram.v", "tmf_rq.v"]
    spec["rq_files"] = {"0": ["tmf_requant.v"], "1": ["tmf_requant_pipe.v"]}
    # Bases, deltas, pair corrections.
    spec["base"], spec["delta"], spec["pair"] = {}, {}, {"1": {}, "4": {}}
    for l in spec["lanes"]:
        for rq in (0, 1):
            spec["base"]["%d,%d" % (l, rq)] = met(by[("tmf_engine", l, rq, ())])
        b0 = spec["base"]["%d,0" % l]
        spec["delta"][str(l)] = {f: sub(met(by[("tmf_engine", l, 0, (f,))]), b0) for f in ids}
    for l in (1, 4):
        b0 = spec["base"]["%d,0" % l]
        for a, b in itertools.combinations(sorted(ids), 2):
            r = by.get(("tmf_engine", l, 0, tuple(sorted((a, b)))))
            if not r:
                continue
            corr = sub(sub(sub(met(r), b0), spec["delta"][str(l)][a]), spec["delta"][str(l)][b])
            # Keep only corrections that matter (more than 10 LUTs or flip-flops).
            if abs(corr["lut4"]) >= 10 or abs(corr["ff"]) >= 10:
                spec["pair"][str(l)][a + "+" + b] = corr
    # Buses.
    spec["buses"] = {}
    for k, (label, top, files, mod) in BUSES.items():
        spec["buses"][k] = {"label": label, "top": top, "files": files, "cost": met(by[(mod, 1, 0, ())])}
    # Sources.
    names = set(spec["core"]) | {n for v in spec["rq_files"].values() for n in v} | {n for b in spec["buses"].values() for n in b["files"] + [b["top"]]}
    spec["sources"] = {n: open(os.path.join(ROOT, "rtl", n)).read() for n in sorted(names)}
    # Validation against the random and all-on configurations (engine only: subtract the bridge).
    val, errs = [], []
    for r in rows:
        if r["kind"] not in ("random", "all"):
            continue
        p = r["params"]
        cfg = {"lanes": p.get("LANES", 1), "rq": p.get("RQ_MODE", 0), "features": [k for k in p if k.startswith("HAS_")], "bus": "uart"}
        e = configure.estimate(spec, cfg)
        pred = e["lut4"] - spec["buses"]["uart"]["cost"]["lut4"]
        err = (pred - r["lut4"]) / r["lut4"]
        errs.append(abs(err))
        val.append({"config": cfg, "measured": r["lut4"], "predicted": pred, "error_pct": round(100 * err, 1)})
    spec["validation"] = {"points": val, "max_abs_error_pct": round(100 * max(errs), 1), "mean_abs_error_pct": round(100 * sum(errs) / len(errs), 1)}
    os.makedirs(os.path.join(ROOT, "configurator"), exist_ok=True)
    with open(os.path.join(ROOT, "configurator", "spec.json"), "w") as f:
        json.dump(spec, f, indent=1, sort_keys=True)
    print("validation: %d points, mean |error| %.1f%%, max %.1f%%" % (len(val), spec["validation"]["mean_abs_error_pct"], spec["validation"]["max_abs_error_pct"]))
    for v in val:
        print("  %-60s measured %5d predicted %5d  %+.1f%%" % (json.dumps(v["config"], sort_keys=True)[:60], v["measured"], v["predicted"], v["error_pct"]))


if __name__ == "__main__":
    main()
