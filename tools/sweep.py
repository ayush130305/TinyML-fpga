# sweep.py - the Yosys measurements behind the configurator's cost table.
# Builds tmf_engine in a structured set of configurations (every LANES / RQ_MODE base, every feature on its own at every
# LANES, all feature pairs at one lane, and random configurations kept apart for validating the fitted model) plus the four
# bus bridges, and writes everything to tools/sweep.json. Takes about half an hour; tools/fit_cost.py turns the result
# into tools/cost_model.json.
#   python3 tools/sweep.py [workers]

import itertools
import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure

# Optional engine features (parameter names).
FEATURES = ["HAS_QUEUE", "HAS_PERF", "HAS_WIN", "HAS_FUSE", "HAS_W4", "HAS_CB"]
# Output file.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep.json")


# One measurement job: (kind, top, params).
def _job(j):
    kind, top, params = j
    return {"kind": kind, "top": top, "params": params, **measure.synth(top, params)}


# All jobs, ordered so the slow ones start first.
def jobs():
    out = []
    # Bases.
    for l in (1, 2, 4, 8):
        for rq in (0, 1):
            out.append(("base", "tmf_engine", {"LANES": l, "RQ_MODE": rq}))
    # Each feature alone, at every lane count.
    for l in (1, 2, 4, 8):
        for f in FEATURES:
            out.append(("single", "tmf_engine", {"LANES": l, "RQ_MODE": 0, f: 1}))
    # Feature pairs at one lane and at four lanes.
    for l in (1, 4):
        for a, b in itertools.combinations(FEATURES, 2):
            out.append(("pair", "tmf_engine", {"LANES": l, "RQ_MODE": 0, a: 1, b: 1}))
    # Random configurations for validation.
    rng = random.Random(7)
    for _ in range(16):
        p = {"LANES": rng.choice((1, 2, 4)), "RQ_MODE": rng.randint(0, 1)}
        for f in FEATURES:
            if rng.random() < 0.5:
                p[f] = 1
        out.append(("random", "tmf_engine", p))
    # Everything on.
    for l in (1, 4):
        out.append(("all", "tmf_engine", {"LANES": l, "RQ_MODE": 1, **{f: 1 for f in FEATURES}}))
    # Bridges (UART with its default baud divider and timeout).
    out.append(("bridge", "uart_bus", {}))
    out.append(("bridge", "spi_bus", {}))
    out.append(("bridge", "i2c_bus", {}))
    out.append(("bridge", "axil_bus", {}))
    # Slowest first.
    out.sort(key=lambda j: -j[2].get("LANES", 1))
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    js = jobs()
    res = []
    with Pool(n) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, js)):
            res.append(r)
            print("%d/%d %s %s lut4=%d ff=%d dsp=%d" % (i + 1, len(js), r["top"], r["params"], r["lut4"], r["ff"], r["dsp"]), flush=True)
            # Save as we go so a stop loses nothing.
            with open(OUT, "w") as f:
                json.dump(res, f, indent=1)
