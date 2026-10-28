# crosscheck.py - run the Python and the JavaScript configurator on the same random configurations and compare the
# estimates, verdicts, warnings, generated files (by SHA-256) and the suggested configurations.
#   python3 tools/crosscheck.py [n]
import hashlib
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "firmware"))
from tinyml_fpga import configure


# Random but valid configurations (including odd RAM sizes and bus options).
def random_configs(spec, n, seed=3):
    rng = random.Random(seed)
    ids = [f["id"] for f in spec["features"]]
    out = []
    for _ in range(n):
        out.append({
            "lanes": rng.choice(spec["lanes"]), "rq": rng.randint(0, 1),
            "features": [i for i in ids if rng.random() < 0.5], "bus": rng.choice(list(spec["buses"])),
            "dsp": rng.random() < 0.7, "budget": rng.choice(spec["budgets"] + [2500, 12345]),
            "options": {"W_AW": rng.randint(8, 15), "X_AW": rng.randint(6, 12), "Y_AW": rng.randint(4, 12),
                        "P_AW": rng.randint(6, 12), "D_AW": rng.randint(7, 12), "clk_hz": rng.choice((12000000, 50000000, 100000000)),
                        "baud": rng.choice((115200, 921600, 3000000)), "i2c_addr": rng.randint(8, 119)}})
    return out


# Compare; returns the number of differences.
def run(n=60):
    spec = configure.load_spec()
    cfgs = random_configs(spec, n)
    suggests = [{"budget": b, "bus": bus, "dsp": d, "must": m} for b in (4000, 8000) for bus in ("uart", "spi") for d in (True, False) for m in ([], ["HAS_WIN"])]
    p = subprocess.run(["node", os.path.join(HERE, "crosscheck.js")], input=json.dumps({"spec": spec, "configs": cfgs, "suggests": suggests}),
                       capture_output=True, text=True, check=True)
    js = json.loads(p.stdout)
    bad = 0
    for cfg, j in zip(cfgs, js["configs"]):
        c = configure.normalize(spec, cfg)
        e = configure.estimate(spec, c)
        files = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in configure.build_files(spec, c)}
        mine = {"est": e, "verdict": configure.verdict(e, c["budget"]), "warnings": configure.warnings(spec, c), "files": files}
        if json.loads(json.dumps(mine)) != j:
            bad += 1
            print("DIFF", json.dumps(cfg)[:200])
            for k in mine:
                if json.loads(json.dumps(mine[k])) != j[k]:
                    print("  field", k)
    for s, j in zip(suggests, js["suggests"]):
        py = configure.suggest(spec, s["budget"], s["bus"], s["dsp"], s["must"])
        if json.loads(json.dumps(py)) != j:
            bad += 1
            print("SUGGEST DIFF", s, py, j)
    print("crosscheck: %d configurations + %d suggestions, %d differences" % (len(cfgs), len(suggests), bad))
    return bad


if __name__ == "__main__":
    sys.exit(1 if run(int(sys.argv[1]) if len(sys.argv) > 1 else 60) else 0)
