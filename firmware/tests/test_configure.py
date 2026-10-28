# test_configure.py - the configurator: cost model sanity, package contents, the generated RTL elaborates for
# every bus / lane / requantiser combination, and the JavaScript twin gives identical output.
import itertools
import json
import os
import shutil
import subprocess
import zipfile

import pytest

from conftest import ROOT
from tinyml_fpga import configure

SPEC = configure.SPEC_PATH
pytestmark = pytest.mark.skipif(not os.path.exists(SPEC), reason="configurator/spec.json not built")


@pytest.fixture(scope="module")
def spec():
    return configure.load_spec()


# More lanes, more features and the pipelined requantiser never make the design smaller.
def test_cost_is_monotonic(spec):
    ids = [f["id"] for f in spec["features"]]
    prev = 0
    for l in spec["lanes"]:
        e = configure.estimate(spec, {"lanes": l})["le"]
        assert e > prev
        prev = e
    base = configure.estimate(spec, {"lanes": 1})["le"]
    assert configure.estimate(spec, {"lanes": 1, "rq": 1})["le"] > base
    for f in ids:
        assert configure.estimate(spec, {"lanes": 1, "features": [f]})["le"] >= base - 5


# The budget verdicts.
def test_verdicts(spec):
    e = configure.estimate(spec, {"lanes": 1})
    assert configure.verdict(e, 12000) == "fits"
    assert configure.verdict(e, e["le"] + 10) == "tight"
    assert configure.verdict(e, e["le"] - 10) == "over"


# Suggestions fit and are as fast as anything that fits.
def test_suggest(spec):
    for b in (4000, 8000):
        s = configure.suggest(spec, b)
        assert s is not None
        assert configure.verdict(configure.estimate(spec, s), b) == "fits"
    big = configure.suggest(spec, 12000)
    assert big["lanes"] == 8 and big["rq"] == 1
    # Required features are honoured.
    assert "HAS_WIN" in configure.suggest(spec, 8000, must=["HAS_WIN"])["features"]
    # Impossible budget.
    assert configure.suggest(spec, 500) is None


# Bad input is rejected.
def test_bad_config(spec):
    with pytest.raises(ValueError):
        configure.normalize(spec, {"lanes": 3})
    with pytest.raises(ValueError):
        configure.normalize(spec, {"features": ["HAS_NOPE"]})
    with pytest.raises(ValueError):
        configure.normalize(spec, {"bus": "usb"})
    with pytest.raises(ValueError):
        configure.normalize(spec, {"options": {"W_AW": 30}})


# Package: right files, parameters rewritten, zip deterministic.
def test_package(spec, tmp_path):
    cfg = {"lanes": 4, "rq": 1, "features": ["HAS_WIN", "HAS_FUSE", "HAS_QUEUE"], "bus": "spi", "budget": 20000}
    files = dict(configure.build_files(spec, cfg))
    assert "rtl/tinyml_spi_top.v" in files and "rtl/spi_bus.v" in files and "rtl/tmf_requant_pipe.v" in files
    assert "rtl/tmf_requant.v" not in files and "rtl/uart_bus.v" not in files
    top = files["rtl/tinyml_spi_top.v"]
    assert "parameter LANES  = 4" in top or "parameter LANES = 4" in top
    assert "HAS_WIN  = 1" in top or "HAS_WIN = 1" in top
    assert "HAS_CB   = 0" in top or "HAS_CB = 0" in top
    z1, z2 = tmp_path / "a.zip", tmp_path / "b.zip"
    configure.write_zip(spec, cfg, str(z1))
    configure.write_zip(spec, cfg, str(z2))
    assert z1.read_bytes() == z2.read_bytes()
    with zipfile.ZipFile(z1) as z:
        assert "README.md" in z.namelist() and "config.json" in z.namelist()


# Every bus x requantiser x lane count (with all features on) elaborates with the generated files only.
@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog missing")
@pytest.mark.parametrize("bus", ["uart", "spi", "i2c", "axil"])
@pytest.mark.parametrize("rq", [0, 1])
def test_generated_rtl_elaborates(spec, tmp_path, bus, rq):
    for lanes, feats in ((1, []), (8, [f["id"] for f in spec["features"]])):
        d = tmp_path / ("%s_%d_%d" % (bus, rq, lanes))
        cfg = {"lanes": lanes, "rq": rq, "features": feats, "bus": bus}
        for name, text in configure.build_files(spec, cfg):
            p = d / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        top = spec["buses"][bus]["top"][:-2]
        files = sorted(str(p) for p in (d / "rtl").glob("*.v"))
        r = subprocess.run(["iverilog", "-g2012", "-s", top, "-o", str(d / "x.vvp")] + files, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


# The JavaScript implementation gives the same estimates, verdicts, files and suggestions.
@pytest.mark.skipif(shutil.which("node") is None, reason="node missing")
def test_javascript_twin():
    r = subprocess.run(["python3", os.path.join(ROOT, "tools", "crosscheck.py"), "40"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
