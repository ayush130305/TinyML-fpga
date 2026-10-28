# test_codebook.py - v0.7: weights stored as 4-bit indices into a per-layer 16-entry table, decompressed in the engine.
import random

import pytest

from conftest import model_path
from tinyml_fpga import Model, CpuEngine, HwEngine, SimBus
from tinyml_fpga import kernels as kn
from tinyml_fpga.converter import convert_bytes
from tinyml_fpga.transform import to_codebook, to_int4, compare


# int8, all-codebook and mixed (codebook + int4 + int8) versions of the test models.
@pytest.fixture(scope="module")
def tmfs():
    pytest.importorskip("tflite")
    out = {}
    for n in ("m_fc", "m_cnn", "m_avg"):
        with open(model_path(n), "rb") as f:
            t8 = convert_bytes(f.read())
        n_w = sum(1 for o in Model(t8).ops if o["op"] in ("conv", "dw", "fc"))
        mixed = to_codebook(t8, [0])[0]
        if n_w > 2:
            mixed = to_int4(mixed, [1])[0]
        out[n] = (t8, to_codebook(t8, "all")[0], mixed)
    return out


# Random int8 inputs.
def _inputs(m, n, seed=0):
    size = 1
    for d in m.input["shape"]:
        size *= d
    rng = random.Random(seed)
    return [bytearray(rng.randrange(256) for _ in range(size)) for _ in range(n)]


# Table lookup in the unpacker.
def test_unpack_with_codebook():
    # Two nibbles per byte, low first, values looked up.
    assert kn.unpack_w4(bytes([0x10, 0xF2]), 4, 1, [10 * i for i in range(16)]) == [0, 10, 20, 150]


# Hardware model and software engine agree on codebook (and mixed) models for several builds.
@pytest.mark.parametrize("lanes", [1, 2, 4])
@pytest.mark.parametrize("kw", [dict(), dict(queue=True, win=True, fuse=True)])
@pytest.mark.parametrize("which", [1, 2])
@pytest.mark.parametrize("name", ["m_fc", "m_cnn", "m_avg"])
def test_engine_matches_cpu(tmfs, name, which, lanes, kw):
    t = tmfs[name][which]
    ref = Model(t, CpuEngine())
    m = Model(t, HwEngine(SimBus(12, 10, 8, 10, lanes=lanes, rq_mode=1, w4=True, cb=True, **kw)))
    for x in _inputs(ref, 3):
        assert m.invoke(x) == ref.invoke(x)


# Small memories force streaming and chunking (the table is uploaded with every chunk).
@pytest.mark.parametrize("aw", [(8, 9, 6, 6), (7, 8, 5, 6)])
@pytest.mark.parametrize("name", ["m_cnn", "m_avg", "m_fc"])
def test_small_engines(tmfs, name, aw):
    t = tmfs[name][1]
    ref = Model(t, CpuEngine())
    m = Model(t, HwEngine(SimBus(*aw, lanes=2, queue=True, win=True, fuse=True, cb=True)))
    for x in _inputs(ref, 3, seed=2):
        assert m.invoke(x) == ref.invoke(x)


# Without the feature the layers run in software, still correct.
def test_engine_without_cb_falls_back(tmfs):
    t = tmfs["m_cnn"][1]
    ref = Model(t, CpuEngine())
    m = Model(t, HwEngine(SimBus(12, 10, 8, 10, w4=True)))
    x = _inputs(ref, 1)[0]
    assert m.invoke(x) == ref.invoke(x)
    assert all(t_[1] == "cpu" for t_ in m.timing if t_[0] in ("conv", "dw", "fc"))


# A codebook layer costs 17 more clocks than the same layer with a plain table-less int4 job.
def test_codebook_load_clocks(tmfs):
    from tinyml_fpga.engine import MODE_CB, MODE_W4
    from tinyml_fpga.kernels import pack_param
    cyc = {}
    for mode in (MODE_W4, MODE_CB):
        eng = HwEngine(SimBus(w_aw=8, x_aw=6, y_aw=4, p_aw=6, w4=True, cb=True))
        eng.upload_w(0, bytes(4), 2)
        eng.upload_p(0, pack_param(0, 1 << 30, 0) + bytes(16))
        eng.run(4, 1, 1, 4, 0, 0, 0, 0, -128, 127, mode, bytes(4))
        cyc[mode] = eng.cycles
    assert cyc[MODE_CB] - cyc[MODE_W4] == 17


# A table that does not fit parameter bank 0 is an error, not silent garbage.
def test_codebook_overflow_is_an_error():
    from tinyml_fpga.engine import MODE_CB
    eng = HwEngine(SimBus(w_aw=8, x_aw=6, y_aw=4, p_aw=5, cb=True))
    with pytest.raises(RuntimeError, match="engine error"):
        eng.run(2, 2, 1, 2, 0, 1, 0, 0, -128, 127, MODE_CB, bytes(2))


# The fitted table is closer to the weights than uniform 4-bit levels for these layers, at 16 bytes per layer.
def test_codebook_error_is_lower(tmfs):
    t8 = tmfs["m_cnn"][0]
    e4 = [r["rel_rms_error"] for r in to_int4(t8, "all")[1]]
    ec = [r["rel_rms_error"] for r in to_codebook(t8, "all")[1]]
    assert sum(ec) < sum(e4)


# Accuracy on real digit images.
def test_codebook_accuracy(tmfs, digits):
    t8, tc, _ = tmfs["m_cnn"]
    ti = Model(t8).input
    xs = [bytearray(v & 0xFF for v in kn.quantize(digits[0][i].reshape(-1).tolist(), ti["scale"], ti["zp"])) for i in range(60)]
    c = compare(t8, tc, xs)
    assert c["same_top1"] >= 0.9, c
