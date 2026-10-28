# test_int4.py - v0.6: 4-bit weights (two per byte). The engine and the software engine agree bit for bit;
# the converter step keeps the outputs close to the int8 model.
import random

import pytest

from conftest import model_path
from tinyml_fpga import Model, CpuEngine, HwEngine, SimBus
from tinyml_fpga import kernels as kn
from tinyml_fpga.converter import convert_bytes
from tinyml_fpga.transform import to_int4, compare


# int8 and all-int4 versions of the test models.
@pytest.fixture(scope="module")
def tmfs():
    pytest.importorskip("tflite")
    out = {}
    for n in ("m_fc", "m_cnn", "m_avg"):
        with open(model_path(n), "rb") as f:
            t8 = convert_bytes(f.read())
        out[n] = (t8, to_int4(t8, "all")[0])
    return out


# Random int8 inputs.
def _inputs(m, n, seed=0):
    size = 1
    for d in m.input["shape"]:
        size *= d
    rng = random.Random(seed)
    return [bytearray(rng.randrange(256) for _ in range(size)) for _ in range(n)]


# Packing and unpacking agree, odd lengths included.
def test_unpack_roundtrip():
    rng = random.Random(1)
    for K in (1, 2, 3, 7, 8, 9):
        vals = [[rng.randrange(-8, 8) for _ in range(K)] for _ in range(3)]
        buf = bytearray()
        for row in vals:
            q = list(row) + ([0] if K & 1 else [])
            for k in range(0, len(q), 2):
                buf.append((q[k] & 15) | ((q[k + 1] & 15) << 4))
        assert kn.unpack_w4(buf, K, 3) == [v for r in vals for v in r]


# The engine (any build mix) gives exactly what the software engine gives for the same 4-bit model.
@pytest.mark.parametrize("lanes", [1, 2, 4])
@pytest.mark.parametrize("kw", [dict(), dict(queue=True, win=True, fuse=True)])
@pytest.mark.parametrize("name", ["m_fc", "m_cnn", "m_avg"])
def test_engine_matches_cpu(tmfs, name, lanes, kw):
    t8, t4 = tmfs[name]
    ref = Model(t4, CpuEngine())
    m = Model(t4, HwEngine(SimBus(12, 10, 8, 10, lanes=lanes, rq_mode=1, w4=True, **kw)))
    for x in _inputs(ref, 3):
        assert m.invoke(x) == ref.invoke(x)


# An engine without the feature runs 4-bit layers in software and still gives the same answers.
def test_engine_without_w4_falls_back(tmfs):
    t8, t4 = tmfs["m_cnn"]
    ref = Model(t4, CpuEngine())
    m = Model(t4, HwEngine(SimBus(12, 10, 8, 10)))
    x = _inputs(ref, 1)[0]
    assert m.invoke(x) == ref.invoke(x)
    assert all(t[1] == "cpu" for t in m.timing if t[0] in ("conv", "dw", "fc"))


# Half the weight RAM is enough: a model that streams its int8 weights keeps the int4 ones resident.
def test_weights_fit_in_half_the_ram(tmfs):
    t8, t4 = tmfs["m_cnn"]
    small = dict(w_aw=10, x_aw=10, y_aw=8, p_aw=10)
    assert not Model(t8, HwEngine(SimBus(**small))).resident
    assert Model(t4, HwEngine(SimBus(**small, w4=True))).resident


# Small memories: streaming, chunks, row splitting, with 4-bit weights.
@pytest.mark.parametrize("aw", [(8, 9, 6, 6), (7, 8, 5, 6)])
@pytest.mark.parametrize("name", ["m_cnn", "m_avg", "m_fc"])
def test_small_engines(tmfs, name, aw):
    t8, t4 = tmfs[name]
    ref = Model(t4, CpuEngine())
    m = Model(t4, HwEngine(SimBus(*aw, lanes=2, queue=True, win=True, fuse=True, w4=True)))
    for x in _inputs(ref, 3, seed=2):
        assert m.invoke(x) == ref.invoke(x)


# The conversion step keeps the model usable: top-1 mostly unchanged on real digit images.
def test_int4_accuracy(tmfs, digits):
    t8, t4 = tmfs["m_cnn"]
    ti = Model(t8).input
    xs = [bytearray(v & 0xFF for v in kn.quantize(digits[0][i].reshape(-1).tolist(), ti["scale"], ti["zp"])) for i in range(60)]
    c = compare(t8, t4, xs)
    assert c["same_top1"] >= 0.9, c
    assert c["mean_abs"] < 0.05, c


# "auto" leaves the first and last weight layers at 8 bits.
def test_auto_selection(tmfs):
    t8, _ = tmfs["m_cnn"]
    t4, rep = to_int4(t8, "auto")
    m = Model(t4)
    bits = [op.get("wbits", 8) for op in m.ops if op["op"] in ("conv", "dw", "fc")]
    assert bits[0] == 8 and bits[-1] == 8 and 4 in bits
