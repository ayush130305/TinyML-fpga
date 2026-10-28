# test_engine_paths.py - the driver paths that small FPGAs take: weights streamed per inference,
# channel chunks, software fallback for layers that can never fit, error reporting.
import random

import pytest

from conftest import model_path
from tinyml_fpga import Model, CpuEngine, HwEngine, SimBus
from tinyml_fpga.converter import convert_bytes
from tinyml_fpga.engine import config_frame, R_CFG, X_ADDR, MODE_RAW


# Converted test models.
@pytest.fixture(scope="module")
def tmfs():
    # Needs the tflite schema package.
    pytest.importorskip("tflite")
    # name -> .tmf bytes.
    out = {}
    for n in ("m_fc", "m_cnn", "m_avg"):
        with open(model_path(n), "rb") as f:
            out[n] = convert_bytes(f.read())
    return out


# Random int8 inputs for a model.
def _inputs(m, n, seed=0):
    # Input length.
    size = 1
    for d in m.input["shape"]:
        size *= d
    # Reproducible.
    rng = random.Random(seed)
    # n buffers.
    return [bytearray(rng.randrange(256) for _ in range(size)) for _ in range(n)]


# Small engines give the same outputs as the software engine.
@pytest.mark.parametrize("aw", [(12, 10, 8, 10), (8, 9, 6, 6), (6, 6, 4, 5), (10, 4, 8, 10)])
@pytest.mark.parametrize("lanes", [1, 2, 4])
@pytest.mark.parametrize("name", ["m_fc", "m_cnn", "m_avg"])
def test_small_engines(tmfs, name, aw, lanes):
    # Banks need at least one parameter record each.
    if (1 << aw[3]) // 16 < lanes:
        pytest.skip("P RAM too small for this many lanes")
    # Reference.
    ref = Model(tmfs[name], CpuEngine())
    # Engine with tiny memories (streaming, chunking and fallback kick in).
    m = Model(tmfs[name], HwEngine(SimBus(*aw, lanes=lanes)))
    # Same results on random inputs.
    for x in _inputs(ref, 4):
        assert m.invoke(x) == ref.invoke(x)


# The tiny configurations really exercise the non-resident and fallback paths.
def test_paths_are_exercised(tmfs):
    # 256-byte weight RAM: m_fc cannot keep its weights resident.
    m = Model(tmfs["m_fc"], HwEngine(SimBus(8, 9, 6, 6)))
    assert not m.resident
    # 16-byte X RAM: the 64-long FC row cannot fit, so that layer falls back to software.
    m = Model(tmfs["m_fc"], HwEngine(SimBus(10, 4, 8, 10)))
    m.invoke(_inputs(m, 1)[0])
    assert ("fc", "cpu") in [(t[0], t[1]) for t in m.timing]


# A bad job raises a clear error (memory overrun detected by the engine).
def test_engine_error():
    # Engine model.
    eng = HwEngine(SimBus(8, 6, 4, 6))
    # X overrun: 100 vectors of 4 bytes in a 64-byte X.
    with pytest.raises(RuntimeError, match="engine error"):
        eng.run(4, 1, 100, 4, 0, 0, 0, 0, -128, 127, 0, b"")


# Raw int32 mode returns accumulators.
def test_raw_mode():
    # Engine model.
    eng = HwEngine(SimBus())
    # Weights 1,2 and bias 5 for one channel.
    from tinyml_fpga.kernels import pack_param
    eng.upload_w(0, bytes([1, 2]))
    eng.upload_p(0, pack_param(5, 1 << 30, 0))
    # 5 + (3+1)*1 + (4+1)*2 = 19 with in_off = 1.
    assert eng.run(2, 1, 1, 2, 0, 0, 1, 0, -128, 127, MODE_RAW, bytes([3, 4])) == [19]


# A missing bitstream is reported instead of producing garbage.
def test_missing_engine():
    # Bus that reads zeros everywhere.
    class Dead:
        def read_block(self, a, n):
            return bytearray(n)
        def write_block(self, a, b):
            pass
    # Clear message.
    with pytest.raises(RuntimeError, match="not found"):
        HwEngine(Dead())
