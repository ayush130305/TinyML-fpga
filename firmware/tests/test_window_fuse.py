# test_window_fuse.py - v0.5: the window generator (no host-side im2col) and layer fusion (one start per run of layers).
import random

import pytest

from conftest import model_path
from tinyml_fpga import Model, CpuEngine, HwEngine, SimBus
from tinyml_fpga.converter import convert_bytes
from tinyml_fpga.engine import window_block, window_vectors, MODE_WIN, MODE_OUTX, MODE_DW


# Converted test models.
@pytest.fixture(scope="module")
def tmfs():
    pytest.importorskip("tflite")
    out = {}
    for n in ("m_fc", "m_cnn", "m_avg"):
        with open(model_path(n), "rb") as f:
            out[n] = convert_bytes(f.read())
    return out


# Random int8 inputs.
def _inputs(m, n, seed=0):
    size = 1
    for d in m.input["shape"]:
        size *= d
    rng = random.Random(seed)
    return [bytearray(rng.randrange(256) for _ in range(size)) for _ in range(n)]


# Every build mix gives the same outputs as the software engine.
@pytest.mark.parametrize("lanes", [1, 2, 4])
@pytest.mark.parametrize("queue,win,fuse", [(False, True, False), (True, True, True), (True, False, True), (True, True, False)])
@pytest.mark.parametrize("name", ["m_fc", "m_cnn", "m_avg"])
def test_window_and_fusion_match_cpu(tmfs, name, lanes, queue, win, fuse):
    ref = Model(tmfs[name], CpuEngine())
    m = Model(tmfs[name], HwEngine(SimBus(12, 10, 8, 10, lanes=lanes, rq_mode=1, queue=queue, win=win, fuse=fuse)))
    for x in _inputs(ref, 3):
        assert m.invoke(x) == ref.invoke(x)


# Small memories: row splitting, channel chunks, chains cut short by X.
@pytest.mark.parametrize("aw", [(12, 9, 7, 10), (10, 8, 6, 8), (8, 7, 5, 6)])
@pytest.mark.parametrize("name", ["m_cnn", "m_avg", "m_fc"])
def test_window_small_memories(tmfs, name, aw):
    ref = Model(tmfs[name], CpuEngine())
    m = Model(tmfs[name], HwEngine(SimBus(*aw, lanes=2, queue=True, win=True, fuse=True)))
    for x in _inputs(ref, 3, seed=5):
        assert m.invoke(x) == ref.invoke(x)


# Window mode sends fewer bytes over the link than host-built patches; chains need fewer frames than single jobs.
def test_link_traffic_drops(tmfs):
    x = None
    stats = {}
    for tag, kw in (("plain", {}), ("win", dict(win=True)), ("fused", dict(win=True, fuse=True, queue=True))):
        sim = SimBus(12, 10, 8, 10, **kw)
        m = Model(tmfs["m_cnn"], HwEngine(sim))
        x = x or _inputs(m, 1)[0]
        m.invoke(x)
        sim.frames = sim.bytes = 0
        m.invoke(x)
        stats[tag] = (sim.frames, sim.bytes)
    assert stats["win"][1] < stats["plain"][1]
    assert stats["fused"][0] <= stats["win"][0]


# Chains are actually planned and used when the build allows them.
def test_chain_is_used(tmfs):
    m = Model(tmfs["m_cnn"], HwEngine(SimBus(12, 10, 8, 10, queue=True, win=True, fuse=True)))
    m.invoke(_inputs(m, 1)[0])
    assert m._chains
    assert any(t[1].endswith("-fused") for t in m.timing)


# keep=True (used by the LiteRT checker) never fuses, so every intermediate tensor exists.
def test_keep_disables_chains(tmfs):
    m = Model(tmfs["m_cnn"], HwEngine(SimBus(12, 10, 8, 10, queue=True, win=True, fuse=True)))
    m.invoke(_inputs(m, 1)[0], keep=True)
    assert all(not t[1].endswith("-fused") for t in m.timing)
    assert len(m.values) == len(m.ops) + 1


# Window descriptor limits: fields that cannot hold the geometry make the model fall back to host patches.
def test_window_block_limits():
    assert window_block(8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 8, 0, 0) is not None
    # Image too large for the 16-bit byte steps.
    assert window_block(300, 300, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 300, 0, 0) is None
    # Stride above 255.
    assert window_block(8, 8, 3, 3, 3, 1, 1, 256, 1, 1, 1, 1, 8, 0, 0) is None


# The window vector model agrees with the plain im2col kernels (padding, stride, dilation, depthwise multiplier).
def test_window_vectors_match_gather():
    from tinyml_fpga import kernels as kn
    rng = random.Random(3)
    for _ in range(60):
        h, w, c = rng.randrange(1, 7), rng.randrange(1, 7), rng.randrange(1, 4)
        kh, kw, dh, dwd = rng.randrange(1, 4), rng.randrange(1, 4), rng.randrange(1, 3), rng.randrange(1, 3)
        sh, sw, ph, pw = rng.randrange(1, 3), rng.randrange(1, 3), rng.randrange(0, 3), rng.randrange(0, 3)
        oh = (h + 2 * ph - ((kh - 1) * dh + 1)) // sh + 1
        ow = (w + 2 * pw - ((kw - 1) * dwd + 1)) // sw + 1
        if oh < 1 or ow < 1:
            continue
        dm = rng.randrange(1, 3)
        img = bytes(rng.randrange(256) for _ in range(h * w * c))
        pad = rng.randrange(256)
        for dw in (False, True):
            r0 = rng.randrange(oh)
            nrows = rng.randrange(1, oh - r0 + 1)
            xb = rng.randrange(0, 5)
            mem = bytes(xb) + img
            wv = window_block(h, w, c, kh, kw, dh, dwd, sh, sw, ph, pw, dm, ow, r0, xb)
            n = c * dm if dw else 3
            got = window_vectors(mem, xb, wv, kh * kw * (1 if dw else c), n, nrows * ow, dw, pad)
            want = bytearray()
            for i in range(nrows * ow):
                oy, ox = r0 + i // ow, i % ow
                if dw:
                    kn.gather_dw(img, h, w, c, dm, oy, ox, kh, kw, sh, sw, dh, dwd, ph, pw, pad, want)
                else:
                    kn.gather_conv(img, h, w, c, oy, ox, kh, kw, sh, sw, dh, dwd, ph, pw, pad, want)
            assert bytes(got) == bytes(want)
