# test_kernels.py - the int8 arithmetic itself.
import math
import random

from tinyml_fpga import kernels as kn
from tinyml_fpga.converter import quantize_multiplier, act_range


# Single-rounding requant equals round-half-up of the exact real product.
def test_mbqm_matches_exact_rounding():
    # Reproducible.
    rng = random.Random(0)
    # Many random cases.
    for _ in range(20000):
        # Operands in the TFLite ranges.
        acc = rng.randrange(-2 ** 31, 2 ** 31)
        mult = rng.randrange(1 << 30, 1 << 31)
        shift = rng.randrange(-31, 31)
        # Exact value acc*mult*2^(shift-31), rounded half up, with exact integers.
        num, den = acc * mult * (2 ** shift if shift > 0 else 1), 2 ** 31 * (2 ** -shift if shift < 0 else 1)
        exact = (2 * num + den) // (2 * den)
        # Same result.
        assert kn.mbqm(acc, mult, shift) == exact


# The multiplier/shift pair reproduces the real scale.
def test_quantize_multiplier_roundtrip():
    # A range of real scales.
    for real in (1e-9, 3.3e-5, 0.0123, 0.5, 0.75, 0.999999, 1.0, 1.7, 123.4):
        # Fixed-point form.
        m, s = quantize_multiplier(real)
        # Q31 mantissa in [0.5, 1).
        assert (1 << 30) <= m < (1 << 31)
        # Reconstruction is within the Q31 resolution.
        assert math.isclose(m * 2.0 ** (s - 31), real, rel_tol=2 ** -30)
    # Zero stays zero.
    assert quantize_multiplier(0.0) == (0, 0)


# Fused activation ranges.
def test_act_range():
    # NONE.
    assert act_range(0, 0.1, 3) == (-128, 127)
    # RELU starts at the zero point.
    assert act_range(1, 0.1, -20) == (-20, 127)
    # RELU6 stops at zp + 6/scale.
    assert act_range(3, 0.05, -128) == (-128, -8)


# Requant clamps to the activation range.
def test_requant_clamp():
    # Large positive clamps high.
    assert kn.requant(10 ** 6, 1 << 30, 0, 0, -10, 10) == 10
    # Large negative clamps low.
    assert kn.requant(-10 ** 6, 1 << 30, 0, 0, -10, 10) == -10
    # In range: 100 * 0.5 + 3.
    assert kn.requant(100, 1 << 30, 0, 3, -128, 127) == 53


# Rounding helper is half away from zero, like std::round.
def test_round_away():
    # Ties.
    assert [kn.round_away(v) for v in (0.5, 1.5, -0.5, -1.5, 2.4, -2.6)] == [1, 2, -1, -2, 2, -3]


# Average pooling rounds like the TFLite reference.
def test_avgpool_rounding():
    # One 2x2 window with sum 5 -> 5/4 = 1.25 -> 1; sum -6 -> -1.5 -> -2.
    y, _, _ = kn.pool([1, 1, 1, 2], 2, 2, 1, 2, 2, 2, 2, False, False, -128, 127)
    assert y == [1]
    y, _, _ = kn.pool([-1, -1, -2, -2], 2, 2, 1, 2, 2, 2, 2, False, False, -128, 127)
    assert y == [-2]


# Parameter records round-trip.
def test_param_pack():
    # Pack and unpack.
    rec = kn.pack_param(-123456, 1518500250, -7)
    assert len(rec) == kn.PARAM_BYTES
    assert kn.unpack_param(rec, 0) == (-123456, 1518500250, -7)
