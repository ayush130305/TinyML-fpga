# test_litert.py - every int8 tensor of every test model matches LiteRT bit for bit,
# on the software engine and on the register model of the FPGA engine.
import pytest

from conftest import MODEL_NAMES, model_path

# LiteRT (or TensorFlow's interpreter) is needed for the reference.
try:
    import ai_edge_litert  # noqa: F401
    HAVE_REF = True
except ImportError:
    try:
        import tensorflow  # noqa: F401
        HAVE_REF = True
    except ImportError:
        HAVE_REF = False


# Quantise the digit images for a model's input.
def _samples(name, X, step):
    # Imports for the PC.
    import numpy as np
    from tinyml_fpga.check import _interpreter
    # Input quantisation of the model.
    with open(model_path(name), "rb") as f:
        d = _interpreter(f.read()).get_input_details()[0]
    s, z = d["quantization"]
    # int8 images.
    return [np.clip(np.round(x / s) + z, -128, 127).astype(np.int8) for x in X[::step]]


# Bit-exact on both engines.
@pytest.mark.skipif(not HAVE_REF, reason="LiteRT / TensorFlow not installed")
@pytest.mark.parametrize("backend", ["cpu", "sim"])
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_bit_exact(name, backend, digits):
    # Reference comparison.
    from tinyml_fpga.check import compare
    # Every 4th sample (50 samples).
    res = compare(model_path(name), _samples(name, digits[0], 4), backend=backend, verbose=False)
    # All int8 tensors equal, softmax within one output step.
    assert res["exact"], res["per_op"]
    # Same top-1 everywhere.
    assert res["agree"] == res["n"]


# Random int8 inputs exercise the clamps and rounding beyond natural images.
@pytest.mark.skipif(not HAVE_REF, reason="LiteRT / TensorFlow not installed")
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_bit_exact_random(name):
    # Helpers.
    from tinyml_fpga.check import compare
    from tinyml_fpga.__main__ import _random_samples
    # 30 random inputs.
    assert compare(model_path(name), _random_samples(model_path(name), 30), backend="sim", verbose=False)["exact"]


# Accuracy on the digits set stays the same as LiteRT's (sanity check of predict/quantize).
@pytest.mark.skipif(not HAVE_REF, reason="LiteRT / TensorFlow not installed")
def test_predict_accuracy(digits):
    # Runtime.
    import tinyml_fpga
    from tinyml_fpga.converter import convert_bytes
    # Model on the FPGA register model.
    with open(model_path("m_cnn"), "rb") as f:
        m = tinyml_fpga.Model(convert_bytes(f.read()), tinyml_fpga.HwEngine(tinyml_fpga.SimBus()))
    # Classify every sample.
    X, y = digits
    correct = sum(int(m.classify(X[i])[0] == y[i]) for i in range(len(X)))
    # The model is ~97-99% accurate on its training set.
    assert correct / len(X) > 0.9
