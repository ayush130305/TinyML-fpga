# conftest.py - shared fixtures: repo root on sys.path, test models and sample data.
import os
import sys

import pytest

# The firmware folder (holds the tinyml_fpga package and these tests).
FIRMWARE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Repo root.
ROOT = os.path.abspath(os.path.join(FIRMWARE, ".."))
# Make 'import tinyml_fpga' importable without installing.
sys.path.insert(0, FIRMWARE)
# Test models directory.
MODELS = os.path.join(FIRMWARE, "tests", "models")
# The three int8 test models (all supported op types between them).
MODEL_NAMES = ("m_fc", "m_cnn", "m_avg")


# Path of a test model.
def model_path(name):
    # .tflite next to the samples.
    return os.path.join(MODELS, name + ".tflite")


# Digit samples (float images and labels).
@pytest.fixture(scope="session")
def digits():
    # numpy is a test dependency on the PC.
    np = pytest.importorskip("numpy")
    # Load once.
    d = np.load(os.path.join(MODELS, "digits_samples.npz"))
    # Images and labels.
    return d["X"], d["y"]
