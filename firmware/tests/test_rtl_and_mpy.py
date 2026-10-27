# test_rtl_and_mpy.py - run the Verilog testbenches (Icarus) and the MicroPython runtime check,
# when those tools are installed.
import os
import shutil
import subprocess
import sys

import pytest

from conftest import FIRMWARE, ROOT, model_path

# Simulation directory.
SIM = os.path.join(ROOT, "rtl", "sim")


# Run a command in rtl/sim/ and return its output.
def _run(cmd, timeout=1800):
    # Capture everything.
    r = subprocess.run(cmd, cwd=SIM, capture_output=True, text=True, timeout=timeout)
    # Combined output.
    return r.stdout + r.stderr


# Engine unit testbench against the Python register model.
@pytest.mark.skipif(not shutil.which("iverilog"), reason="Icarus Verilog not installed")
def test_tb_engine():
    # Fresh random vectors.
    _run([sys.executable, "gen_engine_vectors.py", "7", "60"])
    # Compile and run.
    out = _run(["bash", "run_sim.sh", "engine"])
    # Verdict line.
    assert "PASS tb_engine" in out, out[-2000:]


# Whole-model inference through the UART bridge into the engine.
@pytest.mark.skipif(not shutil.which("iverilog"), reason="Icarus Verilog not installed")
@pytest.mark.parametrize("name", ["m_cnn", "m_avg"])
def test_tb_system(name):
    # Pytest needs the tflite schema to record the inference.
    pytest.importorskip("tflite")
    # Record the inference.
    _run([sys.executable, "gen_system_vectors.py", model_path(name), "2"])
    # Replay through the UART top.
    out = _run(["bash", "run_sim.sh", "uart"])
    # Verdict.
    assert "PASS tb_uart_top" in out, out[-2000:]


# The runtime under the MicroPython unix port (the interpreter microcontrollers run).
@pytest.mark.skipif(not shutil.which("micropython"), reason="MicroPython unix port not installed")
def test_micropython(tmp_path, digits):
    # PC-side helpers.
    pytest.importorskip("tflite")
    import json
    import struct
    import tinyml_fpga
    from tinyml_fpga import kernels as kn
    from tinyml_fpga.converter import convert_bytes
    # Model without the float softmax so outputs compare as bytes.
    with open(model_path("m_cnn"), "rb") as f:
        raw = convert_bytes(f.read())
    hl = struct.unpack_from("<I", raw, 4)[0]
    h = json.loads(raw[8:8 + hl])
    h["ops"] = [o for o in h["ops"] if o["op"] != "softmax"]
    h["output"] = h["ops"][-1]["out"]
    hj = json.dumps(h, separators=(",", ":")).encode()
    tmf = b"TMF1" + struct.pack("<I", len(hj)) + hj + raw[8 + hl:]
    (tmp_path / "m.tmf").write_bytes(tmf)
    # Inputs and expected outputs from CPython.
    m = tinyml_fpga.Model(tmf)
    ti = m.input
    X, E = b"", b""
    for i in range(6):
        xq = bytearray(v & 0xFF for v in kn.quantize(digits[0][i].reshape(-1).tolist(), ti["scale"], ti["zp"]))
        X += xq
        E += bytes(m.invoke(xq))
    (tmp_path / "in.bin").write_bytes(X)
    (tmp_path / "exp.bin").write_bytes(E)
    # Run under MicroPython.
    r = subprocess.run(["micropython", "-X", "heapsize=4M", os.path.join(FIRMWARE, "tests", "mpy", "run_mpy.py"), FIRMWARE,
                        str(tmp_path / "m.tmf"), str(tmp_path / "in.bin"), str(tmp_path / "exp.bin"), "6", "64", "10"],
                       capture_output=True, text=True, timeout=600)
    # Verdict.
    assert "MPY PASS" in r.stdout, r.stdout + r.stderr
