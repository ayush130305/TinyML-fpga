# run_pc.py - PC: run digits.tmf in software, on the FPGA register model, or on a real FPGA over UART.
#   python run_pc.py                 software engine
#   python run_pc.py sim             register model of the FPGA engine (default RAM sizes)
#   python run_pc.py uart:COM5       any FPGA board running rtl/tinyml_uart_top.v at 921600 baud
import os
import sys
import time

# Repo import without installing.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "firmware"))
import tinyml_fpga
from samples import SAMPLES

# Backend from the command line.
backend = sys.argv[1] if len(sys.argv) > 1 else "cpu"
# Baud rate of tinyml_uart_top's default build.
kw = {"baud": 921600} if backend.startswith("uart:") else {}
# Load.
m = tinyml_fpga.load(os.path.join(HERE, "digits.tmf"), backend=backend, **kw)
m.summary()
# Classify the samples.
ok = 0
for label, pixels in SAMPLES:
    # Timed run.
    t0 = time.perf_counter()
    idx, p = m.classify(pixels)
    dt = (time.perf_counter() - t0) * 1e3
    # Report.
    print("label %d -> %d  (p=%.2f, %.1f ms)" % (label, idx, p, dt))
    ok += idx == label
print("correct: %d / %d" % (ok, len(SAMPLES)))
