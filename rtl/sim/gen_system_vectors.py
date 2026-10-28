# gen_system_vectors.py - record every bus frame of a real model inference (runtime -> HwEngine -> SimBus)
# so tb_uart_top.v can replay them through the UART bridge RTL and check every byte read back.
#
# Output: system_vectors.hex, one 32-bit word per line:
#   10_AAAA_LL  write frame of LL+1 bytes at AAAA, followed by LL+1 words with the data bytes
#   11_AAAA_LL  read frame of LL+1 bytes at AAAA, followed by LL+1 words with the expected bytes
#   12_000000   poll STATUS (0x120) until done, expect no error
#   FF_000000   end
# usage: python3 gen_system_vectors.py [model.tflite] [inferences] [lanes] [rq_mode] [queue] [perf] [win] [fuse]

import os
import sys

# Repo root on the import path.
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "firmware")
sys.path.insert(0, ROOT)
import numpy as np
from tinyml_fpga.converter import convert_bytes
from tinyml_fpga.engine import HwEngine, SimBus, R_STATUS
from tinyml_fpga.model import Model
from tinyml_fpga import kernels as kn

# Engine RAM sizes of the testbench top (the tmf_engine.v defaults) and build options.
AW = dict(w_aw=12, x_aw=10, y_aw=8, p_aw=10, lanes=int(sys.argv[3]) if len(sys.argv) > 3 else 1,
          rq_mode=int(sys.argv[4]) if len(sys.argv) > 4 else 0, queue=bool(int(sys.argv[5])) if len(sys.argv) > 5 else False,
          perf=bool(int(sys.argv[6])) if len(sys.argv) > 6 else False,
          win=bool(int(sys.argv[7])) if len(sys.argv) > 7 else False,
          fuse=bool(int(sys.argv[8])) if len(sys.argv) > 8 else False)
# Model and inference count.
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tests", "models", "m_cnn.tflite")
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 2


# Bus that forwards to SimBus and logs frames the way a 128-byte-frame bus (e.g. UartBus) sends them.
class RecordingBus:
    # Wrap a SimBus.
    def __init__(self, sim):
        # Golden model.
        self.sim = sim
        # Recorded words.
        self.words = []

    # Write frames of at most 128 bytes.
    def write_block(self, a, buf):
        # Frame by frame.
        for i in range(0, len(buf), 128):
            # Frame data.
            part = bytes(buf[i:i + 128])
            # Header and data words.
            self.words.append((0x10 << 24) | ((a + i) << 8) | (len(part) - 1))
            self.words.extend(part)
            # Forward to the model.
            self.sim.write_block(a + i, part)

    # Read frames; a STATUS read becomes a poll (the RTL takes real time to finish).
    def read_block(self, a, n):
        # Expected data from the model.
        data = self.sim.read_block(a, n)
        # Status poll.
        if a == R_STATUS:
            # Poll word.
            self.words.append(0x12 << 24)
            # Remaining bytes (the cycle counter) as a normal read.
            a, rest = a + 1, data[1:]
        else:
            # Whole read.
            rest = data
        # Frame by frame.
        for i in range(0, len(rest), 128):
            # Expected bytes.
            part = rest[i:i + 128]
            # Header and expected words.
            self.words.append((0x11 << 24) | ((a + i) << 8) | (len(part) - 1))
            self.words.extend(part)
        # Data back to the driver.
        return data


# Recorder over the golden model.
rec = RecordingBus(SimBus(**AW))
# Runtime on the "hardware" engine.
with open(MODEL, "rb") as f:
    model = Model(convert_bytes(f.read()), HwEngine(rec))
# Sample inputs.
d = np.load(os.path.join(ROOT, "tests", "models", "digits_samples.npz"))
ti = model.input
# A few inferences.
for r in range(RUNS):
    # Quantised sample.
    xq = kn.quantize(d["X"][r].reshape(-1).tolist(), ti["scale"], ti["zp"])
    # Run (weights stay resident after the first upload).
    out = model.invoke(bytearray(v & 0xFF for v in xq))
    # Report the prediction for the log.
    print("inference %d: label %d -> predicted %d" % (r, int(d["y"][r]), max(range(len(out)), key=lambda i: out[i])))
# End marker.
rec.words.append(0xFF << 24)
# Write the stream.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_vectors.hex"), "w") as f:
    f.write("\n".join("%08X" % w for w in rec.words) + "\n")
# Summary.
frames = sum(1 for w in rec.words if (w >> 24) in (0x10, 0x11, 0x12))
print("wrote %d words, %d frames, %d link bytes" % (len(rec.words), frames, rec.sim.bytes))
