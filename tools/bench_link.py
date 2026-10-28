# bench_link.py - link traffic per inference (bus frames and bytes) for a model on different engine builds.
# usage: python3 tools/bench_link.py [model.tflite ...]
# A frame is one bus transaction (at most 128 bytes); over a UART each costs a round trip. Weights are resident,
# so only the per-inference traffic is counted (second inference).

import os
import random
import sys

# Make the package importable from the repo.
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "firmware"))
from tinyml_fpga import Model, HwEngine, SimBus
from tinyml_fpga.converter import convert_bytes


# (frames, bytes) of one inference.
def traffic(tmf, **build):
    # Big enough memories so the weights stay resident.
    sim = SimBus(w_aw=13, x_aw=11, y_aw=10, p_aw=11, lanes=1, rq_mode=1, **build)
    m = Model(tmf, HwEngine(sim))
    n = 1
    for d in m.input["shape"]:
        n *= d
    x = bytearray(random.Random(1).randrange(256) for _ in range(n))
    # First inference uploads the weights; measure the second.
    m.invoke(x)
    sim.frames = sim.bytes = 0
    m.invoke(x)
    return sim.frames, sim.bytes


# Table for the given models.
def main(paths):
    builds = [("plain (host im2col)", {}), ("+ HAS_WIN", dict(win=True)),
              ("+ HAS_WIN, HAS_QUEUE, HAS_FUSE", dict(win=True, queue=True, fuse=True))]
    tmfs = [convert_bytes(open(p, "rb").read()) for p in paths]
    print("%-34s" % "frames / bytes per inference" + "".join("%16s" % os.path.basename(p)[:-7] for p in paths))
    for label, kw in builds:
        print("%-34s" % label + "".join("%16s" % ("%d / %d" % traffic(t, **kw)) for t in tmfs))


if __name__ == "__main__":
    main(sys.argv[1:] or [os.path.join(ROOT, "firmware", "tests", "models", n + ".tflite") for n in ("m_fc", "m_cnn", "m_avg")])
