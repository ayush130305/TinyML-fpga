# bench_cycles.py - engine clocks per inference for a model on different engine builds (register model).
# usage: python3 tools/bench_cycles.py [model.tflite ...]
# The numbers are the cycle counter of the RTL (SimBus reproduces it exactly, tb_engine checks that), so they
# are what the FPGA reports. Link and host time come on top and depend on your bus.

import os
import random
import sys

# Make the package importable from the repo.
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "firmware"))
from tinyml_fpga import Model, HwEngine, SimBus
from tinyml_fpga.converter import convert_bytes


# Wrap an engine so that the clocks of every job are summed.
class Counting(HwEngine):
    # Sum of job clocks since the last reset.
    total = 0

    # Run and accumulate.
    def run(self, *a):
        y = HwEngine.run(self, *a)
        self.total += self.cycles
        return y


# Total engine clocks of one inference of a converted model on a build.
def clocks(tmf, **build):
    # Big enough memories so the weights stay resident.
    eng = Counting(SimBus(w_aw=13, x_aw=11, y_aw=10, p_aw=11, **build))
    m = Model(tmf, eng)
    # Random but fixed input.
    n = 1
    for d in m.input["shape"]:
        n *= d
    rng = random.Random(1)
    eng.total = 0
    m.invoke(bytearray(rng.randrange(256) for _ in range(n)))
    return eng.total


# Table for the given models.
def main(paths):
    # Builds: (label, kwargs).
    builds = [("1 lane, serial rq", dict(lanes=1, rq_mode=0)), ("1 lane, pipelined rq", dict(lanes=1, rq_mode=1)),
              ("2 lanes, pipelined", dict(lanes=2, rq_mode=1)), ("4 lanes, pipelined", dict(lanes=4, rq_mode=1)),
              ("8 lanes, pipelined", dict(lanes=8, rq_mode=1))]
    print("%-24s" % "engine clocks / inference" + "".join("%10s" % os.path.basename(p)[:-7] for p in paths))
    tmfs = [convert_bytes(open(p, "rb").read()) for p in paths]
    for label, kw in builds:
        print("%-24s" % label + "".join("%10d" % clocks(t, **kw) for t in tmfs))


if __name__ == "__main__":
    main(sys.argv[1:] or [os.path.join(ROOT, "firmware", "tests", "models", n + ".tflite") for n in ("m_fc", "m_cnn", "m_avg")])
