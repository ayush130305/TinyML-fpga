# gen_requant_vectors.py - random and corner cases for tb_requant.v, expected values from kernels.requant
# (the same reference the runtime is verified against LiteRT with).
# Output requant_vectors.hex: word 0 = case count, then 4 words per case: acc, mult, {shift,zp,lo,hi}, expected y.
# usage: python3 gen_requant_vectors.py [seed] [cases]

import os
import random
import sys

# Make the package importable from the repo.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "firmware"))
from tinyml_fpga.kernels import requant

# Seed and case count.
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
rng = random.Random(SEED)
# Cases: (acc, mult, shift, zp, lo, hi).
cases = []
# Corner values first.
accs = [0, 1, -1, 2 ** 31 - 1, -2 ** 31, 12345, -12345, 1 << 20, -(1 << 20)]
mults = [0, 1, (1 << 30), (1 << 31) - 1, (1 << 30) + 1]
for a in accs:
    for m in mults:
        for sh in (-31, -15, -1, 0, 1, 30):
            cases.append((a, m, sh, 0, -128, 127))
# Random cases: mostly realistic, some wild.
while len(cases) < N:
    wild = rng.random() < 0.3
    a = rng.randrange(-2 ** 31, 2 ** 31) if wild else rng.randrange(-200000, 200000)
    m = 0 if rng.random() < 0.03 else rng.randrange(1 << 30, 1 << 31)
    sh = rng.randrange(-31, 31) if wild else rng.randrange(-14, 1)
    zp = rng.randrange(-128, 128)
    lo = rng.randrange(-128, 128)
    hi = rng.randrange(lo, 128)
    cases.append((a, m, sh, zp, lo, hi))
# Words.
words = [len(cases)]
for a, m, sh, zp, lo, hi in cases:
    y = requant(a, m, sh, zp, lo, hi)
    words += [a & 0xFFFFFFFF, m & 0xFFFFFFFF, ((sh & 0xFF) << 24) | ((zp & 0xFF) << 16) | ((lo & 0xFF) << 8) | (hi & 0xFF), y & 0xFF]
# Write next to this script.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "requant_vectors.hex"), "w") as f:
    f.write("\n".join("%08X" % w for w in words) + "\n")
print("wrote %d cases (seed %d)" % (len(cases), SEED))
