# gen_engine_vectors.py - random and corner-case jobs for tb_engine.v, with expected results from
# tinyml_fpga.engine.SimBus (the register model the runtime is verified against LiteRT with).
#
# Output: engine_vectors.hex, one 32-bit word per line:
#   01_AAAA_DD  write byte DD to address AAAA
#   02_000000   start the job (CTRL = 1) and wait for STATUS.done
#   03_AAAA_DD  read address AAAA, expect DD
#   04_0000_DD  expect STATUS == DD
#   05_000000   expect CYCLES == the next word
#   FF_000000   end of test
# usage: python3 gen_engine_vectors.py [seed] [jobs]

import os
import random
import struct
import sys

# Make the package importable from the repo.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "firmware"))
from tinyml_fpga.engine import SimBus, config_frame, X_ADDR, Y_ADDR, P_ADDR, W_ADDR, R_CFG, R_STATUS, MODE_DW, MODE_RAW
from tinyml_fpga.kernels import pack_param

# RAM sizes of the testbench build (must match tb_engine.v parameters).
W_AW, X_AW, Y_AW, P_AW = 10, 9, 8, 8
# Seed and job count.
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
# Random generator.
rng = random.Random(SEED)
# Output words.
words = []
# Golden model.
sim = SimBus(W_AW, X_AW, Y_AW, P_AW)


# Write a block to the golden model and emit the matching write words.
def wblock(addr, data):
    # Each byte.
    for i, b in enumerate(data):
        # Command word.
        words.append((0x01 << 24) | ((addr + i) << 8) | b)
    # Same writes into the model (without the start byte semantics: config frames never include CTRL here).
    sim.write_block(addr, data)


# Emit one job and its expected results.
def job(K, N, nvec, xs, wb, pb, ioff, zp, amin, amax, mode, check_cycles=True):
    # Configuration without the trailing start byte.
    cfg = config_frame(K, N, nvec, xs, wb, pb, ioff, zp, amin, amax, mode)[:0x12]
    # Config writes.
    wblock(R_CFG, cfg)
    # Start in the model.
    sim.write(0x11F, 1)
    # Start in the testbench.
    words.append(0x02 << 24)
    # Expected status.
    words.append((0x04 << 24) | sim.status)
    # Results only for successful jobs.
    if sim.status == 2:
        # Output bytes.
        nb = nvec * N * (4 if mode & MODE_RAW else 1)
        # Each output byte.
        for i in range(nb):
            words.append((0x03 << 24) | ((Y_ADDR + i) << 8) | sim.Y[i])
        # Clock count.
        if check_cycles:
            words.append(0x05 << 24)
            words.append(sim.cycle_count)


# Random int8 bytes.
def rbytes(n):
    # Uniform bytes.
    return bytes(rng.randrange(256) for _ in range(n))


# Random parameter records for n channels.
def rparams(n, wild=False):
    # Records.
    out = b""
    for _ in range(n):
        # Bias anywhere in a realistic range (or full int32 when wild).
        bias = rng.randrange(-2 ** 31, 2 ** 31) if wild else rng.randrange(-20000, 20000)
        # Normalised Q31 multiplier, sometimes zero.
        mult = 0 if rng.random() < 0.05 else rng.randrange(1 << 30, 1 << 31)
        # TFLite shift range; mostly negative as in real models.
        shift = rng.randrange(-31, 31) if wild else rng.randrange(-14, 1)
        # Packed record.
        out += pack_param(bias, mult, shift)
    # All records.
    return out


# Directed: one tiny FC, known numbers.
wblock(X_ADDR, bytes([1, 2, 3, 0xFF]))
wblock(W_ADDR, bytes([1, 1, 1, 1, 2, 0xFE, 3, 0xFD]))
wblock(P_ADDR, pack_param(10, 1 << 30, 0) + pack_param(-5, (1 << 31) - 1, -1))
job(4, 2, 1, 4, 0, 0, 0, 0, -128, 127, 0)
# Directed: raw int32 output.
job(4, 2, 1, 4, 0, 0, 3, 0, -128, 127, MODE_RAW)
# Directed: error cases (empty job, X overrun, W overrun, P overrun, Y overrun).
job(0, 1, 1, 1, 0, 0, 0, 0, -128, 127, 0)
job(4, 1, 1 << X_AW, 4, 0, 0, 0, 0, -128, 127, 0, check_cycles=False)
job(4, 1, 1, 4, (1 << W_AW) - 2, 0, 0, 0, -128, 127, 0, check_cycles=False)
job(4, 1, 1, 4, 0, (1 << P_AW) // 16, 0, 0, -128, 127, 0, check_cycles=False)
job(1, 1, (1 << Y_AW) + 1, 1, 0, 0, 0, 0, -128, 127, 0, check_cycles=False)

# Random jobs within the memory sizes.
for j in range(JOBS):
    # Depthwise / raw / wild parameters now and then.
    dw = rng.random() < 0.3
    raw = rng.random() < 0.15
    wild = rng.random() < 0.3
    # Sizes that fit.
    K = rng.randrange(1, 40)
    N = rng.randrange(1, 9)
    ob = 4 if raw else 1
    xs = K * N if dw else K + rng.randrange(0, 4)
    nvec = rng.randrange(1, max(2, min((1 << X_AW) // xs, (1 << Y_AW) // (N * ob)) + 1))
    nvec = min(nvec, 12)
    wb = rng.randrange(0, (1 << W_AW) - N * K + 1)
    pb = rng.randrange(0, (1 << P_AW) // 16 - N + 1)
    # Data.
    wblock(X_ADDR, rbytes(nvec * xs))
    wblock(W_ADDR + wb, rbytes(N * K))
    wblock(P_ADDR + 16 * pb, rparams(N, wild))
    # Quantisation fields.
    ioff = rng.randrange(-127, 129)
    zp = rng.randrange(-128, 128)
    lo = rng.randrange(-128, 128)
    hi = rng.randrange(lo, 128)
    # Emit.
    job(K, N, nvec, xs, wb, pb, ioff, zp, lo, hi, (MODE_DW if dw else 0) | (MODE_RAW if raw else 0))

# Read-back of the config block and identification registers.
for a in range(0x100, 0x112):
    words.append((0x03 << 24) | (a << 8) | sim.read(a))
for a in range(0x130, 0x136):
    words.append((0x03 << 24) | (a << 8) | sim.read(a))
# End marker.
words.append(0xFF << 24)

# Write the file next to this script.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_vectors.hex"), "w") as f:
    f.write("\n".join("%08X" % (w & 0xFFFFFFFF) for w in words) + "\n")
# Summary.
print("wrote %d words, %d jobs (seed %d)" % (len(words), JOBS + 7, SEED))
