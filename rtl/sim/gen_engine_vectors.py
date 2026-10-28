# gen_engine_vectors.py - random and corner-case jobs for tb_engine.v, with expected results from
# tinyml_fpga.engine.SimBus (the register model the runtime is verified against LiteRT with).
# Weights and parameters are placed through the real driver (HwEngine.upload_w / upload_p), so the
# bank layout the driver uses is checked against the RTL as well.
#
# Output: engine_vectors.hex, one 32-bit word per line:
#   01_AAAA_DD  write byte DD to address AAAA
#   02_000000   start the job (CTRL = 1) and wait for STATUS.done
#   03_AAAA_DD  read address AAAA, expect DD
#   04_0000_DD  expect STATUS == DD
#   05_000000   expect CYCLES == the next word
#   FF_000000   end of test
#   06_000000   start the descriptor queue (CTRL = 2) and wait for STATUS.done
# usage: python3 gen_engine_vectors.py [seed] [jobs] [lanes] [rq_mode] [queue] [perf]
# The memory sizes below must match the parameters of tb_engine.v.

import os
import random
import struct
import sys

# Make the package importable from the repo.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "firmware"))
from tinyml_fpga.engine import (SimBus, HwEngine, config_frame, descriptor_bytes, X_ADDR, Y_ADDR, D_ADDR, R_CFG, R_QCNT, R_QDONE,
                                R_PERF, DESC_BYTES, MODE_DW, MODE_RAW)
from tinyml_fpga.kernels import pack_param

# RAM sizes of the testbench build (total over all banks).
W_AW, X_AW, Y_AW, P_AW = 10, 9, 8, 8
# Seed, job count, lanes and requantiser mode.
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
LANES = int(sys.argv[3]) if len(sys.argv) > 3 else 1
RQ_MODE = int(sys.argv[4]) if len(sys.argv) > 4 else 0
QUEUE = int(sys.argv[5]) if len(sys.argv) > 5 else 0
PERF = int(sys.argv[6]) if len(sys.argv) > 6 else 0
# Random generator.
rng = random.Random(SEED)
# Output words.
words = []
# Golden model.
sim = SimBus(W_AW, X_AW, Y_AW, P_AW, LANES, RQ_MODE, queue=bool(QUEUE), d_aw=9, perf=bool(PERF))


# A bus that forwards to the model and records every written byte as a testbench word.
class RecBus:
    # Block write: model plus vectors.
    def write_block(self, addr, data):
        # Each byte.
        for i, b in enumerate(data):
            # Command word.
            words.append((0x01 << 24) | ((addr + i) << 8) | b)
        # Same writes into the model (config frames never include CTRL start here).
        sim.write_block(addr, data)

    # Block read: from the model.
    def read_block(self, addr, n):
        return sim.read_block(addr, n)


# Driver on the recording bus: the same code the runtime uses to place weights.
eng = HwEngine(RecBus())
# Bytes per bank and records per bank.
BW = (1 << W_AW) // LANES
BP = ((1 << P_AW) // 16) // LANES


# Emit one job and its expected results.
def job(K, N, nvec, xs, wb, pb, ioff, zp, amin, amax, mode, check_cycles=True):
    # Configuration without the trailing start byte.
    cfg = config_frame(K, N, nvec, xs, wb, pb, ioff, zp, amin, amax, mode)[:0x12]
    # Config writes.
    eng.bus.write_block(R_CFG, cfg)
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
        # Performance counters.
        if PERF:
            for a in range(R_PERF, R_PERF + 16):
                words.append((0x03 << 24) | (a << 8) | sim.read(a))


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


# Directed: one tiny FC, known numbers (2 channels, K = 4).
eng.bus.write_block(X_ADDR, bytes([1, 2, 3, 0xFF]))
eng.upload_w(0, bytes([1, 1, 1, 1, 2, 0xFE, 3, 0xFD]), 4)
eng.upload_p(0, pack_param(10, 1 << 30, 0) + pack_param(-5, (1 << 31) - 1, -1))
job(4, 2, 1, 4, 0, 0, 0, 0, -128, 127, 0)
# Directed: raw int32 output.
job(4, 2, 1, 4, 0, 0, 3, 0, -128, 127, MODE_RAW)
# Directed: error cases (empty job, X overrun, W overrun, P overrun, Y overrun).
job(0, 1, 1, 1, 0, 0, 0, 0, -128, 127, 0)
job(4, 1, 1 << X_AW, 4, 0, 0, 0, 0, -128, 127, 0, check_cycles=False)
job(4, 1, 1, 4, BW - 2, 0, 0, 0, -128, 127, 0, check_cycles=False)
job(4, 1, 1, 4, 0, BP, 0, 0, -128, 127, 0, check_cycles=False)
job(1, 1, (1 << Y_AW) + 1, 1, 0, 0, 0, 0, -128, 127, 0, check_cycles=False)

# Random jobs within the memory sizes.
for j in range(JOBS):
    # Depthwise / raw / wild parameters now and then.
    dw = rng.random() < 0.3
    raw = rng.random() < 0.15
    wild = rng.random() < 0.3
    # Output channels and rows per bank.
    N = min(rng.randrange(1, 9 + 3 * LANES), LANES * BP)
    rows = (N + LANES - 1) // LANES
    ob = 4 if raw else 1
    # Dot length such that the rows fit in a bank.
    K = rng.randrange(1, max(2, min(40, BW // rows) + 1))
    # X stride and vector count that fit X and Y.
    xs = K * N if dw else K + rng.randrange(0, 4)
    if xs > (1 << X_AW) or N * ob > (1 << Y_AW):
        # Too big for this build: shrink to a one-vector, non-depthwise job.
        dw, xs, N = False, K, min(N, (1 << Y_AW) // ob)
        rows = (N + LANES - 1) // LANES
    nvec = rng.randrange(1, max(2, min((1 << X_AW) // xs, (1 << Y_AW) // (N * ob)) + 1))
    nvec = min(nvec, 12)
    wb = rng.randrange(0, BW - rows * K + 1)
    pb = rng.randrange(0, BP - rows + 1)
    # Data, placed the way the runtime does.
    eng.bus.write_block(X_ADDR, rbytes(nvec * xs))
    eng.upload_w(wb, rbytes(N * K), K)
    eng.upload_p(pb, rparams(N, wild))
    # Quantisation fields.
    ioff = rng.randrange(-127, 129)
    zp = rng.randrange(-128, 128)
    lo = rng.randrange(-128, 128)
    hi = rng.randrange(lo, 128)
    # Emit.
    job(K, N, nvec, xs, wb, pb, ioff, zp, lo, hi, (MODE_DW if dw else 0) | (MODE_RAW if raw else 0))


# Descriptor queue runs: several independent jobs started with one command.
def queue_run(njobs, bad_at=None):
    # Allocation cursors in X, Y, W (per bank) and P (per bank).
    xa = ya = wa = pa = 0
    # Descriptors and what to check afterwards.
    descs, checks = [], []
    for j in range(njobs):
        # Small random job that is not depthwise-heavy.
        dw = rng.random() < 0.25
        raw = rng.random() < 0.15
        N = rng.randrange(1, 8)
        K = rng.randrange(1, 18)
        rows = (N + LANES - 1) // LANES
        ob = 4 if raw else 1
        xs = K * N if dw else K
        nvec = rng.randrange(1, 4)
        # Does it still fit?
        if (xa + nvec * xs > (1 << X_AW) or ya + nvec * N * ob > (1 << Y_AW) or wa + rows * K > BW or pa + rows > BP):
            break
        # Data at this job's own place.
        eng.bus.write_block(X_ADDR + xa, rbytes(nvec * xs))
        eng.upload_w(wa, rbytes(N * K), K)
        eng.upload_p(pa, rparams(N, rng.random() < 0.3))
        ioff = rng.randrange(-127, 129)
        zp = rng.randrange(-128, 128)
        lo = rng.randrange(-128, 128)
        hi = rng.randrange(lo, 128)
        mode = (MODE_DW if dw else 0) | (MODE_RAW if raw else 0)
        # A deliberately empty job to test the error path.
        Kd = 0 if bad_at == j else K
        descs.append(descriptor_bytes(Kd, N, nvec, xs, wa, pa, ioff, zp, lo, hi, mode, xa, ya))
        checks.append((ya, nvec * N * ob))
        xa += nvec * xs
        ya += nvec * N * ob
        wa += rows * K
        pa += rows
    # Write the descriptors (64 bytes per slot) and the count; then start.
    blob = b"".join(d + bytes(DESC_BYTES - len(d)) for d in descs)
    eng.bus.write_block(D_ADDR, blob)
    eng.bus.write_block(R_QCNT, bytes([len(descs)]))
    sim.write(0x11F, 2)
    words.append(0x06 << 24)
    words.append((0x04 << 24) | sim.status)
    # QDONE.
    words.append((0x03 << 24) | (R_QDONE << 8) | sim.read(R_QDONE))
    if sim.status == 2:
        # Every job's outputs.
        for ya_, n in checks:
            for i in range(n):
                words.append((0x03 << 24) | ((Y_ADDR + ya_ + i) << 8) | sim.Y[ya_ + i])
        # Clock count.
        words.append(0x05 << 24)
        words.append(sim.cycle_count)
        # Performance counters.
        if PERF:
            for a in range(R_PERF, R_PERF + 16):
                words.append((0x03 << 24) | (a << 8) | sim.read(a))
    else:
        # Clear the error so the next run starts clean.
        eng.bus.write_block(0x11F, bytes([0x80]))
        sim.write(0x11F, 0x80)


# Queue tests.
if QUEUE:
    for k in range(10):
        queue_run(rng.randrange(1, 6))
    # Error cases: a failing job in the middle (reports how many finished), zero count, too many descriptors.
    queue_run(4, bad_at=2)
    eng.bus.write_block(R_QCNT, bytes([0]))
    sim.write(0x11F, 2)
    words.append(0x06 << 24)
    words.append((0x04 << 24) | sim.status)
    eng.bus.write_block(0x11F, bytes([0x80]))
    sim.write(0x11F, 0x80)
    eng.bus.write_block(R_QCNT, bytes([9]))
    sim.write(0x11F, 2)
    words.append(0x06 << 24)
    words.append((0x04 << 24) | sim.status)
    eng.bus.write_block(0x11F, bytes([0x80]))
    sim.write(0x11F, 0x80)

# Read-back of the config block and identification registers.
for a in range(0x100, 0x112):
    words.append((0x03 << 24) | (a << 8) | sim.read(a))
for a in range(0x130, 0x139):
    words.append((0x03 << 24) | (a << 8) | sim.read(a))
# End marker.
words.append(0xFF << 24)

# Write the file next to this script.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_vectors.hex"), "w") as f:
    f.write("\n".join("%08X" % (w & 0xFFFFFFFF) for w in words) + "\n")
# Summary.
print("wrote %d words, %d jobs (seed %d, lanes %d, rq_mode %d, queue %d, perf %d)" % (len(words), JOBS + 7, SEED, LANES, RQ_MODE, QUEUE, PERF))
