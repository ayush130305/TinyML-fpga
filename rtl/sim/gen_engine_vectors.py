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
# usage: python3 gen_engine_vectors.py [seed] [jobs] [lanes] [rq_mode] [queue] [perf] [win] [fuse]
# The memory sizes below must match the parameters of tb_engine.v.

import os
import random
import struct
import sys

# Make the package importable from the repo.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "firmware"))
from tinyml_fpga.engine import (SimBus, HwEngine, config_frame, descriptor_bytes, X_ADDR, Y_ADDR, D_ADDR, R_CFG, R_QCNT, R_QDONE,
                                R_PERF, DESC_BYTES, MODE_DW, MODE_RAW, MODE_WIN, MODE_OUTX, R_WIN, WIN_FMT, NCFG, NWIN, window_block)
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
WIN = int(sys.argv[7]) if len(sys.argv) > 7 else 0
FUSE = int(sys.argv[8]) if len(sys.argv) > 8 else 0
# Random generator.
rng = random.Random(SEED)
# Output words.
words = []
# Golden model.
sim = SimBus(W_AW, X_AW, Y_AW, P_AW, LANES, RQ_MODE, queue=bool(QUEUE), d_aw=9, perf=bool(PERF), win=bool(WIN), fuse=bool(FUSE))


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


# One job with every field: xbase / ybase, optional window block, outputs read back from Y or (MODE_OUTX) from X.
def full_job(K, N, nvec, xs, wb, pb, ioff, zp, amin, amax, mode, xbase, ybase, win=None):
    # Window block first (when used), then the config bytes up to YBASE.
    if win is not None:
        eng.bus.write_block(R_WIN, struct.pack(WIN_FMT, *win))
    eng.bus.write_block(R_CFG, config_frame(K, N, nvec, xs, wb, pb, ioff, zp, amin, amax, mode, xbase, ybase)[:0x16])
    sim.write(0x11F, 1)
    words.append(0x02 << 24)
    words.append((0x04 << 24) | sim.status)
    if sim.status == 2:
        # Output memory and its window address.
        mem, base = (sim.X, X_ADDR) if mode & MODE_OUTX else (sim.Y, Y_ADDR)
        for i in range(nvec * N * (4 if mode & MODE_RAW else 1)):
            words.append((0x03 << 24) | ((base + ybase + i) << 8) | mem[ybase + i])
        words.append(0x05 << 24)
        words.append(sim.cycle_count)
        if PERF:
            for a in range(R_PERF, R_PERF + 16):
                words.append((0x03 << 24) | (a << 8) | sim.read(a))


# A random convolution / depthwise layer on an h x w x c image: returns a dict, or None if it has no output.
def rand_layer(h, w, c, dw):
    kh, kw = rng.randrange(1, 4), rng.randrange(1, 4)
    dh, dwd = (1 if rng.random() < 0.7 else 2), (1 if rng.random() < 0.7 else 2)
    sh, sw = rng.randrange(1, 3), rng.randrange(1, 3)
    ph, pw = rng.randrange(0, 3), rng.randrange(0, 3)
    oh = (h + 2 * ph - ((kh - 1) * dh + 1)) // sh + 1
    ow = (w + 2 * pw - ((kw - 1) * dwd + 1)) // sw + 1
    if oh < 1 or ow < 1:
        return None
    dm = rng.randrange(1, 3) if dw else 1
    n = c * dm if dw else rng.randrange(1, 6 + LANES)
    r0 = rng.randrange(oh)
    rows = rng.randrange(1, min(oh - r0, 4) + 1)
    return dict(dw=dw, h=h, w=w, c=c, kh=kh, kw=kw, dh=dh, dwd=dwd, sh=sh, sw=sw, ph=ph, pw=pw, dm=dm, n=n,
                k=kh * kw * (1 if dw else c), ow=ow, r0=r0, rows=rows, nvec=rows * ow)


# Random quantisation fields.
def rquant():
    lo = rng.randrange(-128, 128)
    return rng.randrange(-127, 129), rng.randrange(-128, 128), lo, rng.randrange(lo, 128)


# Window jobs (one per random layer) and, on builds with write-back, plain jobs that write into X.
if WIN:
    for j in range(max(10, JOBS // 2)):
        L = rand_layer(rng.randrange(1, 7), rng.randrange(1, 7), rng.randrange(1, 4), rng.random() < 0.35)
        if L is None:
            continue
        rows = (L["n"] + LANES - 1) // LANES
        if rows * L["k"] > BW or rows > BP or L["nvec"] * L["n"] > (1 << Y_AW):
            continue
        img = L["h"] * L["w"] * L["c"]
        # Image somewhere in X (sometimes too far up: the XLEN check must reject the job).
        xbase = rng.randrange(0, 40) if rng.random() < 0.9 else (1 << X_AW) - rng.randrange(0, img)
        outx = bool(FUSE) and rng.random() < 0.4
        ybase = xbase + img + rng.randrange(0, 8) if outx else rng.randrange(0, 8)
        win = window_block(L["h"], L["w"], L["c"], L["kh"], L["kw"], L["dh"], L["dwd"], L["sh"], L["sw"], L["ph"], L["pw"],
                           L["dm"], L["ow"], L["r0"], xbase)
        if win is None:
            continue
        wb, pb = rng.randrange(0, BW - rows * L["k"] + 1), rng.randrange(0, BP - rows + 1)
        eng.bus.write_block(X_ADDR + xbase, rbytes(min(img, (1 << X_AW) - xbase)))
        eng.upload_w(wb, rbytes(L["n"] * L["k"]), L["k"])
        eng.upload_p(pb, rparams(L["n"], rng.random() < 0.3))
        ioff, zp, lo, hi = rquant()
        full_job(L["k"], L["n"], L["nvec"], 0, wb, pb, ioff, zp, lo, hi,
                 MODE_WIN | (MODE_DW if L["dw"] else 0) | (MODE_OUTX if outx else 0), xbase, ybase, win)
elif FUSE:
    for j in range(max(8, JOBS // 3)):
        N, K = rng.randrange(1, 8), rng.randrange(1, 20)
        rows = (N + LANES - 1) // LANES
        nvec = rng.randrange(1, 5)
        if rows * K > BW or rows > BP:
            continue
        xbase, ybase = rng.randrange(0, 20), 100 + rng.randrange(0, 20)
        eng.bus.write_block(X_ADDR + xbase, rbytes(nvec * K))
        eng.upload_w(0, rbytes(N * K), K)
        eng.upload_p(0, rparams(N))
        ioff, zp, lo, hi = rquant()
        full_job(K, N, nvec, K, 0, 0, ioff, zp, lo, hi, MODE_OUTX | (MODE_RAW if rng.random() < 0.2 else 0), xbase, ybase)
    # Output beyond X: must fail.
    eng.upload_w(0, rbytes(4), 4)
    eng.upload_p(0, rparams(1))
    full_job(4, 1, 1, 4, 0, 0, 0, 0, -128, 127, MODE_OUTX, 0, (1 << X_AW) - 0, None)


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



# A chain of layers run by one queue start: window layers and fully-connected layers, each writing its outputs
# into X where the next one reads them; the last one writes to Y (or to X when fusion is the only feature).
def chain_run(nlayers):
    # Image of the first layer at the start of X.
    h, w, c = rng.randrange(2, 7), rng.randrange(2, 7), rng.randrange(1, 4)
    xcur, xfree = 0, h * w * c
    eng.bus.write_block(X_ADDR, rbytes(h * w * c))
    descs, regions, wa, pa = [], [], 0, 0
    for li in range(nlayers):
        last = li == nlayers - 1
        # Layer kind: window layer (needs WIN) or fully connected over the whole previous output.
        if WIN and rng.random() < 0.7:
            L = rand_layer(h, w, c, rng.random() < 0.3)
            if L is None:
                break
            K, N, nvec, xs = L["k"], L["n"], L["nvec"], 0
            win = window_block(h, w, c, L["kh"], L["kw"], L["dh"], L["dwd"], L["sh"], L["sw"], L["ph"], L["pw"], L["dm"],
                               L["ow"], L["r0"], xcur)
            mode, oh, ow_ = MODE_WIN | (MODE_DW if L["dw"] else 0), L["rows"], L["ow"]
            if win is None:
                break
        else:
            K, N, nvec, xs, win = h * w * c, rng.randrange(1, 7), 1, h * w * c, None
            mode, oh, ow_ = 0, 1, 1
        rows_ = (N + LANES - 1) // LANES
        ob = N * nvec
        # Where the outputs go: next free X bytes, or Y for the last layer when fusion is not needed.
        to_x = FUSE and (not last or not QUEUE or rng.random() < 0.3)
        ybase = xfree if to_x else 0
        if (wa + rows_ * K > BW or pa + rows_ > BP or K > (1 << X_AW) or xfree + ob > (1 << X_AW) or ob > (1 << Y_AW)
                or (not to_x and not last)):
            break
        eng.upload_w(wa, rbytes(N * K), K)
        eng.upload_p(pa, rparams(N))
        ioff, zp, lo, hi = rquant()
        descs.append(descriptor_bytes(K, N, nvec, xs, wa, pa, ioff, zp, lo, hi, mode | (MODE_OUTX if to_x else 0), xcur, ybase, win))
        regions.append((to_x, ybase, ob))
        wa += rows_ * K
        pa += rows_
        # The next layer reads this layer's outputs as an oh x ow_ x N image.
        xcur, xfree, h, w, c = ybase, xfree + ob, oh, ow_, N
        if not to_x:
            break
    if not descs:
        return
    blob = b"".join(d + bytes(max(DESC_BYTES, len(d)) - len(d)) for d in descs)
    # Window builds store the window block in every descriptor (zeros where unused).
    if WIN:
        blob = b"".join((d + bytes(NCFG + NWIN - len(d)) if len(d) == NCFG else d) + bytes(DESC_BYTES - max(len(d), NCFG + NWIN)) for d in descs)
    eng.bus.write_block(D_ADDR, blob)
    eng.bus.write_block(R_QCNT, bytes([len(descs)]))
    sim.write(0x11F, 2)
    words.append(0x06 << 24)
    words.append((0x04 << 24) | sim.status)
    words.append((0x03 << 24) | (R_QDONE << 8) | sim.read(R_QDONE))
    if sim.status == 2:
        for to_x, yb, n in regions:
            mem, base = (sim.X, X_ADDR) if to_x else (sim.Y, Y_ADDR)
            for i in range(n):
                words.append((0x03 << 24) | ((base + yb + i) << 8) | mem[yb + i])
        words.append(0x05 << 24)
        words.append(sim.cycle_count)
        if PERF:
            for a in range(R_PERF, R_PERF + 16):
                words.append((0x03 << 24) | (a << 8) | sim.read(a))
    else:
        eng.bus.write_block(0x11F, bytes([0x80]))
        sim.write(0x11F, 0x80)


# Queue tests.
if QUEUE:
    for k in range(12 if (WIN or FUSE) else 0):
        chain_run(rng.randrange(1, 5))
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
print("wrote %d words, %d jobs (seed %d, lanes %d, rq_mode %d, queue %d, perf %d, win %d, fuse %d)" % (len(words), JOBS + 7, SEED, LANES, RQ_MODE, QUEUE, PERF, WIN, FUSE))
