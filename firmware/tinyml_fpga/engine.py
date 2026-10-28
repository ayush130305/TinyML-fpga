# engine.py - the int8 dot-product engine: register map, a software engine and the hardware driver.
# MicroPython-safe. The same map is implemented by rtl/tmf_engine.v and by SimBus below.
#
# Byte address map (as seen on the engine's byte bus, e.g. through the UART bridge):
#   0x100 K        u16   dot-product length
#   0x102 N        u16   output channels
#   0x104 NVEC     u16   input vectors per job
#   0x106 XSTRIDE  u16   bytes between input vectors in X
#   0x108 WBASE    u16   byte offset of channel 0's weight row in W
#   0x10A PBASE    u16   index of channel 0's 16-byte parameter record in P
#   0x10C IN_OFF   s16   added to every input byte (= -input zero point)
#   0x10E OUT_ZP   s8    output zero point
#   0x10F ACT_MIN  s8    clamp low
#   0x110 ACT_MAX  s8    clamp high
#   0x111 MODE     u8    bit0 depthwise (channel o uses x[o*K..]), bit1 raw int32 output
#   0x112 XBASE    u16   byte offset of the job's first input vector in X (version 2)
#   0x114 YBASE    u16   byte offset of the job's first output in Y (version 2)
#   0x140 window block, 31 bytes, write-only (builds with the window generator, MODE bit 2): see WIN_FMT
#   0x118 QCNT     u8    descriptors to run with CTRL = 0x02 (builds with the queue)
#   0x119 QDONE    u8    descriptors finished by the last queue run (RO)
#   0x11F CTRL     u8    write 0x01 = start, 0x02 = run the descriptor queue, 0x80 = clear status
#                        (put it last in the config frame)
#   0x120 STATUS   u8    bit0 busy, bit1 done, bit2 error
#   0x121 CYCLES   u32   clocks the last job took
#   0x130 MAGIC 0x4D, 0x131 VERSION, 0x132 W_AW, 0x133 X_AW, 0x134 Y_AW, 0x135 P_AW  (read-only)
#   0x136 LANES, 0x137 FEATURES (bit0 pipelined requantiser, bit1 queue, bit2 window generator, bit3 fusion,
#         bit6 perf counters), 0x138 D_AW
#   0x160 PERF  four u32: MAC clocks, stall clocks, output bytes, jobs (builds with HAS_PERF)
#   0x4000 D    job descriptors, 64 bytes each: bytes 0..21 are the config block 0x100..0x115, bytes 22..52 the
#               window block 0x140..0x15E (builds with the window generator)
#   0x1000 X   0x2000 Y   0x3000 P   0x8000 W      memory windows (sizes are 2^*_AW bytes)
#
# With LANES > 1 the W and P windows are split into LANES equal banks selected by the top address bits.
# Channel o of a layer lives in bank o % LANES at local row o // LANES; WBASE and PBASE are local (per-bank)
# offsets. upload_w / upload_p below do this layout, callers only pass whole layers.

import struct

from .kernels import PARAM_BYTES, dot_requant, to_i8_list

# Register addresses.
R_CFG, R_CTRL, R_STATUS, R_CYCLES = 0x100, 0x11F, 0x120, 0x121
# Queue registers, descriptor window, performance counters.
R_QCNT, R_QDONE, D_ADDR, R_PERF = 0x118, 0x119, 0x4000, 0x160
# Bytes of config in a descriptor and bytes per descriptor slot.
NCFG, DESC_BYTES = 22, 64
# Window block: address, size and layout (little endian):
#   H W C | KH KW DH DW SH SW DM | OW | IY0 IX0 ORG | SWC SHWC KXS KYS | XLEN
# IY0 / IX0 = image row / column of the first window's top-left tap, ORG = its X address (XBASE included),
# SWC / SHWC = X bytes per output column / row step, KXS / KYS = X bytes per kernel column / row step,
# XLEN = bytes of image the job may read (checked against X).
R_WIN, NWIN, WIN_FMT = 0x140, 31, "<HHHBBBBBBBHhhhHHHHH"
# Read-only identification block.
R_MAGIC = 0x130
# Memory windows.
X_ADDR, Y_ADDR, P_ADDR, W_ADDR = 0x1000, 0x2000, 0x3000, 0x8000
# Engine magic number.
MAGIC = 0x4D
# Register-map versions this driver speaks (1: single lane, 2: adds LANES and FEATURES).
VERSIONS = (1, 2)
# Version written by the current RTL and SimBus.
VERSION = 2
# FEATURES bits.
F_RQ_PIPE, F_QUEUE, F_WIN, F_FUSE, F_PERF = 1, 2, 4, 8, 0x40
# STATUS bits.
BUSY, DONE, ERR = 1, 2, 4
# MODE bits.
MODE_DW, MODE_RAW, MODE_WIN, MODE_OUTX = 1, 2, 4, 8
# CTRL commands.
CMD_START, CMD_QUEUE, CMD_CLEAR = 0x01, 0x02, 0x80


# Pack the configuration block 0x100..0x11F with CTRL=start as its last byte; descriptor_bytes() gives the
# 22 config bytes alone (what a queue descriptor holds).
def descriptor_bytes(K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode, xbase=0, ybase=0, win=None):
    # Same struct as the config frame.
    d = struct.pack("<HHHHHHhbbbBHH", K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode, xbase, ybase)
    # Window block (builds with the window generator): 31 more bytes.
    if win is not None:
        d += struct.pack(WIN_FMT, *win)
    return d


# Pack the configuration block 0x100..0x11F with CTRL=start as its last byte.
def config_frame(K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode, xbase=0, ybase=0):
    # 0x100..0x115 in one struct, little endian.
    head = struct.pack("<HHHHHHhbbbBHH", K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode, xbase, ybase)
    # Pad up to 0x11F, then the start command, so the job starts when the frame's last byte lands.
    return head + bytes(0x1F - len(head)) + bytes([CMD_START])


# Software engine: same interface as HwEngine, runs dot_requant in Python. Used for backend="cpu".
class CpuEngine:
    # Name shown in summaries.
    name = "cpu"
    # No hardware features.
    has_win = has_fuse = has_queue = False
    # The software engine has one contiguous weight store.
    lanes = 1

    # Unlimited capacities (in practice: RAM).
    def __init__(self, w_cap=1 << 24, x_cap=1 << 24, y_cap=1 << 24, p_cap=1 << 20):
        # Capacities in bytes / channels.
        self.w_cap, self.x_cap, self.y_cap, self.p_cap = w_cap, x_cap, y_cap, p_cap
        # Weight store.
        self.W = bytearray()
        # Parameter store.
        self.P = bytearray()
        # Cycle count is not meaningful here.
        self.cycles = 0

    # Store weight bytes at a byte offset (k, the row length, is only needed by banked hardware).
    def upload_w(self, off, data, k=None):
        # Grow the store if needed.
        if len(self.W) < off + len(data):
            self.W += bytes(off + len(data) - len(self.W))
        # Copy.
        self.W[off:off + len(data)] = data

    # Store parameter records starting at a record index.
    def upload_p(self, ch, data):
        # Byte offset of the record.
        off = ch * PARAM_BYTES
        # Grow the store if needed.
        if len(self.P) < off + len(data):
            self.P += bytes(off + len(data) - len(self.P))
        # Copy.
        self.P[off:off + len(data)] = data

    # Run one job; x is a bytes-like buffer of int8 data; returns a list of ints.
    def run(self, K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode, x, xbase=0, ybase=0, win=None):
        # The software engine takes plain vectors only (the model builds them on the host).
        if mode & (MODE_WIN | MODE_OUTX):
            raise ValueError("the software engine has no window generator or fusion")
        # Weights of the N channels.
        w = to_i8_list(self.W[wbase:wbase + N * K])
        # Their parameter records.
        p = self.P[pbase * PARAM_BYTES:(pbase + N) * PARAM_BYTES]
        # Compute.
        return dot_requant(to_i8_list(x), w, p, K, N, nvec, xstride, in_off, out_zp, act_min, act_max,
                           dw=bool(mode & MODE_DW), raw=bool(mode & MODE_RAW))


# Hardware engine driver over any byte bus with write_block(addr, buf) and read_block(addr, n).
class HwEngine:
    # Name shown in summaries.
    name = "fpga"

    # Probe the engine and read its memory sizes.
    def __init__(self, bus, timeout_polls=100000):
        # Byte bus.
        self.bus = bus
        # Status polls before a job is declared hung.
        self.timeout_polls = timeout_polls
        # Identification block (the extra bytes read as 0 on version 1 builds).
        ident = bus.read_block(R_MAGIC, 8)
        # Wrong or missing bitstream.
        if ident[0] != MAGIC:
            raise RuntimeError("tinyml engine not found (magic 0x%02X, expected 0x%02X): is the bitstream loaded?" % (ident[0], MAGIC))
        # Newer register map than this driver knows.
        if ident[1] not in VERSIONS:
            raise RuntimeError("engine register map v%d, driver speaks v%s" % (ident[1], "/".join(str(v) for v in VERSIONS)))
        # Parallel output channels (1 on older builds).
        self.lanes = max(1, ident[6]) if ident[1] >= 2 else 1
        # Optional hardware features.
        self.features = ident[7] if ident[1] >= 2 else 0
        # Descriptor queue and performance counters; the number of descriptor slots.
        self.has_queue, self.has_perf = bool(self.features & F_QUEUE), bool(self.features & F_PERF)
        # Window generator and fusion (outputs into X).
        self.has_win, self.has_fuse = bool(self.features & F_WIN), bool(self.features & F_FUSE)
        # Config bytes in a descriptor of this build.
        self.ncfg = NCFG + (NWIN if self.has_win else 0)
        self.queue_slots = (1 << bus.read_block(R_MAGIC + 8, 1)[0]) // DESC_BYTES if self.has_queue else 0
        # Weight bytes and parameter records per bank (the whole memory when there is one lane).
        self.w_cap = (1 << ident[2]) // self.lanes
        self.p_cap = ((1 << ident[5]) // PARAM_BYTES) // self.lanes
        # Input and output memories are shared by all lanes.
        self.x_cap, self.y_cap = 1 << ident[3], 1 << ident[4]
        # Bytes per bank window.
        self._wbank, self._pbank = self.w_cap, self.p_cap * PARAM_BYTES
        # Clock count of the last job.
        self.cycles = 0
        # Clear any stale status.
        bus.write_block(R_CTRL, bytes([CMD_CLEAR]))

    # Write the weight rows of channels 0.. (k bytes each, consecutive in data) at local byte offset off.
    def upload_w(self, off, data, k=None):
        # One lane: a single contiguous block.
        if self.lanes == 1:
            self.bus.write_block(W_ADDR + off, data)
            return
        # Several lanes: channel c goes to bank c % lanes, local row c // lanes; one block per bank.
        n = len(data) // k
        for b in range(self.lanes):
            # Rows of this bank, back to back (in MicroPython this stays bytes-only).
            rows = bytearray()
            for c in range(b, n, self.lanes):
                rows += data[c * k:(c + 1) * k]
            # Skip banks that got no row.
            if rows:
                self.bus.write_block(W_ADDR + b * self._wbank + off, rows)

    # Write parameter records of channels 0.. at local record index rec.
    def upload_p(self, rec, data):
        # One lane: a single contiguous block.
        if self.lanes == 1:
            self.bus.write_block(P_ADDR + rec * PARAM_BYTES, data)
            return
        # Several lanes: record c goes to bank c % lanes, local record rec + c // lanes.
        n = len(data) // PARAM_BYTES
        for b in range(self.lanes):
            # Records of this bank, back to back.
            recs = bytearray()
            for c in range(b, n, self.lanes):
                recs += data[c * PARAM_BYTES:(c + 1) * PARAM_BYTES]
            # Skip banks that got no record.
            if recs:
                self.bus.write_block(P_ADDR + b * self._pbank + rec * PARAM_BYTES, recs)

    # Poll STATUS until the job (or queue) is done; remember the clock count; raise on error or timeout.
    def _wait(self):
        # Bus shortcut.
        bus = self.bus
        # Poll the status register.
        for _ in range(self.timeout_polls):
            # Status, then the cycle counter, in one read.
            st = bus.read_block(R_STATUS, 5)
            # Error: bad config or a start while busy.
            if st[0] & ERR:
                raise RuntimeError("engine error (status 0x%02X)" % st[0])
            # Finished.
            if st[0] & DONE:
                # Remember the clock count.
                self.cycles = st[1] | (st[2] << 8) | (st[3] << 16) | (st[4] << 24)
                return
        # Never finished.
        raise RuntimeError("engine timeout")

    # Run one job on the FPGA and return the outputs as a list of ints.
    # x goes to X at xbase; the results are read from Y at ybase.
    def run(self, K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode, x, xbase=0, ybase=0, win=None):
        # Bus shortcut.
        bus = self.bus
        # Input vectors (or the image, in window mode) into X; x=None means they are already there.
        if x is not None:
            bus.write_block(X_ADDR + xbase, x)
        # Window block.
        if mode & MODE_WIN:
            if not self.has_win:
                raise RuntimeError("this engine build has no window generator")
            bus.write_block(R_WIN, struct.pack(WIN_FMT, *win))
        # Configuration, ending with the start command.
        bus.write_block(R_CFG, config_frame(K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode, xbase, ybase))
        # Wait.
        self._wait()
        # Raw int32 outputs.
        if mode & MODE_RAW:
            # Four bytes per output.
            y = bus.read_block(Y_ADDR + ybase, nvec * N * 4)
            # Little-endian signed words.
            return list(struct.unpack("<%di" % (nvec * N), y))
        # int8 outputs.
        return to_i8_list(bus.read_block(Y_ADDR + ybase, nvec * N))

    # Run several jobs back to back with one start. jobs is a list of job tuples
    # (K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode, xbase, ybase[, win]); their inputs
    # must already be in X and their outputs are left in Y (read them with read_y). Needs the queue feature.
    def run_queue(self, jobs):
        # Feature and capacity checks.
        if not self.has_queue:
            raise RuntimeError("this engine build has no descriptor queue")
        if len(jobs) > self.queue_slots or len(jobs) > 255:
            raise ValueError("%d jobs, queue holds %d" % (len(jobs), self.queue_slots))
        # Descriptors, 64 bytes per slot, in one block (the config bytes then padding). A job tuple may end with
        # a window tuple; builds with the window generator always store the window block (zeros if unused).
        blob = bytearray()
        for j in jobs:
            d = descriptor_bytes(*j)
            # Jobs without a window still carry the (zero) block so every slot has the build's NCFG bytes.
            if self.has_win and len(d) == NCFG:
                d += bytes(NWIN)
            blob += d + bytes(DESC_BYTES - len(d))
        # Write them, then the count and the start command in one frame (QCNT is at 0x118, CTRL at 0x11F).
        self.bus.write_block(D_ADDR, blob)
        self.bus.write_block(R_QCNT, bytes([len(jobs)]) + bytes(R_CTRL - R_QCNT - 1) + bytes([CMD_QUEUE]))
        # Wait for the whole list.
        self._wait()

    # Read bytes from Y (after run_queue).
    def read_y(self, off, n):
        return self.bus.read_block(Y_ADDR + off, n)

    # Write bytes into X (before run_queue).
    def write_x(self, off, data):
        self.bus.write_block(X_ADDR + off, data)

    # Performance counters of the last job or queue: (mac clocks, stall clocks, output bytes, jobs).
    def perf(self):
        # Needs HAS_PERF.
        if not self.has_perf:
            raise RuntimeError("this engine build has no performance counters")
        return struct.unpack("<4I", self.bus.read_block(R_PERF, 16))


# Clocks the engine takes for a job (the exact model of rtl/tmf_engine.v), with the counters of the performance
# monitor. Returns (cycles, mac_clocks, stall_clocks).
#   shifts : list of the N channels' TFLite shifts          lanes / rq_mode : the build's LANES and RQ_MODE
# Each group of `lanes` channels (one channel for depthwise) costs 1 check + 10 parameter fetches + 1 load +
# K multiply-accumulates + 3 pipeline drain + 1 step = K + 16 clocks when the output unit keeps up. The output
# unit (requantiser + Y writes) works on the finished group while the next one is computed, and the next
# group's hand-over waits until it is free. The job itself adds one set-up clock in front.
def job_stats(K, N, nvec, shifts, mode, lanes=1, rq_mode=0):
    # Clock of the current group's check state, and first clock the output unit is free.
    t, free = 0, 0
    # MAC and stall clock counters.
    mac, stall = 0, 0
    # Raw mode flag and depthwise flag.
    raw, dw = bool(mode & MODE_RAW), bool(mode & MODE_DW)
    # Every vector.
    for _ in range(nvec):
        # Channel counter inside the vector.
        o = 0
        while o < N:
            # Channels in this group.
            cnt = 1 if dw else min(lanes, N - o)
            # Hand-over clock: the MAC pipeline is empty K + 14 clocks after the check, and the unit must be idle.
            hand = max(t + K + 14, free)
            # Counters.
            mac += K
            stall += hand - (t + K + 14)
            # Raw: four byte writes per channel.
            if raw:
                free = hand + 1 + 4 * cnt
            # Pipelined requantiser: one issue per clock, 10 clocks latency.
            elif rq_mode == 1:
                free = hand + cnt + 11
            # Serial requantiser: one result at a time, 37 + ts/8 + ts%8 clocks each.
            else:
                s = hand + 1
                for i in range(cnt):
                    ts = 31 - shifts[o + i]
                    q = ts // 8 + ts % 8
                    last = (s, q)
                    s += 37 + q
                free = last[0] + 38 + last[1]
            # Next group: hand-over, step, check.
            t = hand + 2
            # Next channels.
            o += cnt
    # The job ends when the last group is handed over, stepped, and the output unit is empty; +1 for set-up.
    return max(t, free) + 2, mac, stall


# Clocks of a job as the cycle counter reports them.
def job_cycles(K, N, nvec, shifts, mode, lanes=1, rq_mode=0):
    return job_stats(K, N, nvec, shifts, mode, lanes, rq_mode)[0]


# Input vectors of a windowed job, exactly as the window generator walks them. mem holds the image at xbase.
#   conv: per output pixel the taps (ky, kx, c);  depthwise: per pixel, for every output channel its taps (ky, kx)
# of input channel oc // DM. Taps outside the image are the byte `pad` (which the input offset turns into 0).
def window_vectors(mem, xbase, wv, K, N, nvec, dw, pad):
    H, W, C, KH, KW, DH, DWD, SH, SW, DM, OW, IY0, IX0 = wv[:13]
    out = bytearray()
    for v in range(nvec):
        # First tap of this pixel.
        iy0, ix0 = IY0 + (v // OW) * SH, IX0 + (v % OW) * SW
        for oc in range(N if dw else 1):
            # Input channel of a depthwise output channel.
            ic = oc // DM if dw else 0
            for ky in range(KH):
                for kx in range(KW):
                    iy, ix = iy0 + ky * DH, ix0 + kx * DWD
                    inside = 0 <= iy < H and 0 <= ix < W
                    b = xbase + (iy * W + ix) * C
                    if dw:
                        out.append(mem[b + ic] if inside else pad)
                    elif inside:
                        out += mem[b:b + C]
                    else:
                        out += bytes([pad]) * C
    return out


# Window block of a convolution job, or None if some value does not fit the descriptor fields (then the host
# must build the patches itself). Output rows r0.. of the layer are computed; xbase is where the image sits in X.
#   H W C       input image (NHWC)              kh kw dh dw_   kernel size and dilation
#   sh sw       strides                          ph pw          padding above / left
#   dm          depth multiplier (depthwise)     ow             output pixels per row
def window_block(H, W, C, kh, kw, dh, dw_, sh, sw, ph, pw, dm, ow, r0, xbase):
    # Image row / column of the first tap and its X address.
    iy0, ix0 = r0 * sh - ph, -pw
    org = xbase + (iy0 * W + ix0) * C
    # Byte steps.
    swc, shwc, kxs, kys, xlen = sw * C, sh * W * C, dw_ * C, dh * W * C, H * W * C
    # Field ranges.
    if not (all(0 <= v < 65536 for v in (H, W, C, ow, swc, shwc, kxs, kys, xlen)) and H < 32768 and W < 32768
            and all(0 <= v < 256 for v in (kh, kw, dh, dw_, sh, sw, dm))
            and all(-32768 <= v < 32768 for v in (iy0, ix0, org))):
        return None
    return (H, W, C, kh, kw, dh, dw_, sh, sw, dm, ow, iy0, ix0, org, swc, shwc, kxs, kys, xlen)


# Python model of the RTL behind the byte bus: lets the hardware driver run on a PC with no FPGA.
# It is register-accurate (same map, same STATUS behaviour, same clock counts) and is what the RTL
# testbench is checked against.
class SimBus:
    # Default sizes match the tmf_engine.v defaults.
    def __init__(self, w_aw=12, x_aw=10, y_aw=8, p_aw=10, lanes=1, rq_mode=0, queue=False, d_aw=9, perf=False,
                 win=False, fuse=False):
        # Address widths and build options.
        self.aw = (w_aw, x_aw, y_aw, p_aw)
        self.lanes, self.rq_mode, self.queue, self.d_aw, self.perf = lanes, rq_mode, queue, d_aw, perf
        # Window generator and fusion; config bytes per descriptor.
        self.win, self.fuse = win, fuse
        self.ncfg = NCFG + (NWIN if win else 0)
        # Memories (W and P are bank-major: bank b starts at b * size / lanes).
        self.W, self.X, self.Y, self.P = bytearray(1 << w_aw), bytearray(1 << x_aw), bytearray(1 << y_aw), bytearray(1 << p_aw)
        # Descriptor RAM.
        self.D = bytearray(1 << d_aw) if queue else bytearray()
        # Config registers 0x100..0x11F, window block 0x140..0x15E.
        self.cfg = bytearray(0x20)
        self.wcfg = bytearray(NWIN)
        # STATUS, CYCLES, descriptors done, performance counters.
        self.status, self.cycle_count, self.qdone = 0, 0, 0
        self.pc = [0, 0, 0, 0]
        # Transaction counters (link traffic statistics).
        self.frames, self.bytes = 0, 0

    # Map an address to (memory, offset) or None.
    def _mem(self, a):
        # X window.
        if X_ADDR <= a < X_ADDR + len(self.X):
            return self.X, a - X_ADDR
        # Y window.
        if Y_ADDR <= a < Y_ADDR + len(self.Y):
            return self.Y, a - Y_ADDR
        # P window.
        if P_ADDR <= a < P_ADDR + len(self.P):
            return self.P, a - P_ADDR
        # D window.
        if D_ADDR <= a < D_ADDR + len(self.D):
            return self.D, a - D_ADDR
        # W window.
        if W_ADDR <= a < W_ADDR + len(self.W):
            return self.W, a - W_ADDR
        # Unmapped.
        return None

    # One byte write, exactly as the RTL decodes it.
    def write(self, a, v):
        # Window block.
        if self.win and R_WIN <= a < R_WIN + NWIN:
            if not self.status & BUSY:
                self.wcfg[a - R_WIN] = v
        # Config registers (ignored while busy, like the RTL).
        elif 0x100 <= a < 0x100 + NCFG or a == R_QCNT:
            # Store.
            if not self.status & BUSY:
                self.cfg[a - 0x100] = v
        # Control register.
        elif a == R_CTRL:
            # Start one job.
            if v & CMD_START:
                self._start()
            # Run the descriptor queue.
            elif v & CMD_QUEUE and self.queue:
                self._start_queue()
            # Clear status.
            elif v & CMD_CLEAR:
                self.status = 0
        # Memories (writes ignored while busy, Y is read-only from the bus).
        else:
            # Find the window.
            m = self._mem(a)
            # Store.
            if m and m[0] is not self.Y and not self.status & BUSY:
                m[0][m[1]] = v

    # One byte read.
    def read(self, a):
        # The window block is write-only (reads as zero).
        if R_WIN <= a < R_WIN + NWIN:
            return 0
        # Config read-back (and QCNT).
        if 0x100 <= a < 0x100 + NCFG or a == R_QCNT:
            return self.cfg[a - 0x100]
        # Descriptors done.
        if a == R_QDONE:
            return self.qdone if self.queue else 0
        # Status.
        if a == R_STATUS:
            return self.status
        # Cycle counter bytes.
        if R_CYCLES <= a < R_CYCLES + 4:
            return (self.cycle_count >> (8 * (a - R_CYCLES))) & 0xFF
        # Identification block.
        if R_MAGIC <= a < R_MAGIC + 9:
            feat = ((F_RQ_PIPE if self.rq_mode == 1 else 0) | (F_QUEUE if self.queue else 0) | (F_WIN if self.win else 0)
                    | (F_FUSE if self.fuse else 0) | (F_PERF if self.perf else 0))
            return (MAGIC, VERSION, self.aw[0], self.aw[1], self.aw[2], self.aw[3], self.lanes, feat, self.d_aw if self.queue else 0)[a - R_MAGIC]
        # Performance counters.
        if R_PERF <= a < R_PERF + 16:
            return (self.pc[(a - R_PERF) // 4] >> (8 * ((a - R_PERF) % 4))) & 0xFF if self.perf else 0
        # Memories.
        m = self._mem(a)
        # Mapped byte or zero.
        return m[0][m[1]] if m else 0

    # Block write (counts as one link frame per 128 bytes).
    def write_block(self, a, buf):
        # Statistics.
        self.frames += (len(buf) + 127) // 128
        self.bytes += len(buf)
        # Byte by byte, in order (so CTRL as the last byte starts the job).
        for i in range(len(buf)):
            self.write(a + i, buf[i])

    # Block read.
    def read_block(self, a, n):
        # Statistics.
        self.frames += (n + 127) // 128
        self.bytes += n
        # Gather the bytes.
        return bytearray(self.read(a + i) for i in range(n))

    # Run the job described by 22 config bytes: returns (ok, cycles, mac, stall, output bytes). Writes Y.
    def _job(self, cfgb):
        # Decode the configuration block.
        K, N, nvec, xs, wb, pb, ioff, zp, amin, amax, mode, xbase, ybase = struct.unpack_from("<HHHHHHhbbbBHH", cfgb, 0)
        # Window mode and write-back to X are only honoured on builds that have them (the RTL ignores the bits).
        winm = bool(mode & MODE_WIN) and self.win
        outx = bool(mode & MODE_OUTX) and self.fuse
        # Window block.
        wv = struct.unpack_from(WIN_FMT, cfgb, NCFG) if self.win else None
        # Lanes and bank geometry.
        L = self.lanes
        bw, bp = len(self.W) // L, len(self.P) // L
        # Output bytes per result and depthwise flag.
        ob = 4 if mode & MODE_RAW else 1
        dw = bool(mode & MODE_DW)
        # Empty job: error, nothing computed.
        if K == 0 or N == 0 or nvec == 0:
            return False, 0, 0, 0, 0
        # Output memory (X when fused).
        osize = len(self.X) if outx else len(self.Y)
        # Same per-group bounds checks as the RTL, walking the groups in order.
        ys, bad = ybase, False
        for v in range(nvec):
            o = 0
            while o < N and not bad:
                # Group size, local weight row, local parameter record and X slice of this group.
                cnt = 1 if dw else min(L, N - o)
                row = o // L
                xrow = xbase + v * xs + (o * K if dw else 0)
                # Memory limits.
                xbad = (xbase + wv[18] > len(self.X)) if winm else (xrow + K > len(self.X))
                if xbad or wb + row * K + K > bw or (pb + row) * PARAM_BYTES + PARAM_BYTES > bp or ys + cnt * ob > osize:
                    bad = True
                ys += cnt * ob
                o += cnt
            if bad:
                break
        # Error: nothing is reported beyond the status.
        if bad:
            return False, 0, 0, 0, 0
        # Gather the channels' weight rows and parameter records from their banks.
        w, p = bytearray(), bytearray()
        for o in range(N):
            # Bank and local row of channel o.
            b, r = o % L, o // L
            w += self.W[b * bw + wb + r * K: b * bw + wb + (r + 1) * K]
            p += self.P[b * bp + (pb + r) * PARAM_BYTES: b * bp + (pb + r + 1) * PARAM_BYTES]
        # Window mode: build the input vectors the way the generator walks them; padding adds nothing.
        if winm:
            xin, xs = to_i8_list(window_vectors(self.X, xbase, wv, K, N, nvec, dw, (-ioff) & 0xFF)), K * (N if dw else 1)
        else:
            xin = to_i8_list(self.X[xbase:])
        # Compute with the shared kernel (the input vectors start at xbase).
        y = dot_requant(xin, to_i8_list(w), p, K, N, nvec, xs, ioff, zp, amin, amax, dw=dw, raw=bool(mode & MODE_RAW))
        # Store results at ybase (in Y, or in X when fused).
        mem = self.X if outx else self.Y
        if mode & MODE_RAW:
            # Little-endian int32 words.
            mem[ybase:ybase + 4 * len(y)] = struct.pack("<%di" % len(y), *y)
        else:
            # Bytes.
            for i, v in enumerate(y):
                mem[ybase + i] = v & 0xFF
        # Exact clock count and counters.
        shifts = [struct.unpack_from("<b", p, o * PARAM_BYTES + 8)[0] for o in range(N)]
        cyc, mac, stall = job_stats(K, N, nvec, shifts, mode, L, self.rq_mode)
        return True, cyc, mac, stall, nvec * N * ob

    # CTRL = start: one job from the config registers.
    def _start(self):
        # Run it.
        ok, cyc, mac, stall, outb = self._job(bytes(self.cfg[:NCFG]) + bytes(self.wcfg if self.win else b""))
        # Counters restart with every start.
        self.pc = [mac, stall, outb, 1] if ok else [0, 0, 0, 0]
        # Result.
        self.cycle_count = cyc
        self.status = DONE if ok else ERR | DONE

    # CTRL = queue: run QCNT descriptors from D, back to back.
    def _start_queue(self):
        # Count and capacity checks.
        n = self.cfg[R_QCNT - 0x100]
        if n == 0 or n * DESC_BYTES > len(self.D):
            self.status = ERR | DONE
            return
        # Counters restart.
        self.pc = [0, 0, 0, 0]
        total, self.qdone = 0, 0
        # Each descriptor: ncfg + 1 clocks to load, then the job.
        for i in range(n):
            # The descriptor becomes the config registers (and the window block).
            d = self.D[i * DESC_BYTES: i * DESC_BYTES + self.ncfg]
            self.cfg[:NCFG] = d[:NCFG]
            if self.win:
                self.wcfg[:] = d[NCFG:]
            ok, cyc, mac, stall, outb = self._job(bytes(d))
            # Failed job: stop here.
            if not ok:
                self.status = ERR | DONE
                return
            # Accumulate (the descriptor load takes ncfg + 1 clocks).
            total += self.ncfg + 1 + cyc
            self.pc[0] += mac
            self.pc[1] += stall
            self.pc[2] += outb
            self.pc[3] += 1
            self.qdone += 1
        # Finished.
        self.cycle_count = total
        self.status = DONE
