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
#   0x11F CTRL     u8    write 0x01 = start, 0x80 = clear status (put it last in the config frame)
#   0x120 STATUS   u8    bit0 busy, bit1 done, bit2 error
#   0x121 CYCLES   u32   clocks the last job took
#   0x130 MAGIC 0x4D, 0x131 VERSION, 0x132 W_AW, 0x133 X_AW, 0x134 Y_AW, 0x135 P_AW  (read-only)
#   0x1000 X   0x2000 Y   0x3000 P   0x8000 W      memory windows (sizes are 2^*_AW bytes)

import struct

from .kernels import PARAM_BYTES, dot_requant, to_i8_list

# Register addresses.
R_CFG, R_CTRL, R_STATUS, R_CYCLES = 0x100, 0x11F, 0x120, 0x121
# Read-only identification block.
R_MAGIC = 0x130
# Memory windows.
X_ADDR, Y_ADDR, P_ADDR, W_ADDR = 0x1000, 0x2000, 0x3000, 0x8000
# Engine magic number.
MAGIC = 0x4D
# Register-map version this driver speaks.
VERSION = 1
# STATUS bits.
BUSY, DONE, ERR = 1, 2, 4
# MODE bits.
MODE_DW, MODE_RAW = 1, 2
# CTRL commands.
CMD_START, CMD_CLEAR = 0x01, 0x80


# Pack the configuration block 0x100..0x11F with CTRL=start as its last byte.
def config_frame(K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode):
    # 0x100..0x111 in one struct, little endian.
    head = struct.pack("<HHHHHHhbbbB", K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode)
    # Pad up to 0x11F, then the start command, so the job starts when the frame's last byte lands.
    return head + bytes(0x1F - len(head)) + bytes([CMD_START])


# Software engine: same interface as HwEngine, runs dot_requant in Python. Used for backend="cpu".
class CpuEngine:
    # Name shown in summaries.
    name = "cpu"

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

    # Store weight bytes at a byte offset.
    def upload_w(self, off, data):
        # Grow the store if needed.
        if len(self.W) < off + len(data):
            self.W += bytes(off + len(data) - len(self.W))
        # Copy.
        self.W[off:off + len(data)] = data

    # Store parameter records starting at a channel index.
    def upload_p(self, ch, data):
        # Byte offset of the channel.
        off = ch * PARAM_BYTES
        # Grow the store if needed.
        if len(self.P) < off + len(data):
            self.P += bytes(off + len(data) - len(self.P))
        # Copy.
        self.P[off:off + len(data)] = data

    # Run one job; x is a bytes-like buffer of int8 data; returns a list of ints.
    def run(self, K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode, x):
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
        # Identification block.
        ident = bus.read_block(R_MAGIC, 6)
        # Wrong or missing bitstream.
        if ident[0] != MAGIC:
            raise RuntimeError("tinyml engine not found (magic 0x%02X, expected 0x%02X): is the bitstream loaded?" % (ident[0], MAGIC))
        # Newer register map than this driver knows.
        if ident[1] != VERSION:
            raise RuntimeError("engine register map v%d, driver speaks v%d" % (ident[1], VERSION))
        # Memory sizes from the address widths.
        self.w_cap, self.x_cap, self.y_cap = 1 << ident[2], 1 << ident[3], 1 << ident[4]
        # Parameter capacity in channels.
        self.p_cap = (1 << ident[5]) // PARAM_BYTES
        # Clock count of the last job.
        self.cycles = 0
        # Clear any stale status.
        bus.write_block(R_CTRL, bytes([CMD_CLEAR]))

    # Write weight bytes into W.
    def upload_w(self, off, data):
        # One (chunked) block write.
        self.bus.write_block(W_ADDR + off, data)

    # Write parameter records into P.
    def upload_p(self, ch, data):
        # One (chunked) block write.
        self.bus.write_block(P_ADDR + ch * PARAM_BYTES, data)

    # Run one job on the FPGA and return the outputs as a list of ints.
    def run(self, K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode, x):
        # Bus shortcut.
        bus = self.bus
        # Input vectors into X.
        bus.write_block(X_ADDR, x)
        # Configuration, ending with the start command.
        bus.write_block(R_CFG, config_frame(K, N, nvec, xstride, wbase, pbase, in_off, out_zp, act_min, act_max, mode))
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
                break
        else:
            # Never finished.
            raise RuntimeError("engine timeout")
        # Raw int32 outputs.
        if mode & MODE_RAW:
            # Four bytes per output.
            y = bus.read_block(Y_ADDR, nvec * N * 4)
            # Little-endian signed words.
            return list(struct.unpack("<%di" % (nvec * N), y))
        # int8 outputs.
        return to_i8_list(bus.read_block(Y_ADDR, nvec * N))


# Python model of the RTL behind the byte bus: lets the hardware driver run on a PC with no FPGA.
# It is register-accurate (same map, same STATUS behaviour) and is what the RTL testbench is checked against.
class SimBus:
    # Default sizes match the tmf_engine.v defaults.
    def __init__(self, w_aw=12, x_aw=10, y_aw=8, p_aw=10):
        # Address widths.
        self.aw = (w_aw, x_aw, y_aw, p_aw)
        # Memories.
        self.W, self.X, self.Y, self.P = bytearray(1 << w_aw), bytearray(1 << x_aw), bytearray(1 << y_aw), bytearray(1 << p_aw)
        # Config registers 0x100..0x11F.
        self.cfg = bytearray(0x20)
        # STATUS and CYCLES.
        self.status, self.cycle_count = 0, 0
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
        # W window.
        if W_ADDR <= a < W_ADDR + len(self.W):
            return self.W, a - W_ADDR
        # Unmapped.
        return None

    # One byte write, exactly as the RTL decodes it.
    def write(self, a, v):
        # Config registers (ignored while busy, like the RTL).
        if 0x100 <= a < 0x11F:
            # Store.
            if not self.status & BUSY:
                self.cfg[a - 0x100] = v
        # Control register.
        elif a == R_CTRL:
            # Start.
            if v & CMD_START:
                self._start()
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
        # Config read-back.
        if 0x100 <= a < 0x11F:
            return self.cfg[a - 0x100]
        # Status.
        if a == R_STATUS:
            return self.status
        # Cycle counter bytes.
        if R_CYCLES <= a < R_CYCLES + 4:
            return (self.cycle_count >> (8 * (a - R_CYCLES))) & 0xFF
        # Identification block.
        if R_MAGIC <= a < R_MAGIC + 6:
            return (MAGIC, VERSION) [a - R_MAGIC] if a < R_MAGIC + 2 else self.aw[a - R_MAGIC - 2]
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

    # Execute a job the way tmf_engine.v does, including its error checks.
    def _start(self):
        # A start while busy is an error (cannot happen here because jobs finish instantly).
        K, N, nvec, xs, wb, pb, ioff, zp, amin, amax, mode = struct.unpack_from("<HHHHHHhbbbB", self.cfg, 0)
        # Output bytes per result.
        ob = 4 if mode & MODE_RAW else 1
        # Highest X byte the job touches.
        xmax = (nvec - 1) * xs + (N * K if mode & MODE_DW else K)
        # Same checks as the RTL: empty job or any memory overrun.
        if (K == 0 or N == 0 or nvec == 0 or xmax > len(self.X) or wb + N * K > len(self.W)
                or (pb + N) * PARAM_BYTES > len(self.P) or nvec * N * ob > len(self.Y)):
            # Error and done, nothing computed.
            self.status = ERR | DONE
            return
        # Compute with the shared kernel.
        y = dot_requant(to_i8_list(self.X), to_i8_list(self.W[wb:wb + N * K]),
                        self.P[pb * PARAM_BYTES:(pb + N) * PARAM_BYTES], K, N, nvec, xs, ioff, zp, amin, amax,
                        dw=bool(mode & MODE_DW), raw=bool(mode & MODE_RAW))
        # Store results.
        if mode & MODE_RAW:
            # Little-endian int32 words.
            self.Y[0:4 * len(y)] = struct.pack("<%di" % len(y), *y)
        else:
            # Bytes.
            for i, v in enumerate(y):
                self.Y[i] = v & 0xFF
        # Exact clock count of tmf_engine.v: per output 1 check + 10 param + 1 init + K MACs + 3 drain + 1 next,
        # plus 4 raw byte writes, or the requantiser: 38 + ts/8 + ts%8 clocks with ts = 31 - shift.
        cyc = 0
        for o in range(N):
            # Shift of this channel.
            sh = struct.unpack_from("<b", self.P, (pb + o) * PARAM_BYTES + 8)[0]
            # Requantiser shift distance.
            ts = 31 - sh
            # Clocks for this channel's output.
            cyc += K + (20 if mode & MODE_RAW else 54 + ts // 8 + ts % 8)
        # Same for every vector.
        self.cycle_count = nvec * cyc
        # Done.
        self.status = DONE
