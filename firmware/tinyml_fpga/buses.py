# buses.py - byte buses that carry the engine's register map to real hardware.
#   UartBus : on a PC, through a USB-UART to rtl/uart_bus.v (rtl/tinyml_uart_top.v) on any FPGA board.
#   McuUartBus : on a MicroPython MCU, through a hardware UART wired to the FPGA.
#   SpiBus / McuSpiBus : Linux (spidev) or MicroPython SPI master to rtl/spi_bus.v (rtl/tinyml_spi_top.v).
#   I2cBus / McuI2cBus : Linux or MicroPython I2C master to rtl/i2c_bus.v (rtl/tinyml_i2c_top.v).
#   AxiLiteBus : a CPU that sees the engine through AXI4-Lite (rtl/axil_bus.v), e.g. a Zynq running Linux.
# Any other link works the same way: write a class with write_block(addr, buf) and read_block(addr, n)
# and pass HwEngine(YourBus()) as the backend (see docs/adding_a_board.md).

# Largest burst one frame can carry (the CMD byte holds length-1 in 7 bits).
MAX_FRAME = 128


# UART bridge bus (PC side, pyserial). Frame format:
#   write: [0x00 | n-1] [addr_hi] [addr_lo] [n data bytes]  -> FPGA answers one 0x55 byte when committed
#   read : [0x80 | n-1] [addr_hi] [addr_lo]                 -> FPGA answers n data bytes
class UartBus:
    # Open the serial port.
    def __init__(self, port, baud=115200, timeout=2.0):
        # pyserial is only needed on the PC.
        import serial
        # Port handle.
        self.s = serial.Serial(port, baud, timeout=timeout)
        # Drop anything left over from a previous run.
        self.s.reset_input_buffer()

    # Write a buffer starting at addr.
    def write_block(self, addr, buf):
        # Offset into the buffer.
        i = 0
        # Frame by frame.
        while i < len(buf):
            # This frame's length.
            n = min(MAX_FRAME, len(buf) - i)
            # Address of this frame.
            a = addr + i
            # Header and data.
            self.s.write(bytes([n - 1, a >> 8, a & 0xFF]) + bytes(buf[i:i + n]))
            # Wait for the commit acknowledge.
            ack = self.s.read(1)
            # Missing or wrong acknowledge.
            if ack != b"\x55":
                raise RuntimeError("UART bridge: no write ack at 0x%04X (got %r)" % (a, ack))
            # Next frame.
            i += n

    # Read n bytes starting at addr.
    def read_block(self, addr, n):
        # Result.
        out = bytearray()
        # Frame by frame.
        while len(out) < n:
            # This frame's length.
            m = min(MAX_FRAME, n - len(out))
            # Address of this frame.
            a = addr + len(out)
            # Read header.
            self.s.write(bytes([0x80 | (m - 1), a >> 8, a & 0xFF]))
            # Data back.
            part = self.s.read(m)
            # Short read means the bridge did not answer.
            if len(part) != m:
                raise RuntimeError("UART bridge: read of %d bytes at 0x%04X returned %d" % (m, a, len(part)))
            # Store it.
            out += part
        # All bytes.
        return out


# UART bridge bus for a MicroPython microcontroller (same frames as UartBus, no pyserial).
# Example (UART1 of an RP2040 / RP2350 style board, TX on GPIO8 and RX on GPIO9, wired to the FPGA):
#   bus = McuUartBus(baud=921600, tx=8, rx=9)
#   m = tinyml_fpga.load("model.tmf", backend=tinyml_fpga.HwEngine(bus))
class McuUartBus:
    # Open the hardware UART.
    def __init__(self, baud=921600, tx=8, rx=9, uart_id=1, timeout_ms=2000):
        # MicroPython only.
        from machine import UART, Pin
        # Keep the time helpers.
        import time
        # Time module handle.
        self.time = time
        # Long receive buffer so a 128 byte reply never overflows.
        self.u = UART(uart_id, baudrate=baud, tx=Pin(tx), rx=Pin(rx), rxbuf=1024)
        # Longest wait for a reply.
        self.timeout_ms = timeout_ms
        # Drop anything left over.
        self.u.read()

    # Read exactly n bytes, or raise on timeout.
    def _read_exact(self, n, what):
        # Result so far.
        out = bytearray()
        # Deadline for the whole reply.
        t0 = self.time.ticks_ms()
        # Until all bytes are in.
        while len(out) < n:
            # Whatever has arrived (None when nothing).
            part = self.u.read(n - len(out))
            # Keep it.
            if part:
                out += part
            # Give up when the FPGA stays silent.
            elif self.time.ticks_diff(self.time.ticks_ms(), t0) > self.timeout_ms:
                raise RuntimeError("UART bridge: %s returned %d of %d bytes" % (what, len(out), n))
        # All bytes.
        return out

    # Write a buffer starting at addr.
    def write_block(self, addr, buf):
        # Offset into the buffer.
        i = 0
        # Frame by frame.
        while i < len(buf):
            # This frame's length.
            n = min(MAX_FRAME, len(buf) - i)
            # Address of this frame.
            a = addr + i
            # Header and data in one write.
            self.u.write(bytes([n - 1, a >> 8, a & 0xFF]) + bytes(buf[i:i + n]))
            # Commit acknowledge.
            ack = self._read_exact(1, "write ack at 0x%04X" % a)
            # Wrong acknowledge.
            if ack != b"\x55":
                raise RuntimeError("UART bridge: bad write ack at 0x%04X (got %r)" % (a, bytes(ack)))
            # Next frame.
            i += n

    # Read n bytes starting at addr.
    def read_block(self, addr, n):
        # Result.
        out = bytearray()
        # Frame by frame.
        while len(out) < n:
            # This frame's length.
            m = min(MAX_FRAME, n - len(out))
            # Address of this frame.
            a = addr + len(out)
            # Read header.
            self.u.write(bytes([0x80 | (m - 1), a >> 8, a & 0xFF]))
            # Data back.
            out += self._read_exact(m, "read at 0x%04X" % a)
        # All bytes.
        return out


# SPI frames (rtl/spi_bus.v), chip select low for the whole frame:
#   write: [n-1] [addr_hi] [addr_lo] [n data bytes]
#   read : [0x80 | n-1] [addr_hi] [addr_lo] [dummy] then n clocked-out bytes (the reply is bytes 4 .. 4+n-1)
# Shared by the PC and MicroPython classes; subclasses provide _xfer(tx) -> rx (full duplex, CS handled inside).
class _SpiFrames:
    # Write a buffer starting at addr.
    def write_block(self, addr, buf):
        # Offset into the buffer.
        i = 0
        # Frame by frame.
        while i < len(buf):
            # This frame's length.
            n = min(MAX_FRAME, len(buf) - i)
            # Address of this frame.
            a = addr + i
            # One chip-select frame.
            self._xfer(bytes([n - 1, a >> 8, a & 0xFF]) + bytes(buf[i:i + n]))
            # Next frame.
            i += n

    # Read n bytes starting at addr.
    def read_block(self, addr, n):
        # Result.
        out = bytearray()
        # Frame by frame.
        while len(out) < n:
            # This frame's length.
            m = min(MAX_FRAME, n - len(out))
            # Address of this frame.
            a = addr + len(out)
            # Header, one dummy byte, then m clocks that carry the reply.
            rx = self._xfer(bytes([0x80 | (m - 1), a >> 8, a & 0xFF, 0]) + bytes(m))
            # The reply starts after the four header bytes.
            out += bytes(rx[4:4 + m])
        # All bytes.
        return out


# SPI bridge bus on Linux (python3-spidev): SpiBus(bus=0, device=0, speed_hz=2000000). Keep the SPI clock at most
# the FPGA clock / 6.
class SpiBus(_SpiFrames):
    # Open /dev/spidevBUS.DEVICE in mode 0 (or use a ready spidev-like object).
    def __init__(self, bus=0, device=0, speed_hz=2000000, dev=None):
        # spidev only exists on Linux boards.
        if dev is None:
            import spidev
            dev = spidev.SpiDev()
            dev.open(bus, device)
            dev.mode = 0
            dev.max_speed_hz = speed_hz
        # Device handle.
        self.dev = dev

    # One full-duplex transfer with chip select low throughout.
    def _xfer(self, tx):
        # xfer2 keeps the chip select asserted for the whole list.
        return bytes(self.dev.xfer2(list(tx)))


# SPI bridge bus for MicroPython: spi is a machine.SPI (mode 0, polarity 0, phase 0), cs a machine.Pin (output).
#   spi = SPI(0, baudrate=2000000, polarity=0, phase=0, sck=Pin(18), mosi=Pin(19), miso=Pin(16))
#   bus = McuSpiBus(spi, Pin(17, Pin.OUT, value=1))
class McuSpiBus(_SpiFrames):
    # Keep the handles.
    def __init__(self, spi, cs):
        self.spi = spi
        self.cs = cs
        # Chip select idles high.
        self.cs.value(1)

    # One full-duplex transfer.
    def _xfer(self, tx):
        # Reply buffer of the same length.
        rx = bytearray(len(tx))
        # Select, clock, release.
        self.cs.value(0)
        try:
            self.spi.write_readinto(tx, rx)
        finally:
            self.cs.value(1)
        return rx


# I2C frames (rtl/i2c_bus.v), 7-bit slave address:
#   write: START addr+W [addr_hi] [addr_lo] [data...] STOP
#   read : START addr+W [addr_hi] [addr_lo] RESTART addr+R [data...] STOP
# Subclasses provide _write(data, stop) and _read(n).
class _I2cFrames:
    # Write a buffer starting at addr.
    def write_block(self, addr, buf):
        # Offset into the buffer.
        i = 0
        # Frame by frame.
        while i < len(buf):
            # This frame's length.
            n = min(MAX_FRAME, len(buf) - i)
            # Address of this frame.
            a = addr + i
            # Pointer and data in one transaction.
            self._write(bytes([a >> 8, a & 0xFF]) + bytes(buf[i:i + n]), True)
            # Next frame.
            i += n

    # Read n bytes starting at addr.
    def read_block(self, addr, n):
        # Result.
        out = bytearray()
        # Frame by frame.
        while len(out) < n:
            # This frame's length.
            m = min(MAX_FRAME, n - len(out))
            # Address of this frame.
            a = addr + len(out)
            # Set the pointer, repeated start, read.
            self._write(bytes([a >> 8, a & 0xFF]), False)
            out += self._read(m)
        # All bytes.
        return out


# I2C bridge bus on Linux. dev needs write(addr, data) and write_read(addr, data, n); open_i2c() builds one from smbus2.
class I2cBus(_I2cFrames):
    # Keep the device and the 7-bit slave address.
    def __init__(self, dev, addr=0x2A):
        self.dev = dev
        self.addr = addr

    # Transaction that writes data; stop=False is held open for the read that follows.
    def _write(self, data, stop):
        if stop:
            self.dev.write(self.addr, data)
        else:
            # Remember the pointer; the next _read sends it as the first half of a combined transaction.
            self.ptr = data

    # Combined write-pointer / read transaction.
    def _read(self, n):
        return bytearray(self.dev.write_read(self.addr, self.ptr, n))


# Build the I2cBus device object from smbus2 (Linux /dev/i2c-N).
def open_i2c(bus=1, addr=0x2A):
    # smbus2 only exists on Linux boards.
    from smbus2 import SMBus, i2c_msg

    # Small adapter with the two calls I2cBus needs.
    class _Dev:
        def __init__(self):
            self.b = SMBus(bus)

        def write(self, a, data):
            self.b.i2c_rdwr(i2c_msg.write(a, list(data)))

        def write_read(self, a, data, n):
            r = i2c_msg.read(a, n)
            self.b.i2c_rdwr(i2c_msg.write(a, list(data)), r)
            return bytes(list(r))

    return I2cBus(_Dev(), addr)


# I2C bridge bus for MicroPython: i2c is a machine.I2C (e.g. I2C(0, scl=Pin(5), sda=Pin(4), freq=400000)).
class McuI2cBus(_I2cFrames):
    # Keep the handles.
    def __init__(self, i2c, addr=0x2A):
        self.i2c = i2c
        self.addr = addr

    # Write data; stop=False leaves the bus claimed for the repeated start of the read.
    def _write(self, data, stop):
        self.i2c.writeto(self.addr, data, stop)

    # Read n bytes (the master NACKs the last one).
    def _read(self, n):
        return bytearray(self.i2c.readfrom(self.addr, n))


# AXI4-Lite bus: mem is any object that supports byte slices, such as mmap.mmap(fd, size, offset=BASE) on /dev/mem,
# or a UIO device. The engine's byte address A is at mem[A]; the bridge turns narrow accesses into byte strobes.
class AxiLiteBus:
    # Keep the window and an optional offset into it.
    def __init__(self, mem, base=0):
        self.mem = mem
        self.base = base

    # Write a buffer starting at addr.
    def write_block(self, addr, buf):
        a = self.base + addr
        self.mem[a:a + len(buf)] = bytes(buf)

    # Read n bytes starting at addr.
    def read_block(self, addr, n):
        a = self.base + addr
        return bytearray(self.mem[a:a + n])


# Map the engine's AXI window from physical address phys (Linux, needs root): returns an AxiLiteBus.
def open_axi(phys, size=0x10000, path="/dev/mem"):
    import mmap
    import os
    fd = os.open(path, os.O_RDWR | os.O_SYNC)
    return AxiLiteBus(mmap.mmap(fd, size, offset=phys))
