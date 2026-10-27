# buses.py - byte buses that carry the engine's register map to real hardware.
#   UartBus : on a PC, through a USB-UART to rtl/uart_bus.v (rtl/tinyml_uart_top.v) on any FPGA board.
#   McuUartBus : on a MicroPython MCU, through a hardware UART wired to the FPGA.
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
