# test_buses.py - the SPI, I2C and AXI4-Lite bus classes against Python models of the FPGA bridges
# (the Verilog bridges themselves are checked by rtl/sim/tb_bus.v with the same frame rules).
import random

import pytest

from tinyml_fpga import Model, CpuEngine, HwEngine, SimBus
from tinyml_fpga.buses import SpiBus, McuSpiBus, I2cBus, McuI2cBus, AxiLiteBus, MAX_FRAME
from conftest import model_path
from tinyml_fpga.converter import convert_bytes


# Model of rtl/spi_bus.v on top of a SimBus: consumes one chip-select frame, returns the full-duplex reply.
class FakeSpiSlave:
    def __init__(self, sim):
        self.sim = sim
        self.frames = 0

    def xfer(self, tx):
        tx = bytes(tx)
        self.frames += 1
        rx = bytearray(len(tx))
        cmd, ah, al = tx[0], tx[1], tx[2]
        a = (ah << 8) | al
        if cmd & 0x80:
            # Read: byte 3 is the dummy, the data come out during bytes 4 ..
            n = len(tx) - 4
            assert n >= 1 and n <= (cmd & 0x7F) + 1 <= MAX_FRAME
            rx[4:] = self.sim.read_block(a, n)
        else:
            assert len(tx) - 3 == (cmd & 0x7F) + 1
            self.sim.write_block(a, tx[3:])
        return rx


# Pin stand-in that records the chip-select history.
class FakePin:
    def __init__(self):
        self.v = None
        self.hist = []

    def value(self, v):
        self.v = v
        self.hist.append(v)


# spidev-like object.
class FakeSpiDev:
    def __init__(self, slave):
        self.slave = slave

    def xfer2(self, lst):
        return list(self.slave.xfer(bytes(lst)))


# machine.SPI-like object.
class FakeMachineSpi:
    def __init__(self, slave, cs):
        self.slave, self.cs = slave, cs

    def write_readinto(self, tx, rx):
        # The chip select must be low during the transfer.
        assert self.cs.v == 0
        rx[:] = self.slave.xfer(tx)


# Model of rtl/i2c_bus.v: pointer write, optional data, repeated start read.
class FakeI2cSlave:
    def __init__(self, sim, addr=0x2A):
        self.sim, self.addr = sim, addr
        self.ptr = 0
        self.log = []

    def write(self, a, data, stop=True):
        assert a == self.addr
        data = bytes(data)
        self.ptr = (data[0] << 8) | data[1]
        if len(data) > 2:
            self.sim.write_block(self.ptr, data[2:])
            self.ptr += len(data) - 2
        self.log.append(("w", len(data)))

    def write_read(self, a, data, n):
        self.write(a, data)
        out = self.sim.read_block(self.ptr, n)
        self.ptr += n
        self.log.append(("r", n))
        return bytes(out)

    # machine.I2C style.
    def writeto(self, a, data, stop=True):
        self.write(a, data, stop)
        self._pending = not stop

    def readfrom(self, a, n):
        assert self._pending
        self._pending = False
        out = self.sim.read_block(self.ptr, n)
        self.ptr += n
        self.log.append(("r", n))
        return bytes(out)


# Model of rtl/axil_bus.v seen through byte slices: a bytearray-backed window onto the SimBus.
class FakeAxiMem:
    def __init__(self, sim):
        self.sim = sim

    def __setitem__(self, sl, data):
        self.sim.write_block(sl.start, bytes(data))

    def __getitem__(self, sl):
        return bytes(self.sim.read_block(sl.start, sl.stop - sl.start))


# Converted test models.
@pytest.fixture(scope="module")
def tmfs():
    pytest.importorskip("tflite")
    out = {}
    for n in ("m_fc", "m_cnn"):
        with open(model_path(n), "rb") as f:
            out[n] = convert_bytes(f.read())
    return out


# Random int8 inputs for a model.
def _inputs(m, n, seed=0):
    size = 1
    for d in m.input["shape"]:
        size *= d
    rng = random.Random(seed)
    return [bytearray(rng.randrange(256) for _ in range(size)) for _ in range(n)]


# Build the bus under test around a fresh SimBus.
def make_bus(kind, sim):
    if kind == "spi":
        return SpiBus(dev=FakeSpiDev(FakeSpiSlave(sim)))
    if kind == "mcu_spi":
        cs = FakePin()
        return McuSpiBus(FakeMachineSpi(FakeSpiSlave(sim), cs), cs)
    if kind == "i2c":
        return I2cBus(FakeI2cSlave(sim))
    if kind == "mcu_i2c":
        return McuI2cBus(FakeI2cSlave(sim))
    return AxiLiteBus(FakeAxiMem(sim), base=0)


KINDS = ["spi", "mcu_spi", "i2c", "mcu_i2c", "axi"]


# Raw writes and reads, including lengths that need several frames.
@pytest.mark.parametrize("kind", KINDS)
def test_block_roundtrip(kind):
    sim = SimBus()
    bus = make_bus(kind, sim)
    data = bytes((i * 37 + 5) & 0xFF for i in range(MAX_FRAME * 2 + 17))
    bus.write_block(0x8000, data)
    assert bytes(bus.read_block(0x8000, len(data))) == data
    assert bytes(bus.read_block(0x8000 + 3, 9)) == data[3:12]
    # The engine really received it.
    assert bytes(sim.read_block(0x8000, 16)) == data[:16]


# A whole model on the engine through every bus gives the software reference's answers.
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", ["m_fc", "m_cnn"])
def test_model_over_bus(tmfs, kind, name):
    sim = SimBus(lanes=2, queue=True, win=True, fuse=True)
    ref = Model(tmfs[name], CpuEngine())
    m = Model(tmfs[name], HwEngine(make_bus(kind, sim)))
    for x in _inputs(ref, 3):
        assert m.invoke(x) == ref.invoke(x)


# SPI frames keep the chip select low for exactly one transfer each.
def test_spi_chip_select_per_frame():
    sim = SimBus()
    cs = FakePin()
    slave = FakeSpiSlave(sim)
    bus = McuSpiBus(FakeMachineSpi(slave, cs), cs)
    bus.write_block(0x1000, bytes(10))
    bus.read_block(0x1000, 10)
    assert cs.hist == [1, 0, 1, 0, 1]
    assert slave.frames == 2


# The I2C read goes out as one combined write-pointer / read transaction.
def test_i2c_uses_repeated_start_for_reads():
    sim = SimBus()
    dev = FakeI2cSlave(sim)
    bus = McuI2cBus(dev)
    bus.write_block(0x1000, bytes(range(8)))
    assert bytes(bus.read_block(0x1000, 8)) == bytes(range(8))
    assert dev.log == [("w", 10), ("w", 2), ("r", 8)]


# Backend strings are routed to the right bus class.
def test_backend_strings_reject_unknown(tmfs, tmp_path):
    import tinyml_fpga
    p = tmp_path / "m.tmf"
    p.write_bytes(tmfs["m_fc"])
    with pytest.raises(ValueError):
        tinyml_fpga.load(str(p), backend="nope:1")
