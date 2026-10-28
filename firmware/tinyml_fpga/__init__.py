# tinyml_fpga (project: tinyml-fpga) - run int8 TensorFlow Lite models with their conv / depthwise /
# dense layers on an FPGA.
#
# Same package on both sides:
#   PC (CPython)        : convert .tflite -> .tmf, check it against LiteRT, run it in software, on the
#                         register model, or on any FPGA through the UART bridge.
#   MCU (MicroPython)   : copy the runtime files to the microcontroller and drive the FPGA over any link
#                         by passing HwEngine(your_bus) as the backend.
#
#   import tinyml_fpga
#   m = tinyml_fpga.load("model.tmf", backend="uart:COM5")    # "cpu", "sim", "uart:PORT", "spi:0.0", "i2c:1", "axi:0x40000000", HwEngine(bus)
#   probs = m.predict(sample)
#
# Nothing here imports numpy or the tflite schema at import time, so it loads on MicroPython.

# Package version.
__version__ = "0.9.0"

from .model import Model
from .engine import CpuEngine, HwEngine, SimBus


# Open a .tmf model on the chosen backend and return a ready Model.
def load(path, backend="cpu", **kw):
    # Software reference engine.
    if backend == "cpu":
        engine = CpuEngine()
    # Register-accurate model of the FPGA engine (PC, no hardware); kw: w_aw, x_aw, y_aw, p_aw.
    elif backend == "sim":
        engine = HwEngine(SimBus(**kw))
    # Any FPGA with rtl/uart_bus.v, from a PC: "uart:/dev/ttyUSB1" or "uart:COM5"; kw: baud.
    elif isinstance(backend, str) and backend.startswith("uart:"):
        from .buses import UartBus
        engine = HwEngine(UartBus(backend[5:], **kw))
    # SPI master (Linux spidev) to rtl/spi_bus.v: "spi:BUS.DEVICE", e.g. "spi:0.0"; kw: speed_hz.
    elif isinstance(backend, str) and backend.startswith("spi:"):
        from .buses import SpiBus
        b, _, d = backend[4:].partition(".")
        engine = HwEngine(SpiBus(int(b), int(d or 0), **kw))
    # I2C master (Linux smbus2) to rtl/i2c_bus.v: "i2c:BUS" or "i2c:BUS:ADDR", e.g. "i2c:1:0x2A".
    elif isinstance(backend, str) and backend.startswith("i2c:"):
        from .buses import open_i2c
        b, _, a = backend[4:].partition(":")
        engine = HwEngine(open_i2c(int(b), int(a, 0) if a else 0x2A))
    # CPU with the engine on AXI4-Lite (Linux /dev/mem): "axi:0x40000000" (physical base address).
    elif isinstance(backend, str) and backend.startswith("axi:"):
        from .buses import open_axi
        engine = HwEngine(open_axi(int(backend[4:], 0), **kw))
    # A ready engine object (anything with upload_w / upload_p / run and the *_cap fields).
    elif hasattr(backend, "run"):
        engine = backend
    # Unknown name.
    else:
        raise ValueError("unknown backend %r (use 'cpu', 'sim', 'uart:PORT', 'spi:B.D', 'i2c:BUS', 'axi:BASE' or an HwEngine)" % (backend,))
    # Build the model on it.
    return Model(path, engine)


# Convert a .tflite file to .tmf (PC only).
def convert(src, dst):
    # Heavy imports only when converting.
    from .converter import convert_file
    # Returns the header for inspection.
    return convert_file(src, dst)
