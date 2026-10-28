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
#   m = tinyml_fpga.load("model.tmf", backend="uart:COM5")    # "cpu", "sim", "uart:PORT", HwEngine(bus)
#   probs = m.predict(sample)
#
# Nothing here imports numpy or the tflite schema at import time, so it loads on MicroPython.

# Package version.
__version__ = "0.7.0"

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
    # A ready engine object (anything with upload_w / upload_p / run and the *_cap fields).
    elif hasattr(backend, "run"):
        engine = backend
    # Unknown name.
    else:
        raise ValueError("unknown backend %r (use 'cpu', 'sim', 'uart:PORT' or an HwEngine)" % (backend,))
    # Build the model on it.
    return Model(path, engine)


# Convert a .tflite file to .tmf (PC only).
def convert(src, dst):
    # Heavy imports only when converting.
    from .converter import convert_file
    # Returns the header for inspection.
    return convert_file(src, dst)
