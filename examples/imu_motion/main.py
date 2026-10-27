# main.py - MicroPython board: live motion classification from a BMI323 with tinyml_fpga.
# Copy: python -m tinyml_fpga deploy examples/imu_motion/motion.tmf ; mpremote cp bmi323.py main.py :
# Then: mpremote exec "import main"
# Set the wiring below (I2C pins and address, FPGA UART pins). USE_FPGA = False runs the model on the
# microcontroller; True runs the conv / dense layers on an FPGA with rtl/tinyml_uart_top.v wired to FPGA_TX / FPGA_RX
# (any other link: pass tinyml_fpga.HwEngine(YourBus()), see docs/adding_a_board.md).
import time
from machine import I2C, Pin

import tinyml_fpga
from bmi323 import BMI323

# I2C bus wiring (same as record.py).
I2C_ID, SDA, SCL, ADDR = 0, 4, 5, 0x68
# Window length and the scaling used in train.py (accel / 4 g, gyro / 500 dps).
WIN, SCALE = 128, (4, 4, 4, 500, 500, 500)
# Class names in label order.
CLASSES = ("idle", "shake", "circle")
# Classify every this many new samples (0.5 s at 100 Hz).
HOP = 50

# Where the model runs (see the header).
USE_FPGA = False
# UART pins on this board that are wired to the FPGA's uart_rx / uart_tx (used when USE_FPGA is True).
FPGA_TX, FPGA_RX = 8, 9
# True: conv / dense layers run on the FPGA (rtl/tinyml_uart_top.v); False: everything runs on this microcontroller.
if USE_FPGA:
    from tinyml_fpga.buses import McuUartBus
    BACKEND = tinyml_fpga.HwEngine(McuUartBus(921600, FPGA_TX, FPGA_RX))
else:
    BACKEND = "cpu"

# Sensor and model.
imu = BMI323(I2C(I2C_ID, sda=Pin(SDA), scl=Pin(SCL), freq=400000), addr=ADDR)
model = tinyml_fpga.load("motion.tmf", backend=BACKEND)
model.summary()

# Sliding window of scaled samples (flat, 6 values per sample).
window = []
# Samples since the last classification.
new = 0
# Next sample time.
t = time.ticks_us()
# Forever.
while True:
    # One sample, scaled like the training data.
    s = imu.read()
    window.extend([s[i] / SCALE[i] for i in range(6)])
    # Keep the last WIN samples.
    if len(window) > WIN * 6:
        del window[:6]
    new += 1
    # Classify every HOP samples once the window is full.
    if len(window) == WIN * 6 and new >= HOP:
        new = 0
        # Timed inference.
        t0 = time.ticks_us()
        idx, p = model.classify(window)
        print("%-6s p=%.2f  (%d us)" % (CLASSES[idx], p, time.ticks_diff(time.ticks_us(), t0)))
        # Inference took longer than one sample period: restart the clock instead of reading a burst of repeats.
        t = time.ticks_us()
    # 100 Hz.
    t = time.ticks_add(t, 10000)
    while time.ticks_diff(t, time.ticks_us()) > 0:
        pass
