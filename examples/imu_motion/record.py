# record.py - MicroPython board: record labelled BMI323 data at 100 Hz into a CSV on the board's flash.
#   mpremote exec "import record; record.run(label=1, seconds=20)"     # 0 idle, 1 shake, 2 circle
#   mpremote cp :motion_1.csv data/                                      # then train.py data/*.csv
# Set the I2C pins below to where your BMI323 is wired.
import time
from machine import I2C, Pin

from bmi323 import BMI323

# I2C bus wiring (change to your board's pins).
I2C_ID, SDA, SCL, ADDR = 0, 4, 5, 0x68


# Record `seconds` of samples with one label into motion_<label>.csv (appends).
def run(label, seconds=20, rate_hz=100):
    # Sensor.
    imu = BMI323(I2C(I2C_ID, sda=Pin(SDA), scl=Pin(SCL), freq=400000), addr=ADDR)
    # Sample period.
    period = 1000000 // rate_hz
    # Output file.
    with open("motion_%d.csv" % label, "a") as f:
        # Next sample time.
        t = time.ticks_us()
        # Fixed number of samples.
        for _ in range(seconds * rate_hz):
            # One row: label, accel xyz (g), gyro xyz (dps).
            f.write("%d,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\n" % ((label,) + imu.read()))
            # Keep the rate.
            t = time.ticks_add(t, period)
            while time.ticks_diff(t, time.ticks_us()) > 0:
                pass
    # Done.
    print("recorded %d s of label %d" % (seconds, label))
