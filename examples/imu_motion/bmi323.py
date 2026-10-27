# bmi323.py - minimal MicroPython I2C driver for the Bosch BMI323 (accel + gyro), 100 Hz, +-8 g, +-2000 dps.
# Written from the BMI323 datasheet register map; NOT yet tested on hardware. If you already have a
# working BMI323 driver, use it instead: main.py and record.py only need read() -> (ax, ay, az, gx, gy, gz)
# in g and deg/s.
#   BMI323 registers are 16 bit, little endian. An I2C read returns 2 dummy bytes before the data.
import struct
import time

# 7-bit I2C address (SDO low = 0x68, SDO high = 0x69).
ADDR = 0x68
# Registers.
CHIP_ID, ACC_DATA_X, ACC_CONF, GYR_CONF, CMD = 0x00, 0x03, 0x20, 0x21, 0x7E
# Normal mode, 100 Hz, +-8 g.
ACC_CONF_VAL = 0x4028
# Normal mode, 100 Hz, +-2000 dps.
GYR_CONF_VAL = 0x4048
# LSB per g at +-8 g, LSB per dps at +-2000 dps.
ACC_LSB, GYR_LSB = 4096.0, 16.384


# BMI323 on an I2C bus.
class BMI323:
    # Reset and configure; i2c is a machine.I2C (or SoftI2C) object.
    def __init__(self, i2c, addr=ADDR):
        # Bus and address.
        self.i2c, self.addr = i2c, addr
        # Soft reset.
        self._w(CMD, 0xDEAF)
        time.sleep_ms(5)
        # Chip ID low byte must be 0x43.
        cid = self._r(CHIP_ID, 1)[0] & 0xFF
        if cid != 0x43:
            raise RuntimeError("BMI323 not found (chip id 0x%02X at 0x%02X)" % (cid, addr))
        # Accelerometer and gyroscope on.
        self._w(ACC_CONF, ACC_CONF_VAL)
        self._w(GYR_CONF, GYR_CONF_VAL)
        time.sleep_ms(50)

    # Write one 16-bit register.
    def _w(self, reg, val):
        # Register address, then LSB, MSB.
        self.i2c.writeto_mem(self.addr, reg, struct.pack("<H", val))

    # Read n 16-bit registers (signed), skipping the 2 dummy bytes.
    def _r(self, reg, n):
        # 2 dummy bytes + 2n data bytes.
        b = self.i2c.readfrom_mem(self.addr, reg, 2 + 2 * n)
        # Signed words.
        return struct.unpack("<%dh" % n, b[2:])

    # One sample: (ax, ay, az) in g and (gx, gy, gz) in deg/s.
    def read(self):
        # ACC_DATA_X..Z then GYR_DATA_X..Z are consecutive registers.
        r = self._r(ACC_DATA_X, 6)
        # Scale to physical units.
        return (r[0] / ACC_LSB, r[1] / ACC_LSB, r[2] / ACC_LSB, r[3] / GYR_LSB, r[4] / GYR_LSB, r[5] / GYR_LSB)
