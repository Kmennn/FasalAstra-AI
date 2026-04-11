# PURPOSE: Read wand velocity from MPU6050 IMU via I2C
# WIRING:
#   MPU6050 VCC → RPi4 3.3V (Pin 1)
#   MPU6050 GND → RPi4 GND  (Pin 6)
#   MPU6050 SDA → RPi4 SDA  (Pin 3 / GPIO 2)
#   MPU6050 SCL → RPi4 SCL  (Pin 5 / GPIO 3)
#
# WHY MPU6050:
#   Gives us real wand velocity in m/s
#   Without this, we assume fixed speed = inaccurate
#   With this, Ti adjusts dynamically = surgical precision
#
# VERIFY I2C CONNECTION:
#   Run: sudo i2cdetect -y 1
#   Should show 0x68 in the grid

import smbus2
import time
import math

MPU6050_ADDR   = 0x68
PWR_MGMT_1     = 0x6B
ACCEL_XOUT_H   = 0x3B

class IMUReader:
    def __init__(self):
        try:
            self.bus = smbus2.SMBus(1)  # I2C bus 1
            # Wake up MPU6050 (it starts in sleep mode)
            self.bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)
            time.sleep(0.1)
            print("  ✅ MPU6050 IMU initialized at 0x68")
            self.available = True
        except Exception as e:
            print(f"  ⚠️ IMU not found: {e}")
            print("  ⚠️ Using fallback velocity: 1.2 m/s")
            self.available = False

        self.prev_vel  = 0.0
        self.prev_time = time.time()

    def _read_raw(self, addr):
        high = self.bus.read_byte_data(MPU6050_ADDR, addr)
        low  = self.bus.read_byte_data(MPU6050_ADDR, addr+1)
        val  = (high << 8) | low
        if val >= 0x8000:
            val = val - 65536
        return val

    def get_velocity(self):
        """
        Returns wand velocity in m/s
        Uses accelerometer integration
        Falls back to 1.2 m/s if IMU unavailable
        """
        if not self.available:
            return 1.2   # fallback for testing

        try:
            # Read accelerometer X axis (forward motion)
            accel_x_raw = self._read_raw(ACCEL_XOUT_H)
            accel_x_ms2 = accel_x_raw / 16384.0 * 9.81

            # Integrate acceleration to velocity
            now   = time.time()
            dt    = now - self.prev_time
            delta = accel_x_ms2 * dt

            # Simple low-pass filter
            velocity = self.prev_vel * 0.85 + abs(delta) * 0.15

            # Clamp to realistic walking range
            velocity = max(0.0, min(velocity, 3.0))

            self.prev_vel  = velocity
            self.prev_time = now

            return velocity

        except Exception:
            return 1.2   # fallback

    def get_status(self):
        return "MPU6050 Active" if self.available else "Fallback 1.2m/s"
