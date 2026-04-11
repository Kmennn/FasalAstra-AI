# PURPOSE: Control 12V Solenoid via RPi4 GPIO
# WIRING:
#   RPi4 GPIO 17 (Pin 11) → Relay IN
#   Relay COM              → 12V Battery +
#   Relay NO               → Solenoid +
#   Solenoid -             → 12V Battery -
#   RPi4 GND (Pin 6)       → Relay GND
#   RPi4 3.3V (Pin 1)      → Relay VCC
#
# WHY RELAY not direct GPIO:
#   RPi GPIO = 3.3V 16mA max
#   Solenoid = 12V 500mA
#   Direct connection = burns RPi instantly
#   Relay = electrically isolated = safe

import RPi.GPIO as GPIO
import time

SOLENOID_PIN    = 17     # GPIO 17 = Pin 11
BURST_DURATION  = 0.050  # 50ms spray burst

class SolenoidController:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(SOLENOID_PIN, GPIO.OUT)
        GPIO.output(SOLENOID_PIN, GPIO.LOW)  # start closed
        print("  ✅ Solenoid GPIO initialized — Pin 17")

    def fire(self, duration_ms=50):
        """
        Fire solenoid for exactly duration_ms milliseconds
        This is the PRECISION BURST
        """
        duration_s = duration_ms / 1000.0
        GPIO.output(SOLENOID_PIN, GPIO.HIGH)  # open valve
        time.sleep(duration_s)
        GPIO.output(SOLENOID_PIN, GPIO.LOW)   # close valve

    def test_fire(self):
        """Quick test — call once on startup to verify wiring"""
        print("  🔥 Test fire — 50ms burst...")
        self.fire(50)
        print("  ✅ Solenoid test OK")

    def cleanup(self):
        GPIO.output(SOLENOID_PIN, GPIO.LOW)
        GPIO.cleanup()
        print("  GPIO cleaned up")
