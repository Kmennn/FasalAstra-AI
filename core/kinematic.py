# PURPOSE: Calculate exact firing time Ti using kinematic formula
# THE FORMULA:
#   Ti = (D / V) - (inference_latency + mechanical_latency)
#
# WHY SUBTRACT LATENCIES:
#   If we don't subtract, solenoid fires AFTER weed passes
#   We must "lead" the target like a hunter leads a moving bird
#
# LATENCY BREAKDOWN:
#   inference_latency  = 33ms  (YOLOv8-nano on ESP32-S3)
#   mechanical_latency = 7ms   (solenoid valve opening time)
#   total_latency      = 40ms  = 0.040 seconds
#
# EXAMPLE:
#   D = 0.30m, V = 1.2 m/s
#   Raw time = 0.30/1.2 = 250ms
#   Ti = 250ms - 40ms = 210ms ← timer is set to THIS

INFERENCE_LATENCY_S  = 0.033   # 33ms YOLOv8-nano
MECHANICAL_LATENCY_S = 0.007   # 7ms solenoid open time
TOTAL_LATENCY_S      = INFERENCE_LATENCY_S + MECHANICAL_LATENCY_S

SOLENOID_BURST_MS    = 50      # spray duration in ms
MIN_VELOCITY         = 0.1     # m/s — if farmer stops, don't spray
MAX_DISTANCE         = 0.55    # m — beyond this, weed not detectable

class KinematicCalculator:
    def __init__(self):
        self.latency = TOTAL_LATENCY_S

    def calculate_ti(self, distance_m, velocity_ms):
        """
        Calculate Time-to-Impact in milliseconds
        
        distance_m  : real-world distance from weed to nozzle (meters)
        velocity_ms : wand velocity from MPU6050 IMU (meters/second)
        
        Returns: (ti_ms, is_valid, reason)
        """

        # SAFETY CHECK 1: Farmer stopped walking
        if velocity_ms < MIN_VELOCITY:
            return 0, False, "velocity_too_low"

        # SAFETY CHECK 2: Weed too far (out of spray range)
        if distance_m > MAX_DISTANCE:
            return 0, False, "weed_out_of_range"

        # CORE KINEMATIC FORMULA
        raw_time_s = distance_m / velocity_ms
        ti_s       = raw_time_s - self.latency

        # SAFETY CHECK 3: Negative timer (weed already passed)
        if ti_s <= 0:
            return 0, False, "weed_already_passed"

        ti_ms = ti_s * 1000

        return ti_ms, True, "fire"

    def get_spray_duration(self):
        return SOLENOID_BURST_MS


# Quick test
if __name__ == '__main__':
    calc = KinematicCalculator()

    scenarios = [
        (0.30, 1.2,  "Normal walking"),
        (0.15, 1.2,  "Close weed, fast walk"),
        (0.30, 0.5,  "Slow walking"),
        (0.30, 0.05, "Farmer stopped"),
        (0.60, 1.2,  "Weed too far"),
    ]

    print("Kinematic Calculator Test:")
    print("-" * 55)
    for d, v, label in scenarios:
        ti, valid, reason = calc.calculate_ti(d, v)
        status = f"FIRE in {ti:.1f}ms" if valid else f"SKIP ({reason})"
        print(f"  {label:25s} D={d}m V={v}m/s → {status}")
