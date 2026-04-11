# PURPOSE: Convert pixel coordinates → real-world cm distance
# WHY HOMOGRAPHY:
#   Camera is fixed at 45° angle on the wand
#   This means pixel Y position directly maps to distance
#   from nozzle tip — we pre-calculate this once
#   so ESP32 doesn't waste clock cycles computing it live
#
# WHY THIS MATTERS FOR KINEMATIC PIPELINE:
#   Ti = D / V
#   D comes from THIS module (pixel → real distance)
#   Without accurate D, the timer fires at wrong time
#   = herbicide misses the weed by centimeters

import numpy as np

class HomographyMapper:
    def __init__(self, image_height=320, image_width=320):
        self.H = image_height
        self.W = image_width

        # CALIBRATION TABLE
        # Measured physically by placing ruler under camera
        # at 45° angle — done ONCE during setup
        #
        # pixel_y : real_distance_cm from nozzle
        self.calibration = {
            0:   50.0,   # top of frame = 50cm ahead
            80:  40.0,
            160: 30.0,   # center = 30cm ahead
            240: 15.0,
            320: 0.0     # bottom = directly under nozzle
        }

        # Build interpolation arrays
        self.pixels = np.array(list(self.calibration.keys()))
        self.distances = np.array(list(self.calibration.values()))

    def pixel_to_distance(self, cx, cy):
        """
        Convert weed centroid pixel (cx, cy) to real distance in cm
        cx = center x of bounding box (not used — distance is Y only)
        cy = center y of bounding box
        Returns: distance in METERS (for kinematic formula)
        """
        # Interpolate between calibration points
        distance_cm = np.interp(cy, self.pixels, self.distances)
        distance_m  = distance_cm / 100.0

        return distance_m

    def is_in_frame(self, cx, cy):
        return 0 <= cx <= self.W and 0 <= cy <= self.H


# Quick test
if __name__ == '__main__':
    mapper = HomographyMapper()

    test_points = [
        (160, 0,   "Top of frame"),
        (160, 160, "Center"),
        (160, 240, "Lower area"),
        (160, 320, "Nozzle point"),
    ]

    print("Homography Calibration Test:")
    print("-" * 40)
    for cx, cy, label in test_points:
        d = mapper.pixel_to_distance(cx, cy)
        print(f"  {label:20s} → {d*100:.1f} cm ({d:.3f} m)")
