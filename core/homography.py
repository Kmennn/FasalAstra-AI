# UPDATED FOR Pi Camera OV5647:
# OV5647 has 54° horizontal FOV
# Camera mounted at 45° angle on wand
# Recalibrated distance table for 640x640 resolution
#
# CALIBRATION METHOD:
# Place ruler on ground, capture frame, note pixel Y
# vs actual distance from nozzle tip
# Do this once physically after mounting

import numpy as np

class HomographyMapper:
    def __init__(self,
                 image_height=640,
                 image_width=640):
        self.H = image_height
        self.W = image_width

        # RECALIBRATED FOR Pi Camera OV5647 + 45° mount
        # at 640x640 resolution
        # pixel_y : real_distance_cm from nozzle
        #
        # NOTE: These are APPROXIMATE values
        # Must physically calibrate after mounting
        # See rpi/field_calibrate.py for calibration tool
        self.calibration = {
            0:   65.0,   # top of frame  = 65cm ahead
            128: 50.0,
            256: 35.0,   # center        = 35cm ahead
            384: 20.0,
            512: 8.0,
            640: 0.0     # bottom        = nozzle point
        }

        self.pixels    = np.array(list(self.calibration.keys()))
        self.distances = np.array(list(self.calibration.values()))

    def pixel_to_distance(self, cx, cy):
        distance_cm = np.interp(cy, self.pixels, self.distances)
        distance_m  = distance_cm / 100.0
        return distance_m

    def is_in_frame(self, cx, cy):
        return 0 <= cx <= self.W and 0 <= cy <= self.H
