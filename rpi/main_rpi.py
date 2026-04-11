# PURPOSE: Full FasalAstra pipeline on Raspberry Pi 4
# This is your HACKATHON DEMO file
#
# WHAT IT DOES:
#   1. Pi Camera captures 640x640 at 30 FPS
#   2. YOLOv8n detects weeds in ~8ms on RPi4
#   3. MPU6050 reads real wand velocity
#   4. Kinematic calculator fires at exact Ti
#   5. RPi GPIO triggers solenoid for 50ms
#   6. Live display on HDMI screen (judges can see it!)
#   7. Anti-double-trigger prevents waste
#   8. Crop exclusion zone protects cash crops
#
# RUN: python3 rpi/main_rpi.py

import cv2
import time
import threading
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector   import FasalAstraDetector
from core.homography import HomographyMapper
from core.kinematic  import KinematicCalculator
from core.tracker    import WeedTracker
from core.exclusion  import CropExclusionZone
from rpi.gpio_controller import SolenoidController
from rpi.imu_reader      import IMUReader

# ── CONFIG ────────────────────────────────────────────────
MODEL_PATH   = 'runs/fasal_astra_v3_rpi/weights/best.pt'
SHOW_DISPLAY = True   # set False if no monitor connected
USE_PI_CAM   = True   # set False to use USB webcam for testing
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  🌾 FasalAstra v3 — Raspberry Pi 4 Edition")
print("  Predictive Kinematic Weed Strike System")
print("=" * 55 + "\n")

# Initialize all components
print("Initializing components...")
detector   = FasalAstraDetector(MODEL_PATH)
mapper     = HomographyMapper()
calculator = KinematicCalculator()
tracker    = WeedTracker()
exclusion  = CropExclusionZone()
solenoid   = SolenoidController()
imu        = IMUReader()

# Test solenoid on startup
solenoid.test_fire()
print()

# Initialize camera
if USE_PI_CAM:
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_preview_configuration(
            main={"size": (640, 640), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        time.sleep(1)
        print("  ✅ Pi Camera OV5647 initialized — 640x640")
        use_picam = True
    except Exception as e:
        print(f"  ⚠️ Pi Camera failed: {e}")
        print("  ⚠️ Falling back to USB webcam")
        cap = cv2.VideoCapture(0)
        use_picam = False
else:
    cap = cv2.VideoCapture(0)
    use_picam = False
    print("  ✅ USB webcam initialized")

# Stats
stats = {
    'frames'  : 0,
    'fired'   : 0,
    'aborted' : 0,
    'weeds'   : 0,
    'start'   : time.time()
}

def fire_with_delay(ti_ms, weed_id):
    """
    Fire solenoid after Ti milliseconds
    Runs in background thread so camera never stops
    WHY THREAD: Main loop must keep capturing frames
    Solenoid delay cannot block camera pipeline
    """
    time.sleep(ti_ms / 1000.0)
    solenoid.fire(50)
    print(f"  🔥 FIRED weed ID={weed_id} after {ti_ms:.0f}ms")

print("\n✅ All systems GO — Starting pipeline...\n")
print("  Press Ctrl+C to stop\n")
print("-" * 55)

try:
    while True:
        t_frame = time.time()
        stats['frames'] += 1

        # ── CAPTURE ───────────────────────────────────────
        if use_picam:
            frame = cam.capture_array()
        else:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 640))

        # ── GET REAL VELOCITY FROM IMU ────────────────────
        velocity_ms = imu.get_velocity()

        # ── DETECT ────────────────────────────────────────
        all_detections = detector.detect(frame)
        weeds          = detector.get_weeds_only(all_detections)
        stats['weeds'] += len(weeds)

        for weed in weeds:
            cx, cy   = weed['centroid']
            weed_box = weed['box']
            conf     = weed['confidence']

            # ANTI-DOUBLE-TRIGGER
            should_fire, weed_id = tracker.should_fire(cx, cy)
            if not should_fire:
                continue

            # CROP EXCLUSION ZONE
            is_safe, overlap, _ = exclusion.is_safe_to_spray(
                weed_box, all_detections
            )
            if not is_safe:
                stats['aborted'] += 1
                print(f"  🛡️ ABORTED — crop overlap "
                      f"{overlap:.0%} ID={weed_id}")
                continue

            # HOMOGRAPHY — pixel to real distance
            distance_m = mapper.pixel_to_distance(cx, cy)

            # KINEMATIC CALCULATION
            ti_ms, valid, reason = calculator.calculate_ti(
                distance_m, velocity_ms
            )

            if not valid:
                continue

            # FIRE IN BACKGROUND THREAD
            stats['fired'] += 1
            t = threading.Thread(
                target=fire_with_delay,
                args=(ti_ms, weed_id),
                daemon=True
            )
            t.start()

            print(f"  ⏱️  Timer set: D={distance_m*100:.1f}cm "
                  f"V={velocity_ms:.2f}m/s "
                  f"Ti={ti_ms:.0f}ms "
                  f"conf={conf:.0%}")

        # ── LIVE DISPLAY (judges can see this!) ───────────
        if SHOW_DISPLAY:
            display = frame.copy()

            # Draw detections
            for det in all_detections:
                x1,y1,x2,y2 = [int(v) for v in det['box']]
                if det['class_id'] == 0:   # weed = red
                    color = (0, 0, 255)
                    label = f"WEED {det['confidence']:.0%}"
                elif det['class_id'] == 1: # crop = green
                    color = (0, 200, 0)
                    label = "CROP SAFE"
                else:                       # soil = gray
                    color = (150, 150, 150)
                    label = "SOIL"

                cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
                cv2.putText(display, label,
                           (x1, y1-6),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, color, 1)

            # Stats overlay
            fps = 1 / (time.time() - t_frame + 0.001)
            runtime = time.time() - stats['start']

            lines = [
                f"FasalAstra v3 | RPi4 | {fps:.0f} FPS",
                f"IMU Velocity : {velocity_ms:.2f} m/s  "
                f"({imu.get_status()})",
                f"Fired        : {stats['fired']}",
                f"Crop Aborts  : {stats['aborted']}",
                f"Runtime      : {runtime:.0f}s",
            ]
            for i, line in enumerate(lines):
                cv2.putText(display, line, (8, 20+i*22),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255,255,255), 1)

            cv2.imshow('FasalAstra — Live Field View', display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\n\nStopping...")

finally:
    # Cleanup
    if use_picam:
        cam.stop()
    else:
        cap.release()

    cv2.destroyAllWindows()
    solenoid.cleanup()

    # Final report
    runtime = time.time() - stats['start']
    print("\n" + "=" * 55)
    print("  FasalAstra — Session Complete")
    print("=" * 55)
    print(f"  Runtime        : {runtime:.0f} seconds")
    print(f"  Frames         : {stats['frames']}")
    print(f"  Weeds detected : {stats['weeds']}")
    print(f"  Solenoid fired : {stats['fired']}")
    print(f"  Crop aborted   : {stats['aborted']}")
    if stats['weeds'] > 0:
        safety = (1 - stats['aborted']/
                  max(stats['weeds'],1)) * 100
        print(f"  Crop safety    : {safety:.1f}%")
    print("=" * 55)
