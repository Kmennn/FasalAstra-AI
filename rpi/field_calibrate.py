# PURPOSE: Physically calibrate the Homography Table
# for YOUR specific camera mount angle and height
#
# WHY THIS IS CRITICAL:
#   Every physical setup is different:
#   - Camera height from ground varies
#   - Exact angle of 45° is never perfect
#   - Lens distortion shifts pixel positions
#   - Wrong calibration = solenoid fires 5-10cm off target
#     = weed survives = entire system fails
#
# HOW TO USE (do this ONCE after mounting camera):
#   1. Place a ruler flat on the ground under the wand
#   2. Run this script on RPi4
#   3. Click on the ruler markings in the live view
#   4. Script auto-generates your homography table
#   5. Paste the output into core/homography.py
#
# EXPECTED TIME: 10-15 minutes

import cv2
import numpy as np
import json
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── CONFIG ────────────────────────────────────────────────
USE_PI_CAM      = True
SAVE_PATH       = 'rpi/calibration_data.json'
OUTPUT_PY_PATH  = 'rpi/calibration_output.py'
IMAGE_SIZE      = 640
# ─────────────────────────────────────────────────────────

# Calibration points collected by user
# Format: [(pixel_y, real_distance_cm), ...]
calibration_points = []

# Current frame (global for mouse callback)
current_frame = None

def mouse_callback(event, x, y, flags, param):
    """
    User clicks on a known distance marking on ruler
    Records pixel_y → real distance
    """
    global calibration_points, current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        # Ask user to type the real distance
        print(f"\n  📍 You clicked pixel position Y={y}")
        dist = input("  Enter real distance in CM from nozzle tip: ")

        try:
            dist_cm = float(dist)
            calibration_points.append((y, dist_cm))
            print(f"  ✅ Recorded: pixel_y={y} → {dist_cm}cm")
            print(f"  Total points: {len(calibration_points)}")
            print(f"  Need minimum 5 points for good calibration")

            # Draw marker on frame
            cv2.circle(current_frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(
                current_frame,
                f"{dist_cm}cm",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

        except ValueError:
            print("  ❌ Invalid input — enter a number like 30")

def generate_homography_table(points):
    """
    Fit a smooth curve through calibration points
    Returns dictionary of pixel_y: distance_cm
    at standard intervals
    """
    if len(points) < 3:
        print("❌ Need at least 3 points!")
        return None

    # Sort by pixel_y
    points_sorted = sorted(points, key=lambda p: p[0])

    pixels    = np.array([p[0] for p in points_sorted], dtype=float)
    distances = np.array([p[1] for p in points_sorted], dtype=float)

    # Fit polynomial curve (degree 2 = smooth arc)
    # WHY POLYNOMIAL: Real-world distance vs pixel
    # is not linear due to perspective — it's a curve
    coeffs = np.polyfit(pixels, distances, deg=2)
    poly   = np.poly1d(coeffs)

    # Generate clean table at standard intervals
    table = {}
    intervals = [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640]

    for px in intervals:
        dist = float(poly(px))
        dist = max(0.0, round(dist, 1))  # no negative distances
        table[px] = dist

    return table, coeffs

def save_results(table, coeffs, raw_points):
    """Save calibration data and generate ready-to-paste code"""

    # Save raw data as JSON
    data = {
        'timestamp'   : time.strftime('%Y-%m-%d %H:%M:%S'),
        'raw_points'  : raw_points,
        'poly_coeffs' : coeffs.tolist(),
        'table'       : {str(k): v for k, v in table.items()}
    }

    with open(SAVE_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n  ✅ Raw data saved: {SAVE_PATH}")

    # Generate ready-to-paste Python code
    lines = []
    lines.append("# ── AUTO-GENERATED CALIBRATION ──────────────────")
    lines.append("# Generated: " + time.strftime('%Y-%m-%d %H:%M:%S'))
    lines.append("# Raw calibration points collected:")
    for px, cm in sorted(raw_points, key=lambda p: p[0]):
        lines.append(f"#   pixel_y={px:4d} → {cm:.1f}cm from nozzle")
    lines.append("")
    lines.append("# PASTE THIS INTO core/homography.py")
    lines.append("# Replace the self.calibration dictionary")
    lines.append("")
    lines.append("self.calibration = {")
    for px, dist in sorted(table.items()):
        lines.append(f"    {px:4d}: {dist:.1f},   # {dist:.1f}cm from nozzle")
    lines.append("}")

    with open(OUTPUT_PY_PATH, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  ✅ Ready-to-paste code: {OUTPUT_PY_PATH}")

def run_calibration():
    global current_frame

    print("\n" + "=" * 55)
    print("  FasalAstra — Field Calibration Tool")
    print("  Pi Camera OV5647 + 45° Wand Mount")
    print("=" * 55)
    print("""
  SETUP INSTRUCTIONS:
  ─────────────────────────────────────────────
  1. Mount camera on wand at your actual angle
  2. Hold wand at normal spraying height
  3. Place a measuring tape flat on the ground
     pointing away from the nozzle tip
  4. Run this script and look at the live view
  5. Click on each distance marking you can see
     (e.g. click on 10cm mark, type "10")
  6. Collect at least 6 points for accuracy
  7. Press S to save and generate calibration
  8. Press Q to quit without saving
  ─────────────────────────────────────────────
    """)

    input("  Press ENTER when camera is mounted and ready...")

    # Initialize camera
    if USE_PI_CAM:
        try:
            from picamera2 import Picamera2
            cam    = Picamera2()
            config = cam.create_preview_configuration(
                main={"size": (IMAGE_SIZE, IMAGE_SIZE),
                      "format": "RGB888"}
            )
            cam.configure(config)
            cam.start()
            time.sleep(1)
            print("\n  ✅ Pi Camera started")
            use_picam = True
        except Exception as e:
            print(f"\n  ⚠️ Pi Camera error: {e}")
            print("  Using USB webcam instead")
            cap = cv2.VideoCapture(0)
            use_picam = False
    else:
        cap = cv2.VideoCapture(0)
        use_picam = False
        print("\n  ✅ Webcam started")

    # Setup display window
    window_name = 'FasalAstra Calibration — Click ruler markings'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n  📸 Live view active")
    print("  👆 Click on ruler markings + type distance")
    print("  S = Save calibration | Q = Quit\n")

    while True:
        # Capture frame
        if use_picam:
            frame = cam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

        current_frame = frame.copy()

        # Draw horizontal reference lines every 64 pixels
        for py in range(0, IMAGE_SIZE + 1, 64):
            cv2.line(frame, (0, py), (IMAGE_SIZE, py),
                    (60, 60, 60), 1)
            cv2.putText(frame, f"y={py}",
                       (2, py - 3),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.35, (100, 100, 100), 1)

        # Draw collected points
        for py, cm in calibration_points:
            cv2.line(frame, (0, py), (IMAGE_SIZE, py),
                    (0, 255, 0), 2)
            cv2.putText(frame, f"{cm}cm",
                       (IMAGE_SIZE - 70, py - 4),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.55, (0, 255, 0), 2)

        # Status overlay
        status_lines = [
            "CALIBRATION MODE",
            f"Points: {len(calibration_points)}/6 minimum",
            "Click ruler → type distance",
            "S=Save  Q=Quit",
        ]
        for i, line in enumerate(status_lines):
            cv2.putText(frame, line,
                       (8, 20 + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 255, 255) if i == 0 else (255, 255, 255),
                       1)

        # Progress bar
        progress = min(len(calibration_points) / 6.0, 1.0)
        bar_w    = int(IMAGE_SIZE * progress)
        cv2.rectangle(frame, (0, IMAGE_SIZE-8),
                     (bar_w, IMAGE_SIZE),
                     (0, 255, 0), -1)
        cv2.putText(frame,
                   f"Progress: {len(calibration_points)}/6",
                   (IMAGE_SIZE//2 - 50, IMAGE_SIZE - 10),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, (0,0,0), 1)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            if len(calibration_points) < 3:
                print("  ❌ Need at least 3 points first!")
                continue

            print("\n  💾 Generating calibration table...")
            result = generate_homography_table(calibration_points)

            if result:
                table, coeffs = result

                print("\n  📊 Generated Homography Table:")
                print("  " + "-" * 35)
                for px, dist in sorted(table.items()):
                    bar = "█" * int(dist / 3)
                    print(f"  pixel_y={px:4d} → {dist:5.1f}cm  {bar}")

                save_results(table, coeffs, calibration_points)

                print(f"""
  ✅ CALIBRATION COMPLETE!
  ─────────────────────────────────────────
  Next steps:
  1. Open: {OUTPUT_PY_PATH}
  2. Copy the self.calibration dictionary
  3. Paste into: core/homography.py
  4. Replace the existing calibration table
  5. Run: python3 rpi/main_rpi.py
  ─────────────────────────────────────────
                """)
                break

        elif key == ord('q'):
            print("\n  Calibration cancelled")
            break

    # Cleanup
    if use_picam:
        cam.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()

def verify_calibration():
    """
    After calibration, verify by showing predicted
    distances live on camera feed
    """
    print("\n" + "=" * 55)
    print("  FasalAstra — Calibration Verification")
    print("=" * 55)

    # Load saved calibration
    if not os.path.exists(SAVE_PATH):
        print("  ❌ No calibration found. Run calibration first!")
        return

    with open(SAVE_PATH) as f:
        data = json.load(f)

    coeffs = np.array(data['poly_coeffs'])
    poly   = np.poly1d(coeffs)

    print("  📸 Live verification view")
    print("  Move ruler under camera — check if cm labels match")
    print("  Q = Quit verification\n")

    if USE_PI_CAM:
        try:
            from picamera2 import Picamera2
            cam    = Picamera2()
            config = cam.create_preview_configuration(
                main={"size": (IMAGE_SIZE, IMAGE_SIZE),
                      "format": "RGB888"}
            )
            cam.configure(config)
            cam.start()
            time.sleep(1)
            use_picam = True
        except Exception:
            cap = cv2.VideoCapture(0)
            use_picam = False
    else:
        cap = cv2.VideoCapture(0)
        use_picam = False

    while True:
        if use_picam:
            frame = cam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

        # Overlay predicted distances on every row
        for py in range(0, IMAGE_SIZE, 32):
            dist_cm = float(poly(py))
            dist_cm = max(0.0, dist_cm)

            # Color: green = close, red = far
            ratio = min(dist_cm / 65.0, 1.0)
            color = (
                int(255 * (1 - ratio)),
                int(255 * (1 - ratio)),
                int(255 * ratio)
            )

            cv2.putText(frame,
                       f"{dist_cm:.1f}cm",
                       (IMAGE_SIZE - 75, py + 12),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, color, 1)
            cv2.line(frame, (IMAGE_SIZE - 80, py),
                    (IMAGE_SIZE, py), color, 1)

        cv2.putText(frame, "VERIFICATION MODE",
                   (8, 20),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Check right labels match ruler",
                   (8, 44),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.45, (255, 255, 255), 1)

        cv2.imshow('FasalAstra — Calibration Verification', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    if use_picam:
        cam.stop()
    else:
        cap.release()
    cv2.destroyAllWindows()


# ── MAIN ──────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  Choose mode:")
    print("  1 → Run calibration (first time)")
    print("  2 → Verify existing calibration")
    choice = input("\n  Enter 1 or 2: ").strip()

    if choice == '1':
        run_calibration()
    elif choice == '2':
        verify_calibration()
    else:
        print("  Invalid choice")
