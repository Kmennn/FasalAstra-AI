# PURPOSE: Simulate the ENTIRE FasalAstra pipeline on your PC
# WHY SIMULATE BEFORE ESP32:
#   Testing logic on PC = instant feedback
#   Testing on ESP32 = flash + wait + debug = slow
#   Get logic perfect here → then port to Arduino
#
# THIS FILE PROVES TO JUDGES:
#   Your system is deterministic and real
#   Every component is working and integrated
#   You have actual numbers (Ti, overlap %, confidence)

import cv2
import time
from core.detector   import FasalAstraDetector
from core.homography import HomographyMapper
from core.kinematic  import KinematicCalculator
from core.tracker    import WeedTracker
from core.exclusion  import CropExclusionZone

# ─── CONFIG ──────────────────────────────────────────────
MODEL_PATH      = 'runs/detect/runs/fasal_astra_v23/weights/best.pt'
SIMULATED_SPEED = 1.2    # m/s — simulated walking speed (on ESP32 from MPU6050)
# ─────────────────────────────────────────────────────────

# Initialize all pipeline components
print("\n" + "=" * 55)
print("  FasalAstra — Predictive Kinematic Pipeline")
print("  Full System Simulation")
print("=" * 55 + "\n")

detector   = FasalAstraDetector(MODEL_PATH)
mapper     = HomographyMapper()
calculator = KinematicCalculator()
tracker    = WeedTracker()
exclusion  = CropExclusionZone()

# Stats tracking
stats = {
    'frames'       : 0,
    'weeds_found'  : 0,
    'fired'        : 0,
    'skipped_track': 0,
    'skipped_crop' : 0,
    'skipped_vel'  : 0,
}

def process_frame(frame):
    """
    Full pipeline — one frame
    Returns annotated frame + action taken
    """
    t_start = time.time()
    stats['frames'] += 1

    # ── STEP 1: DETECTION ────────────────────────────────
    all_detections = detector.detect(frame)
    weeds = detector.get_weeds_only(all_detections)
    stats['weeds_found'] += len(weeds)

    action_log = []

    for weed in weeds:
        cx, cy     = weed['centroid']
        confidence = weed['confidence']
        weed_box   = weed['box']

        # ── STEP 2: ANTI-DOUBLE-TRIGGER ──────────────────
        should_fire, weed_id = tracker.should_fire(cx, cy)
        if not should_fire:
            stats['skipped_track'] += 1
            action_log.append(f"SKIP (already tracked) ID={weed_id}")
            continue

        # ── STEP 3: CROP EXCLUSION ZONE ──────────────────
        is_safe, overlap, reason = exclusion.is_safe_to_spray(
            weed_box, all_detections
        )
        if not is_safe:
            stats['skipped_crop'] += 1
            action_log.append(
                f"ABORT (crop overlap {overlap:.0%}) ID={weed_id}"
            )
            continue

        # ── STEP 4: HOMOGRAPHY — PIXEL TO DISTANCE ───────
        distance_m = mapper.pixel_to_distance(cx, cy)

        # ── STEP 5: KINEMATIC CALCULATION ────────────────
        # NOTE: On real ESP32, velocity comes from MPU6050
        # Here we use simulated walking speed
        velocity_ms = SIMULATED_SPEED

        ti_ms, valid, reason = calculator.calculate_ti(
            distance_m, velocity_ms
        )

        if not valid:
            stats['skipped_vel'] += 1
            action_log.append(f"SKIP ({reason}) ID={weed_id}")
            continue

        # ── STEP 6: FIRE ──────────────────────────────────
        stats['fired'] += 1
        action_log.append(
            f"🔥 FIRE ID={weed_id} "
            f"D={distance_m*100:.1f}cm "
            f"Ti={ti_ms:.0f}ms "
            f"conf={confidence:.0%}"
        )

        # ── ANNOTATE FRAME ────────────────────────────────
        x1,y1,x2,y2 = [int(v) for v in weed_box]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
        cv2.putText(
            frame,
            f"WEED Ti={ti_ms:.0f}ms",
            (x1, y1-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0,0,255), 1
        )

    # Annotate crops in green
    for det in all_detections:
        if det['class_id'] == 1:
            x1,y1,x2,y2 = [int(v) for v in det['box']]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
            cv2.putText(
                frame, "CROP SAFE",
                (x1, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0,200,0), 1
            )

    # ── STATS OVERLAY ─────────────────────────────────────
    t_end = time.time()
    fps = 1 / (t_end - t_start + 0.001)

    overlay_lines = [
        f"FasalAstra v2 | {fps:.0f} FPS",
        f"Frames: {stats['frames']}",
        f"Fired: {stats['fired']}",
        f"Tracked: {tracker.get_active_count()} active weeds",
    ]
    for i, line in enumerate(overlay_lines):
        cv2.putText(
            frame, line,
            (8, 20 + i*18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255,255,255), 1
        )

    return frame, action_log


# ── MAIN LOOP ─────────────────────────────────────────────
print("Choose input source:")
print("  1 → Webcam (live test)")
print("  2 → Video file")
print("  3 → Single image")
choice = input("\nEnter 1/2/3: ").strip()

if choice == '1':
    cap = cv2.VideoCapture(0)
    print("\n✅ Webcam opened — press Q to quit\n")

elif choice == '2':
    path = input("Video file path: ").strip()
    cap = cv2.VideoCapture(path)

elif choice == '3':
    path = input("Image file path: ").strip()
    frame = cv2.imread(path)
    result, logs = process_frame(frame)
    for log in logs:
        print(f"  {log}")
    cv2.imshow('FasalAstra Simulation', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# Video/webcam loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.resize(frame, (320, 320))
    result, logs = process_frame(frame)

    for log in logs:
        print(f"  {log}")

    cv2.imshow('FasalAstra — Kinematic Pipeline Simulation', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# FINAL REPORT
print("\n" + "=" * 55)
print("  FasalAstra — Session Report")
print("=" * 55)
print(f"  Total frames processed : {stats['frames']}")
print(f"  Weeds detected         : {stats['weeds_found']}")
print(f"  Solenoid fired         : {stats['fired']}")
print(f"  Skipped (tracked)      : {stats['skipped_track']}")
print(f"  Aborted (crop overlap) : {stats['skipped_crop']}")
print(f"  Skipped (velocity)     : {stats['skipped_vel']}")
if stats['weeds_found'] > 0:
    eff = stats['fired'] / stats['weeds_found'] * 100
    print(f"\n  Pipeline efficiency    : {eff:.1f}%")
print("=" * 55)
