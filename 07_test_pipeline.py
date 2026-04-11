# Quick automated test of the full pipeline
# Runs on first available test image without user interaction

import cv2
import os
from core.detector   import FasalAstraDetector
from core.homography import HomographyMapper
from core.kinematic  import KinematicCalculator
from core.tracker    import WeedTracker
from core.exclusion  import CropExclusionZone

SIMULATED_SPEED = 1.2  # m/s (walking speed, on real ESP32: from MPU6050)

# Find test image
test_dir = 'datasets/merged/images/train'
images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]

if not images:
    print("❌ No test images found!")
    exit()

# Try the known good weed image first, fall back to any available image
preferred = 'ds1_1000_frame_583_jpg.rf.4b686ff9837763357617ada49eff8b0d.jpg'
if preferred in images:
    chosen_image = preferred
else:
    chosen_image = images[0]
    print(f"⚠️  Preferred test image not found, using: {chosen_image}")

img_path = os.path.join(test_dir, chosen_image)
print(f"\n📸 Testing with: {chosen_image}\n")

# Initialize pipeline
detector   = FasalAstraDetector()
mapper     = HomographyMapper()
calculator = KinematicCalculator()
tracker    = WeedTracker()
exclusion  = CropExclusionZone()

# Load and resize image
frame = cv2.imread(img_path)
if frame is None:
    print(f"❌ Failed to load image: {img_path}")
    exit()
frame = cv2.resize(frame, (320, 320))

# ── DETECTION ──────────────────────────────────────────
all_detections = detector.detect(frame)
weeds = detector.get_weeds_only(all_detections)

print(f"✅ Detections found: {len(all_detections)} total")
print(f"   Weeds: {len(weeds)}")
print(f"   Crops: {sum(1 for d in all_detections if d['class_id'] == 1)}\n")

# ── PIPELINE TEST ──────────────────────────────────────
fire_count = 0
for i, weed in enumerate(weeds, 1):
    cx, cy     = weed['centroid']
    confidence = weed['confidence']
    weed_box   = weed['box']

    print(f"Weed {i}: confidence={confidence:.0%}")
    
    # Anti-double-trigger check
    should_fire, weed_id = tracker.should_fire(cx, cy)
    if not should_fire:
        print(f"  ↳ SKIP: Already tracked (ID={weed_id})")
        continue

    # Crop safety check
    is_safe, overlap, reason = exclusion.is_safe_to_spray(weed_box, all_detections)
    if not is_safe:
        print(f"  ↳ ABORT: {reason} (overlap={overlap:.0%})")
        continue

    # Pixel to real-world distance
    distance_m = mapper.pixel_to_distance(cx, cy)
    print(f"  ↳ Distance: {distance_m*100:.1f}cm")

    # Kinematic calculation
    ti_ms, valid, reason = calculator.calculate_ti(distance_m, SIMULATED_SPEED)
    
    if not valid:
        print(f"  ↳ SKIP: {reason}")
        continue

    # FIRE!
    fire_count += 1
    print(f"  ↳ 🔥 FIRE! Ti={ti_ms:.0f}ms")

print(f"\n{'='*55}")
print("Pipeline Test Complete")
print(f"  Weeds detected   : {len(weeds)}")
print(f"  Solenoid fired   : {fire_count}")
if len(weeds) > 0:
    eff = fire_count / len(weeds) * 100
    print(f"  Efficiency      : {eff:.0f}%")
print(f"{'='*55}\n")
