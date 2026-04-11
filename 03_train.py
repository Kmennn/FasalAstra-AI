# FasalAstra v3 — Optimized Training for 16,000 Image Dataset
#
# WHY EVERYTHING CHANGED vs OLD VERSION:
# ┌─────────────────────────────────────────────────────┐
# │ Old: 3,500 images → epochs=50, batch=16, lr=0.001  │
# │ New: 16,000 images → epochs=100, batch=32, lr=0.01 │
# │                                                      │
# │ Rule: More data = needs more epochs to fully learn  │
# │       More data = can afford higher learning rate   │
# │       More data = bigger batch = more stable grad   │
# └─────────────────────────────────────────────────────┘

from ultralytics import YOLO
import torch
import os
import time
import yaml

# ── PRE-FLIGHT CHECKS ─────────────────────────────────────
print("\n" + "="*60)
print("  FasalAstra v3 — Optimized 16K Dataset Training")
print("  RTX 3050 | YOLOv8-nano | Indian Farm Conditions")
print("="*60)

# GPU check
if not torch.cuda.is_available():
    print("\n❌ NO GPU DETECTED!")
    print("   Training on CPU will take 10-15x longer")
    print("   Make sure CUDA drivers are installed")
    input("   Press ENTER to continue on CPU anyway...")
    DEVICE = 'cpu'
else:
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n  GPU    : {gpu_name}")
    print(f"  VRAM   : {vram_gb:.1f} GB")
    print(f"  CUDA   : {torch.version.cuda}")
    DEVICE = 0

    # RTX 3050 has 4GB VRAM
    # Warn if less than expected
    if vram_gb < 3.5:
        print(f"\n  ⚠️  Low VRAM detected ({vram_gb:.1f}GB)")
        print(f"  ⚠️  Reducing batch size automatically")
        BATCH_SIZE = 16
    else:
        BATCH_SIZE = 32
        print(f"\n  ✅ VRAM sufficient — using batch={BATCH_SIZE}")

# Dataset check
YAML_PATH = 'datasets/merged/data.yaml'

if not os.path.exists(YAML_PATH):
    print("\n❌ datasets/merged/data.yaml not found!")
    print("   Run 02_merge.py first!")
    exit()

# Count total images
with open(YAML_PATH) as f:
    yaml_data = yaml.safe_load(f)

train_dir = os.path.join(
    yaml_data.get('path','datasets/merged'),
    'images/train'
)
val_dir = os.path.join(
    yaml_data.get('path','datasets/merged'),
    'images/val'
)

train_count = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
val_count   = len(os.listdir(val_dir))   if os.path.exists(val_dir)   else 0
total_count = train_count + val_count

print(f"\n  📊 Dataset Statistics:")
print(f"     Train images : {train_count:,}")
print(f"     Val images   : {val_count:,}")
print(f"     Total        : {total_count:,}")
print(f"     Classes      : {yaml_data.get('nc','?')} "
      f"({yaml_data.get('names',{})})")

# ── SMART TRAINING PARAMETERS ─────────────────────────────
#
# WHY EPOCHS = 100 (was 50):
#   16,000 images has 4x more data to learn
#   Model needs more passes to fully absorb all weed varieties
#   Each epoch = model sees all 16k images once
#   50 epochs on 16k = model still "discovering" patterns
#   100 epochs = fully converged, no patterns left unseen
#
# WHY LR = 0.01 (was 0.001):
#   More data = gradients are more reliable (less noisy)
#   Higher LR = faster convergence on reliable gradients
#   With 3.5k images, high LR caused unstable training
#   With 16k images, high LR is safe and faster
#
# WHY WARMUP = 5 EPOCHS (was 3):
#   Larger dataset = more diverse initial batches
#   Warmup prevents early chaotic updates
#   Gradually ramps LR from 0 → 0.01 over 5 epochs
#   Especially important because our data has 10 sources
#   (each source has slightly different distribution)
#
# WHY PATIENCE = 20 (was 10):
#   With more data, improvements come slower
#   patience=10 would stop training too early
#   Model might still be learning at epoch 60-70
#   patience=20 gives it enough room to keep improving
#
# WHY CLOSE_MOSAIC = 15 (was default):
#   Mosaic augmentation combines 4 images into 1
#   Great for early learning but hurts final accuracy
#   Turning it off for last 15 epochs = cleaner final weights
#   Standard trick for large dataset training

EPOCHS        = 100
LR_INITIAL    = 0.01
LR_FINAL      = 0.001   # cosine decay target
WARMUP_EPOCHS = 5
PATIENCE      = 20
CLOSE_MOSAIC  = 15      # turn off mosaic last N epochs
IMAGE_SIZE    = 640

print(f"\n  🎯 Training Configuration:")
print(f"     Epochs        : {EPOCHS}")
print(f"     Batch size    : {BATCH_SIZE}")
print(f"     Image size    : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"     Learning rate : {LR_INITIAL} → {LR_FINAL} (cosine)")
print(f"     Warmup        : {WARMUP_EPOCHS} epochs")
print(f"     Patience      : {PATIENCE} epochs")
print(f"     Close mosaic  : last {CLOSE_MOSAIC} epochs")

# Estimate training time
# RTX 3050: ~45s per epoch at 640x640 batch=32
mins_per_epoch = 45 / 60
est_minutes    = EPOCHS * mins_per_epoch
print(f"\n  ⏱️  Estimated time : ~{est_minutes:.0f} mins "
      f"({est_minutes/60:.1f} hrs) on RTX 3050")
print(f"     (with early stopping could be less)\n")

try:
    input("  Press ENTER to start training...")
except EOFError:
    print("  Auto-starting training in non-interactive mode...")
start_time = time.time()

# ── LOAD MODEL ────────────────────────────────────────────
#
# WHY yolov8n.pt (pretrained):
#   Already knows 80 COCO classes including plants
#   Has strong feature extractors from ImageNet pretraining
#   We just add "weed / crop / soil" knowledge on top
#   Training from scratch would need 5x more data and time
#
# ALTERNATIVE we considered but rejected:
#   yolov8s.pt (small) → 2x bigger → won't fit RPi4 at 30fps
#   yolov8n-seg.pt     → segmentation overkill for our use
#   Custom backbone    → too complex, no benefit for this task

print("  Loading YOLOv8-nano pretrained weights...")
model = YOLO('yolov8n.pt')
print("  ✅ Base model loaded\n")

# ── TRAIN ─────────────────────────────────────────────────
print("  🚀 Starting training...\n")
print("  Watch for these good signs:")
print("  → box_loss decreasing each epoch")
print("  → mAP50 increasing past epoch 20")
print("  → No 'nan' values in losses\n")
print("─"*60)

results = model.train(
    data    = YAML_PATH,
    project = 'runs',
    name    = 'fasal_astra_v3_16k',
    exist_ok= True,

    # ── CORE PARAMETERS ───────────────────────────────────
    epochs      = EPOCHS,
    imgsz       = IMAGE_SIZE,
    batch       = BATCH_SIZE,
    device      = DEVICE,

    # ── LEARNING RATE SCHEDULE ────────────────────────────
    # WHY COSINE DECAY:
    #   Starts at lr0, smoothly decays to lrf
    #   Better than step decay for large datasets
    #   Final epochs train with low LR = fine-tuning mode
    optimizer     = 'AdamW',
    lr0           = LR_INITIAL,
    lrf           = 0.1,        # final lr = lr0 * lrf = 0.001
    momentum      = 0.937,
    weight_decay  = 0.0005,
    warmup_epochs = WARMUP_EPOCHS,
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.1,

    # ── EARLY STOPPING ────────────────────────────────────
    patience = PATIENCE,

    # ── SAVING ────────────────────────────────────────────
    save         = True,
    save_period  = 10,    # save checkpoint every 10 epochs

    # ── AUGMENTATIONS ─────────────────────────────────────
    # Each augmentation maps to a REAL FasalAstra use case:
    #
    # hsv_h=0.015 → Different weed species have diff hues
    # hsv_s=0.7   → Dry Indian summer vs monsoon wet soil
    # hsv_v=0.4   → Morning shadow vs harsh 2pm sunlight
    # degrees=5   → Wand held at slightly different angles
    # translate=0.1→ Wand moves forward during sweep
    # scale=0.6   → Small seedling weeds vs mature weeds
    # shear=3     → Camera perspective shift during sweep
    # flipud=0.3  → Occasional upward camera tilt
    # fliplr=0.5  → Left/right sweep direction
    # mosaic=1.0  → Simulate multiple weeds in one frame
    # copy_paste=0.4→ Paste weeds into crop images
    #                  (CRITICAL: teaches crop-weed proximity)
    # close_mosaic → Turn off mosaic for final 15 epochs
    #                (cleaner convergence)
    hsv_h        = 0.015,
    hsv_s        = 0.7,
    hsv_v        = 0.4,
    degrees      = 5.0,
    translate    = 0.1,
    scale        = 0.6,
    shear        = 3.0,
    flipud       = 0.3,
    fliplr       = 0.5,
    mosaic       = 1.0,
    copy_paste   = 0.4,
    close_mosaic = CLOSE_MOSAIC,

    # ── LOSS WEIGHTS ──────────────────────────────────────
    # WHY box=10 (increased from 7.5):
    #   Precise bounding box = precise centroid
    #   Precise centroid = accurate kinematic calculation
    #   Inaccurate centroid = solenoid misses weed by cm
    #   This is our most critical metric
    #
    # WHY cls=0.8 (increased from 0.5):
    #   With 10 diverse datasets, class confusion increases
    #   Higher cls weight = model more carefully
    #   distinguishes weed vs crop vs soil
    #   Prevents "crop classified as weed" = false spray
    box = 10.0,
    cls = 0.8,
    dfl = 1.5,

    # ── MULTI-SCALE TRAINING ──────────────────────────────
    # WHY multi_scale=True:
    #   Trains on images from 480px to 800px randomly
    #   Makes model robust to different heights
    #   Farmer holds wand at different heights = different scale
    #   This single setting dramatically improves real-world perf
    multi_scale = True,

    # ── VALIDATION ────────────────────────────────────────
    val     = True,
    plots   = True,    # saves training curves to runs folder
)

# ── POST-TRAINING REPORT ──────────────────────────────────
elapsed     = time.time() - start_time
elapsed_min = elapsed / 60

print("\n" + "="*60)
print("  FasalAstra v3 — Training Complete!")
print("="*60)
print(f"\n  ⏱️  Total time    : {elapsed_min:.1f} minutes")
print(f"  📁 Results saved : runs/fasal_astra_v3_16k/")
print(f"  🏆 Best model    : runs/fasal_astra_v3_16k/"
      f"weights/best.pt")
print(f"  💾 Last model    : runs/fasal_astra_v3_16k/"
      f"weights/last.pt")

# Print final metrics if available
try:
    metrics   = results.results_dict
    map50     = metrics.get('metrics/mAP50(B)',     0)
    map95     = metrics.get('metrics/mAP50-95(B)',  0)
    precision = metrics.get('metrics/precision(B)', 0)
    recall    = metrics.get('metrics/recall(B)',    0)

    print(f"\n  📊 Final Model Metrics:")
    print(f"     Precision  : {precision:.2%}")
    print(f"     Recall     : {recall:.2%}")
    print(f"     mAP50      : {map50:.2%}")
    print(f"     mAP50-95   : {map95:.2%}")

    # Verdict
    print(f"\n  📋 Deployment Verdict:")
    if map50 > 0.90:
        print("     🟢 OUTSTANDING — exceeds target!")
        print("        Push straight to RPi4 deployment")
    elif map50 > 0.85:
        print("     🟢 EXCELLENT — production ready")
        print("        Safe for hackathon demo")
    elif map50 > 0.75:
        print("     🟡 GOOD — demo ready")
        print("        Consider adding 200 farm photos")
    else:
        print("     🔴 NEEDS WORK")
        print("        Check label consistency in datasets")
        print("        Run: python 04_evaluate.py for details")

except Exception:
    print("\n  ℹ️  Run python 04_evaluate.py for full metrics")

print(f"""
  🚀 NEXT STEPS:
  ─────────────────────────────────────────────────
  1. python 04_evaluate.py   → check full metrics
  2. python 05_export.py     → export ONNX for RPi4
  3. Copy to RPi4:
     scp runs/fasal_astra_v3_16k/weights/best.pt \\
         pi@raspberrypi.local:~/FasalAstra_AI/runs/
  4. python3 rpi/main_rpi.py → LIVE DEMO 🌾
  ─────────────────────────────────────────────────
""")
