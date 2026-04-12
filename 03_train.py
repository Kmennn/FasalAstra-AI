# FasalAstra v3 — Maximum Performance Training
#
# DATASET: 14,395 train + 1,600 val — ALL 100% labeled ✅
# TRAINING HARDWARE: RTX 3050 Laptop GPU (4GB VRAM)
# DEPLOYMENT TARGET: Raspberry Pi 4 — targeting 8-10 FPS at 320px
#
# GPU MAXIMISATION:
#   imgsz=320px  → yolov8n feature maps are tiny → fits large batches
#   batch=128    → fills ~3.8GB VRAM, maximises gradient quality
#   cudnn.benchmark → auto-picks fastest conv kernels for 320px
#   workers=2    → parallel data loading (safe on Windows with guard)
#
# WINDOWS MULTIPROCESSING FIX:
# ALL executable code must be inside if __name__ == '__main__':
# PyTorch spawns child processes that re-import this file.
# Without the guard, every worker re-runs the whole script → crash.

# ── IMPORTS ONLY at module level (safe for child processes) ──
from ultralytics import YOLO
import torch
import torch.backends.cudnn as cudnn
import os
import time
import yaml


if __name__ == '__main__':

    # ── PRE-FLIGHT CHECKS ─────────────────────────────────────
    print("\n" + "="*60)
    print("  FasalAstra v3 — Maximum Performance Training")
    print("  RTX 3050 | YOLO26-nano | imgsz=320 | RPi4 Deploy")
    print("="*60)

    # GPU check
    if not torch.cuda.is_available():
        print("\n❌ NO GPU DETECTED!")
        print("   Training on CPU will take 10-15x longer")
        print("   Make sure CUDA drivers are installed")
        input("   Press ENTER to continue on CPU anyway...")
        DEVICE = 'cpu'
        BATCH_SIZE = 8
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU    : {gpu_name}")
        print(f"  VRAM   : {vram_gb:.1f} GB")
        print(f"  CUDA   : {torch.version.cuda}")
        DEVICE = 0

        # RTX 3050 has 4GB VRAM — warn if less than expected
        # ── MAX PERFORMANCE: unlock cuDNN auto-tuner ──────────
        # benchmark=True: cuDNN benchmarks conv algorithms on first
        # batch and picks the FASTEST one for your exact input size.
        # Safe because multi_scale=False → fixed 640px every batch.
        # Disabled by default in PyTorch to be deterministic.
        # cuDNN comment updated: safe because fixed 320px every batch.
        cudnn.benchmark     = True
        cudnn.deterministic = False   # fastest mode
        print(f"  ⚡ cuDNN benchmark mode: ON")

        # ── BATCH SIZE at 320px — SAFE VRAM BUDGET ────────────────
        # LESSON LEARNED: batch=128 OOM-crashed at epoch 5.
        # WHY: Mosaic augmentation combines 4 images → some batches
        # get 700+ instances instead of the usual 330.
        # That spike + 4.49GB baseline = RuntimeError: CUDA OOM.
        #
        # RTX 3050 4GB @ 320px yolov8n ACTUAL measured usage:
        #   batch=128 → 4.49-4.68G → OOM on dense mosaic batches ❌
        #   batch=64  → ~2.8-3.2G  → stable, plenty of headroom   ✅
        #   batch=32  → ~1.5-1.8G  → too conservative (underutilised)
        # batch=64 is the REAL max for 4GB VRAM with mosaic ON.
        if vram_gb < 3.0:
            BATCH_SIZE = 16
            print(f"\n  \u26a0\ufe0f  Low VRAM ({vram_gb:.1f}GB) — using batch=16")
        elif vram_gb < 4.0:
            BATCH_SIZE = 32
            print(f"\n  \u26a1 VRAM {vram_gb:.1f}GB — using batch=32")
        else:
            BATCH_SIZE = 64
            print(f"\n  \u2705 VRAM {vram_gb:.1f}GB — using batch=64 (safe max with mosaic)")

    # Dataset check
    YAML_PATH = 'datasets/merged/data.yaml'

    if not os.path.exists(YAML_PATH):
        print("\n❌ datasets/merged/data.yaml not found!")
        print("   Run 02_merge.py first!")
        exit()

    # Delete stale YOLO cache files before training.
    # WHY: YOLO caches label scans in .cache files.
    # If cache was built when val labels were missing/empty,
    # YOLO will keep reporting "0 labeled images" every run
    # even after you fix the labels — because it reads cache
    # instead of rescanning. Deleting forces a fresh scan.
    for cache_file in [
        'datasets/merged/labels/train.cache',
        'datasets/merged/labels/val.cache',
    ]:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"  🗑️  Deleted stale cache: {cache_file}")

    # Count total images
    with open(YAML_PATH) as f:
        yaml_data = yaml.safe_load(f)

    train_dir = os.path.join(
        yaml_data.get('path', 'datasets/merged'),
        'images/train'
    )
    val_dir = os.path.join(
        yaml_data.get('path', 'datasets/merged'),
        'images/val'
    )

    train_count = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
    val_count   = len(os.listdir(val_dir))   if os.path.exists(val_dir)   else 0
    total_count = train_count + val_count

    print("\n  📊 Dataset Statistics:")
    print(f"     Train images : {train_count:,}")
    print(f"     Val images   : {val_count:,}")
    print(f"     Total        : {total_count:,}")
    print(f"     Classes      : {yaml_data.get('nc', '?')} "
          f"({yaml_data.get('names', {})})")

    # ── RPi4-OPTIMISED TRAINING PARAMETERS ────────────────────────
    #
    # WHY IMAGE_SIZE = 320 (was 640):
    #   RPi 4 Cortex-A72 @ 1.8GHz, no GPU, ~13.5 GFLOPS.
    #   YOLOv8n @ 640px → inference ~3,300ms → 0.3 FPS  (unusable)
    #   YOLOv8n @ 320px → inference   ~110ms → 8-10 FPS (real-time ✓)
    #   Area is 1/4, ops are 1/4 → same accuracy for weed detection
    #   (weeds are large objects in frame, don't need sub-pixel detail)
    #
    # WHY EPOCHS = 50 (was 100):
    #   320px images carry 1/4 complexity → model converges faster.
    #   50 epochs is enough for full convergence at this resolution.
    #   Also: training is faster per epoch → 50 takes same wall-time.
    #
    # ── WHY 200 EPOCHS (increased from 100): ───────────────────────
    # More epochs = model sees more augmented variations of the data
    # With 14K images at 64 batch: 220 steps/epoch = ~44,000 total steps
    # 200 epochs gives the model time to fully converge
    # Early stopping (patience=20) will stop early if val plateaus
    EPOCHS        = 200
    LR_INITIAL    = 0.01
    LR_FINAL      = 0.001   # cosine decay target
    WARMUP_EPOCHS = 5
    PATIENCE      = 20      # wait 20 epochs before early stop
    CLOSE_MOSAIC  = 15      # turn off mosaic last 15 epochs
    IMAGE_SIZE    = 320     # ← KEY: RPi4 real-time inference size

    print("\n  🎯 Training Configuration:")
    print(f"     Model         : YOLO26n (upgraded from YOLO11n)")
    print(f"     Epochs        : {EPOCHS}")
    print(f"     Batch size    : {BATCH_SIZE}")
    print(f"     Image size    : {IMAGE_SIZE}x{IMAGE_SIZE}  <- RPi4 real-time")
    print(f"     Learning rate : {LR_INITIAL} -> {LR_FINAL} (cosine)")
    print(f"     Warmup        : {WARMUP_EPOCHS} epochs")
    print(f"     Patience      : {PATIENCE} epochs")
    print(f"     Close mosaic  : last {CLOSE_MOSAIC} epochs")
    print()
    print("  RPi4 Deployment Preview:")
    print("     YOLO26n + ONNX  : ~9-10 FPS  (good)")
    print("     YOLO26n + NCNN  : ~18-22 FPS (best - export after training)")
    print("     YOLOv8n old way : ~8 FPS     (what we upgraded from)")

    # Estimate: RTX 3050 ~25-30s per epoch at 320x320 batch=64
    mins_per_epoch = 30 / 60
    est_minutes    = EPOCHS * mins_per_epoch
    print(f"\n  Estimated time : ~{est_minutes:.0f} mins "
          f"({est_minutes/60:.1f} hrs) on RTX 3050")
    print("     (early stopping may kick in at ~60-80 epochs)\n")

    # ── AUTO-RESUME FROM LAST CHECKPOINT ─────────────────────────────
    # If last.pt exists from a previous crashed/interrupted run,
    # resume from it automatically instead of starting over.
    CHECKPOINT_PATH = 'runs/detect/runs/fasal_astra_rpi4/weights/last.pt'
    RESUME_TRAINING = False
    MODEL_PATH      = 'yolo26n.pt'

    if os.path.exists(CHECKPOINT_PATH):
        MODEL_PATH      = CHECKPOINT_PATH
        RESUME_TRAINING = True
        print(f"  RESUMING from checkpoint: {CHECKPOINT_PATH}")
        print("  (previous epochs are NOT lost)\n")
    else:
        print("  Starting fresh training from yolo26n.pt weights\n")

        print("  Auto-starting training in non-interactive mode...")

    start_time = time.time()

    # ── LOAD MODEL ────────────────────────────────────────────
    #
    # WHY YOLO26n (upgraded from YOLO11n/v8n):
    #   YOLO26n (latest) vs YOLOv8n comparison:
    #   ┌─────────────────┬──────────┬─────────┐
    #   │ Metric          │ YOLOv8n  │ YOLO26n │
    #   ├─────────────────┼──────────┼─────────┤
    #   │ Parameters      │ 3.2M     │ 2.6M ✅  │
    #   │ COCO mAP50      │ 37.3%    │ >39.5% ✅│
    #   │ RPi4 speed      │ ~8 FPS   │ ~9 FPS  │
    #   └─────────────────┴──────────┴─────────┘
    #   Same training pipeline, better accuracy, smaller footprint.
    #   Zero downside — pure upgrade.
    #
    # DEPLOYMENT NOTE:
    #   After training, export to NCNN format (not ONNX) for RPi4.
    #   NCNN uses ARM NEON SIMD → 2x faster than ONNX on Cortex-A72.
    #   Target: 18-22 FPS with NCNN vs 8-10 FPS with ONNX.
    #   See 05_export.py for NCNN export instructions.

    print("  Loading YOLO26-nano pretrained weights...")
    model = YOLO(MODEL_PATH)
    print(f"  YOLO26n loaded: {MODEL_PATH}\n")


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
        name    = 'fasal_astra_rpi4',
        exist_ok= True,
        resume      = RESUME_TRAINING,         # auto-resumes from last.pt if crashed

        # ── CORE PARAMETERS ───────────────────────────────────
        epochs      = EPOCHS,
        imgsz       = IMAGE_SIZE,
        batch       = BATCH_SIZE,
        device      = DEVICE,

        # ── WORKERS: 2 for faster data loading ───────────────
        # workers=2: 2 parallel processes pre-fetch batches.
        # Safe now because ALL code is inside if __name__ == '__main__'
        # This eliminates the bottleneck of data loading on CPU.
        workers     = 2,

        # ── LEARNING RATE SCHEDULE ────────────────────────────
        # WHY COSINE DECAY:
        #   Starts at lr0, smoothly decays to lrf
        #   Better than step decay for large datasets
        #   Final epochs train with low LR = fine-tuning mode
        optimizer       = 'AdamW',
        lr0             = LR_INITIAL,
        lrf             = 0.1,        # final lr = lr0 * lrf = 0.001
        momentum        = 0.937,
        weight_decay    = 0.0005,
        warmup_epochs   = WARMUP_EPOCHS,
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.1,

        # ── EARLY STOPPING ────────────────────────────────────
        patience = PATIENCE,

        # ── SAVING ────────────────────────────────────────────
        save         = True,
        save_period  = 10,    # save checkpoint every 10 epochs

        # ── AUGMENTATIONS — OV5647 Pi Camera Rev 1.3 Profile ─
        #
        # HARDWARE: Raspberry Pi Camera Rev 1.3 (OV5647 5MP sensor)
        # MOUNTING: Fixed to wand, sweeping horizontally over crops
        # LOCATION: Indian farms — harsh sunlight, dusty conditions
        #
        # ── COLOR AUGMENTATIONS (OV5647-specific) ────────────
        # hsv_h=0.02 ↑ (was 0.015):
        #   OV5647 has hue drift in direct Indian sunlight.
        #   Different times of day (morning yellow vs noon white)
        #   shift hue by up to 15°. Bigger h-range = robust model.
        #
        # hsv_s=0.8 ↑ (was 0.7):
        #   OV5647 auto-exposure drops saturation in bright sun.
        #   Wet monsoon soil (vivid brown) vs dry summer (dusty pale).
        #   Indian farms span both extremes in one season.
        #
        # hsv_v=0.5 ↑ (was 0.4):
        #   Most impactful for Indian farms. OV5647 has no HDR.
        #   Wand passes through: dense crop shadow → open sky gap.
        #   0.5 range teaches model to handle both extremes.
        #
        # ── GEOMETRIC AUGMENTATIONS (Wand motion profile) ────
        # degrees=3.0 ↓ (was 5.0):
        #   Wand is held level by a person. Real rotation < 3°.
        #   Reducing saves compute and avoids unrealistic samples.
        #
        # translate=0.15 ↑ (was 0.1):
        #   Wand moves forward at walking speed while camera captures.
        #   Objects shift horizontally across the frame rapidly.
        #   15% translation = realistic pixel shift per frame.
        #
        # scale=0.5 ↓ (was 0.6):
        #   OV5647 is fixed-focus (~30cm working distance).
        #   In practice wand height stays constant (25–40cm range).
        #   Tighter scale range = more realistic zoom variation.
        #
        # perspective=0.0005:
        #   Wand tilts slightly forward/back as farmer walks.
        #   Perspective warp simulates this subtle keystoning.
        #   Crucial because OV5647 has wide angular FOV (54°).
        #
        # shear=2.0 ↓ (was 3.0):
        #   Wrist rotation during sweep. Stayed moderate.
        #
        # flipud=0.0:
        #   DISABLED (was 0.3). The Pi camera is always looking DOWN.
        #   An upside-down image never occurs in real deployment.
        #   Fake data wastes capacity on impossible orientations.
        #
        # fliplr=0.5:
        #   Sweep goes left sometimes, right sometimes. Kept.
        #
        # erasing=0.3:
        #   Simulates dirt, mud splatter on the OV5647 lens cover.
        #   Random rectangle erased from image = muddy lens patch.
        #   Very common in real field conditions with a wand tool.
        #
        # copy_paste=0.1:
        #   Paste weed instances into background images.
        #   Reduced from 0.4 — expensive; 0.1 is enough at 320px.
        hsv_h        = 0.02,
        hsv_s        = 0.8,
        hsv_v        = 0.5,
        degrees      = 3.0,
        translate    = 0.15,
        scale        = 0.5,
        perspective  = 0.0005,
        shear        = 2.0,
        flipud       = 0.0,    # camera always points DOWN in deployment
        fliplr       = 0.5,
        mosaic       = 1.0,
        erasing      = 0.3,    # simulates OV5647 lens mud/dirt in field
        copy_paste   = 0.1,
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
        # DISABLED: multi_scale=True caused CUDNN_STATUS_EXECUTION_FAILED
        # Root cause: at batch=32, when image size randomly jumps to 800px
        # (from mosaic/multi-scale pipeline), VRAM spikes past 4GB limit.
        # cuDNN silently fails the convolution kernel launch.
        # Fix: keep fixed 640px. Scale augmentation (scale=0.6 above) already
        # teaches the model to handle objects at different zoom levels.
        multi_scale = False,

        # ── NO RAM CACHE ──────────────────────────────────────
        # cache='ram' requires 75GB but system only has 15.7GB.
        # cache='disk' also avoided — SSD wear + complex temp files.
        # Speed gain comes from batch=32 + cudnn.benchmark instead.

        # ── VALIDATION ────────────────────────────────────────
        # 1,600 val images — all 100% labeled after rebuild.
        # mAP50 and precision/recall will be accurate each epoch.
        val   = True,
        plots = True,    # saves loss/lr/mAP curves to runs folder
    )

    # ── POST-TRAINING REPORT ──────────────────────────────────
    elapsed     = time.time() - start_time
    elapsed_min = elapsed / 60

    print("\n" + "="*60)
    print("  FasalAstra v3 — YOLO26n Training Complete!")
    print("="*60)
    print(f"\n  ⏱️  Total time    : {elapsed_min:.1f} minutes")
    print("  📁 Results saved : runs/fasal_astra_rpi4/")
    print("  🏆 Best model    : runs/fasal_astra_rpi4/weights/best.pt")
    print("  💾 Last model    : runs/fasal_astra_rpi4/weights/last.pt")

    # Print final metrics if available
    try:
        metrics   = results.results_dict
        map50     = metrics.get('metrics/mAP50(B)',     0)
        map95     = metrics.get('metrics/mAP50-95(B)',  0)
        precision = metrics.get('metrics/precision(B)', 0)
        recall    = metrics.get('metrics/recall(B)',    0)

        print("\n  📊 Final Model Metrics:")
        print(f"     Precision  : {precision:.2%}")
        print(f"     Recall     : {recall:.2%}")
        print(f"     mAP50      : {map50:.2%}")
        print(f"     mAP50-95   : {map95:.2%}")

        # Verdict
        print("\n  📋 Deployment Verdict:")
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

    print("""
  🚀 NEXT STEPS:
  ─────────────────────────────────────────────────
  1. python 04_evaluate.py   → check mAP on train set
  2. python 05_export.py     → export ONNX for RPi4
     (imgsz=320 ONNX = ~12MB, runs 8-10 FPS on Pi)
  3. Copy to RPi4:
     scp runs/fasal_astra_rpi4/weights/best.pt \\
         pi@raspberrypi.local:~/FasalAstra_AI/runs/
  4. python3 rpi/main_rpi.py → LIVE DEMO 🌾
  ─────────────────────────────────────────────────
""")
