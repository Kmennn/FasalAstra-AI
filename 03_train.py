# UPDATED FOR RPi4:
# imgsz=640 instead of 320
# WHY: RPi4 has enough power to run 640x640 inference
# BENEFIT: Better detection of small weeds under canopy
# Pi Camera OV5647 captures at higher resolution natively

from ultralytics import YOLO
import torch
import os

print("=" * 55)
print("  FasalAstra v3 — RPi4 + Pi Camera Training")
print("=" * 55)
print(f"\n  GPU  : {torch.cuda.get_device_name(0)}")
print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n")

YAML_PATH = 'datasets/merged/data.yaml'

if not os.path.exists(YAML_PATH):
    print("❌ Run 02_merge.py first!")
    exit()

model = YOLO('yolov8n.pt')

results = model.train(
    data=YAML_PATH,

    epochs=50,
    imgsz=640,         # UPGRADED: was 320, now 640 for RPi4
    batch=16,          # reduced batch because 640 uses more VRAM
    device=0,
    save=True,
    save_period=10,
    project='runs',
    name='fasal_astra_v3_rpi',
    patience=10,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,

    # AUGMENTATIONS — same farm conditions
    flipud=0.3,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    mosaic=1.0,
    copy_paste=0.3,

    box=7.5,
    cls=0.5,
    dfl=1.5,
)

print("\n✅ Training complete!")
print("   Best model: runs/fasal_astra_v3_rpi/weights/best.pt")
print("   This model runs directly on RPi4 — no TFLite needed!")
