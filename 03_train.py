# UPDATED: Better augmentations for Indian farm conditions
# NEW: mosaic + copy_paste for dense canopy simulation
# NEW: Explicit IMU-aware augmentations for Predictive Kinematic Pipeline

from ultralytics import YOLO
import torch
import os

if __name__ == '__main__':
    print("=" * 55)
    print("  FasalAstra — Predictive Kinematic Pipeline Training")
    print("=" * 55)
    print(f"\n  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  CUDA  : {torch.version.cuda}\n")

    YAML_PATH = 'datasets/merged/data.yaml'

    if not os.path.exists(YAML_PATH):
        print("❌ Run 02_merge.py first!")
        exit()

    model = YOLO('yolov8n.pt')

    results = model.train(
        data=YAML_PATH,

        # CORE
        epochs=50,
        imgsz=320,
        batch=32,
        device=0,
        save=True,
        save_period=10,
        project='runs',
        name='fasal_astra_v2',
        patience=10,
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,

        # AUGMENTATIONS — Each one maps to a real farm condition
        flipud=0.3,       # wand tilt up/down
        fliplr=0.5,       # left/right sweep
        hsv_h=0.015,      # weed color variation
        hsv_s=0.7,        # dry vs wet soil
        hsv_v=0.4,        # sun vs canopy shadow
        translate=0.1,    # wand movement blur
        scale=0.5,        # weed size variation
        mosaic=1.0,       # simulates crowded canopy
        copy_paste=0.3,   # NEW: paste weeds into crop images
                          # WHY: teaches model weeds INSIDE canopy

        # LOSS WEIGHTS
        # WHY box weight higher: We need precise centroids
        # for kinematic calculation — wrong centroid = missed spray
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    print("\n✅ Training complete!")
    print("   Best model: runs/fasal_astra_v2/weights/best.pt")
