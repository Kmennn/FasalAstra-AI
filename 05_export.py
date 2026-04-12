# UPDATED FOR RPi4:
# Export to ONNX instead of TFLite
# WHY ONNX for RPi4:
#   RPi4 has full ARM Cortex-A72 CPU
#   ONNX Runtime is optimized for ARM
#   Faster than TFLite on RPi4
#   No quantization loss — maintains 84.9% mAP
#
# We still keep TFLite export option
# for future ESP32 final product

from ultralytics import YOLO
import os
import shutil

MODEL_PATH = 'runs/detect/runs/fasal_astra_rpi4/weights/best.pt'
OUTPUT_DIR = 'models/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    model = YOLO(MODEL_PATH)

    print("=" * 55)
    print("  FasalAstra — Exporting for Raspberry Pi 4")
    print("=" * 55)

    # PRIMARY: ONNX for RPi4
    print("\n[1/2] Exporting ONNX (for RPi4)...")
    model.export(
        format='onnx',
        imgsz=320,
        simplify=True,      # WHY: reduces model graph complexity
        opset=12            # WHY: RPi4 ONNX Runtime supports opset 12
    )

    # SECONDARY: TFLite INT8 (for future ESP32 product)
    print("\n[2/2] Exporting TFLite INT8 (for future ESP32)...")
    model.export(
        format='tflite',
        int8=True,
        imgsz=320,
        data='datasets/merged/data.yaml'
    )

    # Copy to models folder
    for fname in os.listdir('runs/detect/runs/fasal_astra_rpi4/weights/'):
        if fname.endswith('.onnx') or fname.endswith('.tflite'):
            shutil.copy(
                f'runs/detect/runs/fasal_astra_rpi4/weights/{fname}',
                f'{OUTPUT_DIR}/{fname}'
            )

    print("\n✅ Export complete!")
    print("   models/best.onnx     → use on Raspberry Pi 4")
    print("   models/best.tflite   → future ESP32 product")
except Exception as e:
    print(f"Error during export: {e}")
    print("Ensure you have trained the model using 03_train.py first!")
