# PURPOSE: Convert trained model to ESP32-S3 compatible format
# WHY TFLite INT8:
#   PyTorch model (.pt) = 6MB  → ESP32 cannot run
#   TFLite INT8 (.tflite) = ~1.5MB → fits perfectly
# WHY INT8 QUANTIZATION:
#   Converts 32-bit floats to 8-bit integers
#   4x smaller size
#   2-4x faster on ESP32 hardware accelerator
#   Only ~2-3% accuracy drop = acceptable

from ultralytics import YOLO
import shutil
import os

if __name__ == '__main__':
    MODEL_PATH  = 'runs/detect/runs/fasal_astra_v23/weights/best.pt'
    YAML_PATH   = 'datasets/merged/data.yaml'
    OUTPUT_DIR  = 'models/'

    model = YOLO(MODEL_PATH)

    print("=" * 50)
    print("FasalAstra — Exporting for ESP32-S3")
    print("=" * 50)

    print("\n🔄 Converting to TFLite INT8...")

    model.export(
        format='tflite',
        int8=True,
        imgsz=320,
        data=YAML_PATH
    )

    # Copy to models folder
    tflite_src = 'runs/detect/runs/fasal_astra_v23/weights/best_saved_model'
    tflite_dst = 'models/fasal_astra_esp32.tflite'

    found_tflite = False
    if os.path.exists(tflite_src):
        for f in os.listdir(tflite_src):
            if f.endswith('.tflite'):
                shutil.copy(
                    f'{tflite_src}/{f}',
                    tflite_dst
                )
                found_tflite = True
                break

    if not found_tflite:
        print("\n❌ TFLite file not found in export output!")
        print(f"   Checked: {tflite_src}")
        print("   Try running export again or check Ultralytics output path.")
        exit(1)

    size_mb = os.path.getsize(tflite_dst) / 1e6

    print(f"\n✅ Export complete!")
    print(f"   File     : {tflite_dst}")
    print(f"   Size     : {size_mb:.2f} MB")
    print(f"   Classes  : weed / crop (2 classes)")
    print(f"   Format   : TFLite INT8")
    print(f"\n🚀 Flash this file to your ESP32-S3!")
    print(f"   Next step: ESP32 Arduino firmware code")
