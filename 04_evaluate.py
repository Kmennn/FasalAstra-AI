# PURPOSE: Check if model is good enough for ESP32 deployment
# WHY EVALUATE BEFORE EXPORTING:
# Bad model on ESP32 = sprays crops = farmer loses money
# We must verify precision/recall before deployment

import os
import sys
from ultralytics import YOLO

if __name__ == '__main__':
    MODEL_PATH = 'runs/detect/runs/fasal_astra_rpi4/weights/best.pt'
    YAML_PATH  = 'datasets/merged/data.yaml'

    # Pre-flight checks
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("   Run 03_train.py first!")
        sys.exit(1)

    if not os.path.exists(YAML_PATH):
        print(f"❌ Dataset config not found: {YAML_PATH}")
        print("   Run 02_merge.py first!")
        sys.exit(1)

    model   = YOLO(MODEL_PATH)
    metrics = model.val(data=YAML_PATH)

    p     = metrics.box.mp
    r     = metrics.box.mr
    map50 = metrics.box.map50
    map95 = metrics.box.map

    print("\n" + "=" * 50)
    print("FasalAstra — Model Evaluation Report")
    print("=" * 50)

    print("\n📊 Overall Scores:")
    print(f"   Precision  : {p:.2%}  (fires only on real weeds)")
    print(f"   Recall     : {r:.2%}  (catches all weeds)")
    print(f"   mAP50      : {map50:.2%}  (main accuracy score)")
    print(f"   mAP50-95   : {map95:.2%}  (strict accuracy score)")

    print("\n🌾 Per-Class Performance:")
    per_class_results = {}
    for i, name in metrics.names.items():
        class_map50 = metrics.box.maps[i]
        per_class_results[name] = class_map50
        print(f"   {name:12s} -> mAP50: {class_map50:.2%}")

    print("\n📋 Verdict for ESP32 Deployment:")
    if map50 > 0.85:
        verdict = "🟢 EXCELLENT — Deploy with confidence"
    elif map50 > 0.75:
        verdict = "🟡 GOOD      — Safe for hackathon demo"
    elif map50 > 0.60:
        verdict = "🟠 OKAY      — Add Indian farm images"
    else:
        verdict = "🔴 NEEDS WORK — Check dataset labels"
    print(f"   {verdict}")

    # WHY THIS MATTERS FOR JUDGES:
    # Crop precision must be HIGH
    # If crop precision low = model sprays crops = project fails
    print("\n⚠️  Critical Safety Checks:")

    # Check weed class
    weed_map = per_class_results.get('weed')
    if weed_map is not None:
        print(f"   Weed class mAP50 : {weed_map:.2%}")
        if weed_map > 0.80:
            print("   ✅ Weed detection is reliable")
        else:
            print("   ⚠️  Weed detection could be improved")
    else:
        print("   ⚠️  Weed class not found in results")

    # Check crop class
    crop_map = per_class_results.get('crop')
    if crop_map is not None:
        print(f"   Crop class mAP50 : {crop_map:.2%}")
        if crop_map > 0.80:
            print("   ✅ Crops are safe from false spraying")
        else:
            print("   ❌ WARNING: Risk of spraying crops!")
    else:
        print("   ⚠️  Crop class not found in results")

    # Save results to file for future reference
    results_file = 'evaluation_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("FasalAstra — Model Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model    : {MODEL_PATH}\n")
        f.write(f"Dataset  : {YAML_PATH}\n")
        f.write(f"Precision: {p:.4f}\n")
        f.write(f"Recall   : {r:.4f}\n")
        f.write(f"mAP50    : {map50:.4f}\n")
        f.write(f"mAP50-95 : {map95:.4f}\n")
        f.write(f"Verdict  : {verdict}\n")
        for name, val in per_class_results.items():
            f.write(f"{name}_mAP50: {val:.4f}\n")

    print(f"\n💾 Results saved to: {results_file}")
    print("=" * 50)
