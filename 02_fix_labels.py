# FasalAstra — Label Fixer
#
# PROBLEM: datasets/merged/labels/train/ has only 1,300 label files
#          but datasets/merged/images/train/ has 46,953 images.
#          The 02_merge.py script copied images but NOT their labels.
#
# HOW THIS SCRIPT WORKS:
#   1. Scans every image in datasets/merged/images/train/
#   2. Looks for a matching .txt label file in ALL source dataset folders
#   3. Copies found labels into datasets/merged/labels/train/
#   4. Reports how many labels were rescued
#
# YOLO label format: each .txt file = one image's bounding boxes
#   Each line: <class_id> <x_center> <y_center> <width> <height>
#   All values are 0.0–1.0 (normalised to image dimensions)

import os
import shutil
import glob
import sys

# Force UTF-8 encoding for standard output to avoid UnicodeEncodeError in Windows cmd/PowerShell
if sys.stdout and hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

if __name__ == '__main__':

    MERGED_IMAGES = 'datasets/merged/images/train'
    MERGED_LABELS = 'datasets/merged/labels/train'

    os.makedirs(MERGED_LABELS, exist_ok=True)

    # ── FIND ALL SOURCE DATASET LABEL FOLDERS ─────────────────
    # Walk every subfolder inside datasets/ looking for label dirs.
    # Roboflow datasets use: <dataset>/labels/train/
    # Kaggle datasets use:   <dataset>/labels/ or <dataset>/train/labels/
    SOURCE_ROOTS = [
        'datasets',
    ]

    print("\n" + "="*60)
    print("  FasalAstra — Label Rescue Script")
    print("  Fixing: 1,300 labels → should be 46,953")
    print("="*60)

    # Build a giant lookup: stem → full label path
    # stem = filename without extension e.g. "weed_0042"
    print("\n  🔍 Scanning all source datasets for label files...")
    label_lookup: dict[str, str] = {}

    for root in SOURCE_ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip the merged folder itself to avoid self-copying
            if 'merged' in dirpath:
                continue
            for fname in filenames:
                if fname.endswith('.txt'):
                    stem = os.path.splitext(fname)[0]
                    full_path = os.path.join(dirpath, fname)
                    # If same stem found twice, prefer the one with content
                    if stem not in label_lookup:
                        label_lookup[stem] = full_path
                    else:
                        # Keep whichever file has more content (bigger = richer)
                        existing_size = os.path.getsize(label_lookup[stem])
                        new_size      = os.path.getsize(full_path)
                        if new_size > existing_size:
                            label_lookup[stem] = full_path

    print(f"  ✅ Indexed {len(label_lookup):,} label files from source datasets")

    # ── SCAN ALL MERGED IMAGES ─────────────────────────────────
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [
        f for f in os.listdir(MERGED_IMAGES)
        if os.path.splitext(f)[1].lower() in image_exts
    ]
    print(f"\n  📷 Found {len(image_files):,} images in merged/images/train/")

    # ── COPY MATCHING LABELS ───────────────────────────────────
    rescued   = 0
    already   = 0
    not_found = 0

    for img_fname in image_files:
        stem = os.path.splitext(img_fname)[0]
        dest_label = os.path.join(MERGED_LABELS, stem + '.txt')

        # Skip if label already exists and has content
        if os.path.exists(dest_label) and os.path.getsize(dest_label) > 0:
            already += 1
            continue

        if stem in label_lookup:
            src = label_lookup[stem]
            shutil.copy2(src, dest_label)
            rescued += 1
        else:
            not_found += 1

    # ── REPORT ─────────────────────────────────────────────────
    total_labeled = already + rescued
    coverage      = total_labeled / len(image_files) * 100 if image_files else 0

    print(f"\n  📊 Label Rescue Results:")
    print(f"     Already had labels : {already:,}")
    print(f"     Rescued now        : {rescued:,}")
    print(f"     Still missing      : {not_found:,}  (genuine background images)")
    print(f"     ─────────────────────────────")
    print(f"     Total labeled      : {total_labeled:,} / {len(image_files):,}")
    print(f"     Coverage           : {coverage:.1f}%")

    if coverage > 90:
        print(f"\n  🟢 EXCELLENT — dataset is ready for training!")
    elif coverage > 50:
        print(f"\n  🟡 DECENT — training will work but accuracy limited")
        print(f"     Consider running 01_download.py again for more labeled data")
    else:
        print(f"\n  🔴 LOW COVERAGE — {not_found:,} images have no matching labels")
        print(f"     The source datasets may not have label files at all.")
        print(f"     Check: do your datasets/rf_*/ folders have a labels/ subfolder?")

    # ── DELETE STALE CACHE ─────────────────────────────────────
    # YOLO caches the label scan. After we add new labels, the old
    # cache would still show 1,300 labels. Delete it to force rescan.
    for cache_path in [
        'datasets/merged/labels/train.cache',
        'datasets/merged/labels/val.cache',
    ]:
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"\n  🗑️  Deleted stale cache: {cache_path}")

    print("\n  ✅ Run python 03_train.py now to start training with fixed labels!\n")
