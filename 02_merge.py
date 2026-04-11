# PURPOSE: Combine all 4 datasets into one clean merged dataset
# WHY MERGE: More variety = model handles more weed types
# WHY RENAME FILES: Prevents filename conflicts between datasets
# WHY 3 CLASSES ONLY:
#   weed (0) → fire solenoid
#   crop (1) → blocked by safety zone
#   soil (2) → no action

import os
import shutil
import yaml

DATASETS   = ['datasets/ds1', 'datasets/ds2',
              'datasets/ds3', 'datasets/ds4']
MERGED_DIR = 'datasets/merged'

def infer_num_classes():
    max_class_id = -1
    for split in ['train', 'val']:
        lbl_dir = f'{MERGED_DIR}/labels/{split}'
        if not os.path.exists(lbl_dir):
            continue
        for fname in os.listdir(lbl_dir):
            if not fname.endswith('.txt'):
                continue
            fpath = f'{lbl_dir}/{fname}'
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        continue
                    if cid > max_class_id:
                        max_class_id = cid
    return max_class_id + 1 if max_class_id >= 0 else 1

def create_folders():
    for split in ['train', 'val']:
        os.makedirs(f'{MERGED_DIR}/images/{split}', exist_ok=True)
        os.makedirs(f'{MERGED_DIR}/labels/{split}', exist_ok=True)
    print("✅ Merged folder structure created")

def merge_all():
    total = 0
    for ds in DATASETS:
        ds_name = ds.split('/')[-1]
        print(f"\nMerging {ds_name}...")
        count = 0

        for split in ['train', 'val']:
            split_candidates = [split]
            if split == 'val':
                split_candidates.append('valid')

            img_dir = None
            lbl_dir = None
            for s in split_candidates:
                # Pattern A: dataset/images/train and dataset/labels/train
                p_img_a = f'{ds}/images/{s}'
                p_lbl_a = f'{ds}/labels/{s}'
                # Pattern B: dataset/train/images and dataset/train/labels
                p_img_b = f'{ds}/{s}/images'
                p_lbl_b = f'{ds}/{s}/labels'

                if os.path.exists(p_img_a):
                    img_dir = p_img_a
                    lbl_dir = p_lbl_a
                    break
                if os.path.exists(p_img_b):
                    img_dir = p_img_b
                    lbl_dir = p_lbl_b
                    break

            if not img_dir or not os.path.exists(img_dir):
                continue

            for idx, fname in enumerate(os.listdir(img_dir)):
                # WHY UNIQUE NAME: Avoids overwriting same-named files
                unique = f'{ds_name}_{idx}_{fname}'

                shutil.copy(
                    f'{img_dir}/{fname}',
                    f'{MERGED_DIR}/images/{split}/{unique}'
                )

                label_fname = os.path.splitext(fname)[0] + '.txt'
                label_path  = f'{lbl_dir}/{label_fname}'
                unique_lbl  = f'{ds_name}_{idx}_{label_fname}'

                if os.path.exists(label_path):
                    shutil.copy(
                        label_path,
                        f'{MERGED_DIR}/labels/{split}/{unique_lbl}'
                    )
                count += 1

        print(f"   {ds_name}: {count} images merged")
        total += count

    print(f"\n✅ Total merged: {total} images")

def create_yaml():
    # WHY THIS YAML:
    # YOLOv8 reads this file to know:
    # - Where is the data
    # - How many classes
    # - What are the class names
    nc = infer_num_classes()
    data = {
        'path': os.path.abspath(MERGED_DIR),
        'train': 'images/train',
        'val':   'images/val',
        'nc': nc,
        'names': {i: f'class_{i}' for i in range(nc)}
    }
    with open(f'{MERGED_DIR}/data.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print("✅ data.yaml created")

    train = len(os.listdir(f'{MERGED_DIR}/images/train'))
    val   = len(os.listdir(f'{MERGED_DIR}/images/val'))
    print(f"\n📊 Final Dataset Summary:")
    print(f"   Training   : {train} images")
    print(f"   Validation : {val} images")
    print(f"   Classes    : {nc} classes (auto-detected)")

if __name__ == '__main__':
    create_folders()
    merge_all()
    create_yaml()
