import os
import sys
import yaml

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

labeled_datasets = [
    ('datasets/rf_02_weedcrop',           'rf_02'),
    ('datasets/rf_03_crop_weed',           'rf_03'),
    ('datasets/rf_04_canopy',              'rf_04'),
    ('datasets/rf_05_mixed',               'rf_05'),
    ('datasets/rf_06_project_weeds',       'rf_06'),
    ('datasets/rf_08_cotton_recognition',  'rf_08'),
    ('datasets/rf_09_weed_2025',           'rf_09'),
    ('datasets/kg_03_crop_weed_bbox',      'kg_03'),
]

for ds_path, name in labeled_datasets:
    yaml_path = os.path.join(ds_path, 'data.yaml')
    if os.path.exists(yaml_path):
        with open(yaml_path, encoding='utf-8') as f:
            d = yaml.safe_load(f)
        print(f"{name}: nc={d.get('nc', '?')} names={d.get('names', d.get('labels', '?'))}")
    else:
        cls_path = os.path.join(ds_path, 'classes.txt')
        if os.path.exists(cls_path):
            with open(cls_path, encoding='utf-8') as f:
                classes = f.read().strip()
            print(f"{name}: classes.txt -> {classes}")
        else:
            print(f"{name}: no yaml/classes.txt found")

    # Also sample a label file to see what class IDs are used
    label_root = None
    for split in ['train', 'valid', 'val']:
        p = os.path.join(ds_path, split, 'labels')
        if not os.path.exists(p):
            p = os.path.join(ds_path, 'labels', split)
        if os.path.exists(p):
            label_root = p
            break
    if not label_root:
        # Try flat labels
        p = os.path.join(ds_path, 'labels')
        if os.path.exists(p):
            label_root = p

    if label_root:
        class_ids = set()
        for fname in os.listdir(label_root)[:50]:
            if not fname.endswith('.txt'):
                continue
            with open(os.path.join(label_root, fname), encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            class_ids.add(int(parts[0]))
                        except ValueError:
                            pass
        print(f"  -> Unique class IDs in labels: {sorted(class_ids)}")
    print()
