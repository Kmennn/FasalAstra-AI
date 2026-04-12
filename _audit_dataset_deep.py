"""Deep audit of all source datasets for FasalAstra."""
import os, sys, glob, yaml

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE = 'datasets'

def audit_rf_dataset(name, path):
    """Audit a Roboflow-format dataset."""
    total_imgs = 0
    total_lbls = 0
    class_counts = {}
    bbox_counts = 0
    
    # Check data.yaml
    yaml_path = os.path.join(path, 'data.yaml')
    class_names = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            nc = data.get('nc', '?')
            names = data.get('names', {})
            class_names = names
    else:
        nc = '?'
        names = {}
    
    for split in ['train', 'valid', 'test', '']:
        img_dir = os.path.join(path, split, 'images') if split else os.path.join(path, 'images')
        lbl_dir = os.path.join(path, split, 'labels') if split else os.path.join(path, 'labels')
        
        if not os.path.isdir(img_dir):
            continue
        
        imgs = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}]
        total_imgs += len(imgs)
        
        if os.path.isdir(lbl_dir):
            for f in imgs:
                stem = os.path.splitext(f)[0]
                lbl_path = os.path.join(lbl_dir, stem + '.txt')
                if os.path.exists(lbl_path):
                    total_lbls += 1
                    content = open(lbl_path, 'r').read().strip()
                    if content:
                        for line in content.split('\n'):
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls = int(parts[0])
                                class_counts[cls] = class_counts.get(cls, 0) + 1
                                bbox_counts += 1
    
    coverage = (total_lbls / total_imgs * 100) if total_imgs > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"  {name} -> {path}")
    print(f"  Images: {total_imgs:,}  |  Labels: {total_lbls:,}  |  Coverage: {coverage:.1f}%")
    print(f"  Total bboxes: {bbox_counts:,}")
    print(f"  data.yaml: nc={nc}, names={names}")
    print(f"  Class distribution (ORIGINAL IDs before remap):")
    for cls_id in sorted(class_counts.keys()):
        pct = class_counts[cls_id] / bbox_counts * 100 if bbox_counts > 0 else 0
        cname = class_names.get(cls_id, f'class_{cls_id}') if isinstance(class_names, dict) else (class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}')
        print(f"    {cls_id} ({cname}): {class_counts[cls_id]:,} ({pct:.1f}%)")

# Audit all datasets
datasets = [
    ('rf_01_augmented_weeds', 'datasets/rf_01_augmented_weeds'),
    ('rf_02_weedcrop', 'datasets/rf_02_weedcrop'),
    ('rf_03_crop_weed', 'datasets/rf_03_crop_weed'),
    ('rf_04_canopy', 'datasets/rf_04_canopy'),
    ('rf_05_mixed', 'datasets/rf_05_mixed'),
    ('rf_06_project_weeds', 'datasets/rf_06_project_weeds'),
    ('rf_08_cotton_recognition', 'datasets/rf_08_cotton_recognition'),
    ('rf_09_weed_2025', 'datasets/rf_09_weed_2025'),
]

# Also check unused datasets
unused = [
    ('gh_01_cwfid', 'datasets/gh_01_cwfid'),
    ('gh_02_crop_weed_github', 'datasets/gh_02_crop_weed_github'),
    ('kg_01_deepweeds', 'datasets/kg_01_deepweeds'),
    ('kg_02_weed_detection', 'datasets/kg_02_weed_detection'),
    ('kg_03_crop_weed_bbox', 'datasets/kg_03_crop_weed_bbox'),
    ('kg_04_seedlings', 'datasets/kg_04_seedlings'),
    ('mh_weed16', 'datasets/mh_weed16'),
]

print("="*60)
print("  CURRENTLY USED IN TRAINING (via 02_rebuild_merge.py)")
print("="*60)
for name, path in datasets:
    if os.path.isdir(path):
        audit_rf_dataset(name, path)
    else:
        print(f"\n  {name}: DIRECTORY NOT FOUND!")

print("\n\n" + "="*60)
print("  NOT USED IN TRAINING (excluded datasets)")
print("="*60)
for name, path in unused:
    if os.path.isdir(path):
        audit_rf_dataset(name, path)
    else:
        print(f"\n  {name}: DIRECTORY NOT FOUND!")
