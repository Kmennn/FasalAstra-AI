# ============================================================
#  FasalAstra v3 — Dataset Rebuild & Remapper
#  Properly merges all labeled datasets into one clean set.
#
#  THE PROBLEM (why we're here):
#    - datasets/merged had only 2.8% label coverage
#    - Classification datasets (no bboxes) were included
#    - Each source dataset used DIFFERENT class ID schemes
#
#  THIS SCRIPT:
#    1. ONLY copies images that have matching YOLO label files
#    2. REMAPS every dataset's class IDs to canonical:
#         0 = weed   (fire solenoid)
#         1 = crop   (safety zone, no action)
#         2 = soil   (no action)
#    3. Splits 90% train / 10% val from ALL pairs combined
#    4. Writes a correct data.yaml
# ============================================================

import os
import sys
import shutil
import yaml
import random

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── CANONICAL CLASSES ────────────────────────────────────────
WEED = 0
CROP = 1
SOIL = 2

# ── DATASET CONFIGURATIONS ───────────────────────────────────
DATASET_CONFIGS = [

    # rf_02_weedcrop: nc=2, names=['crop'=0, 'weed'=1]
    {
        'path':    'datasets/rf_02_weedcrop',
        'pattern': 'rf',
        'remap':   {0: CROP, 1: WEED},
        'skip':    set(),
        'name':    'rf02',
    },

    # rf_03_crop_weed: nc=5 — eggplant/pepper/tomato are crops
    {
        'path':    'datasets/rf_03_crop_weed',
        'pattern': 'rf',
        'remap':   {0: CROP, 1: CROP, 2: CROP, 3: CROP, 4: WEED},
        'skip':    set(),
        'name':    'rf03',
    },

    # rf_04_canopy: nc=2, names=['crop'=0, 'weed'=1]
    {
        'path':    'datasets/rf_04_canopy',
        'pattern': 'rf',
        'remap':   {0: CROP, 1: WEED},
        'skip':    set(),
        'name':    'rf04',
    },

    # kg_03_crop_weed_bbox: classes.txt -> 0=crop, 1=weed
    {
        'path':    'datasets/kg_03_crop_weed_bbox',
        'pattern': 'mixed_dir',
        'remap':   {0: CROP, 1: WEED},
        'skip':    set(),
        'name':    'kg03',
    },

    # rf_06_project_weeds: nc=1, names=['Weeds'=0]
    {
        'path':    'datasets/rf_06_project_weeds',
        'pattern': 'rf',
        'remap':   {0: WEED},
        'skip':    set(),
        'name':    'rf06',
    },

    # rf_08_cotton: Cotton=1 is crop, rest are weeds
    {
        'path':    'datasets/rf_08_cotton_recognition',
        'pattern': 'rf',
        'remap':   {0: WEED, 1: CROP, 2: WEED, 3: WEED, 4: WEED, 5: WEED, 6: WEED},
        'skip':    set(),
        'name':    'rf08',
    },

    # rf_09_weed_2025: nc=2, names=['faba'=0, 'weeds'=1]
    {
        'path':    'datasets/rf_09_weed_2025',
        'pattern': 'rf',
        'remap':   {0: CROP, 1: WEED},
        'skip':    set(),
        'name':    'rf09',
    },

    # rf_01_augmented_weeds: uses train/images subfolder directly
    {
        'path':    'datasets/rf_01_augmented_weeds',
        'pattern': 'rf_sub',
        'remap':   {0: WEED, 1: CROP},
        'skip':    set(),
        'name':    'rf01',
    },
]

MERGED_DIR = 'datasets/merged'
VAL_RATIO  = 0.10
SEED       = 42


def remap_label_file(src_path, remap, skip):
    """Read YOLO label and remap class IDs. Returns list of lines or None."""
    lines_out = []
    try:
        with open(src_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                src_cls = int(parts[0])
                if src_cls in skip:
                    continue
                tgt_cls = remap.get(src_cls, WEED)  # unknown classes -> weed
                lines_out.append(f"{tgt_cls} " + " ".join(parts[1:]))
    except Exception as e:
        print(f"    [WARN] Could not read {src_path}: {e}")
        return None
    return lines_out if lines_out else None


def collect_pairs_rf(ds_cfg):
    """Roboflow layout: ds/{train,valid,test}/{images,labels}/"""
    ds_path = ds_cfg['path']
    pairs = []
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(ds_path, split, 'images')
        lbl_dir = os.path.join(ds_path, split, 'labels')
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            stem     = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + '.txt')
            if os.path.exists(lbl_path):
                pairs.append((os.path.join(img_dir, fname), lbl_path))
    return pairs


def collect_pairs_rf_sub(ds_cfg):
    """RF-style but images are directly in ds/train/images/ or ds/images/"""
    ds_path = ds_cfg['path']
    pairs = []
    for split in ['train', 'valid', 'test', '']:
        img_dir = os.path.join(ds_path, split, 'images') if split else os.path.join(ds_path, 'images')
        lbl_dir = os.path.join(ds_path, split, 'labels') if split else os.path.join(ds_path, 'labels')
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            stem     = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + '.txt')
            if os.path.exists(lbl_path):
                pairs.append((os.path.join(img_dir, fname), lbl_path))
        if pairs:
            break  # Stop after first valid split to avoid duplicates
    return pairs


def collect_pairs_mixed_dir(ds_cfg):
    """Images and labels are in the exact same directory (e.g., kg_03)."""
    ds_path = ds_cfg['path']
    pairs = []
    
    # Check if there is a 'data' subfolder, e.g. 'agri_data/data'
    data_dir = os.path.join(ds_path, 'agri_data', 'data')
    if not os.path.exists(data_dir):
        data_dir = ds_path
        
    if not os.path.isdir(data_dir):
        return []
        
    for fname in os.listdir(data_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(data_dir, stem + '.txt')
            if os.path.exists(lbl_path):
                pairs.append((os.path.join(data_dir, fname), lbl_path))
    return pairs


def rebuild():
    random.seed(SEED)

    # ── SETUP DIRECTORIES ───────────────────────────────────
    for sub in ['images', 'labels']:
        for split in ['train', 'val']:
            os.makedirs(os.path.join(MERGED_DIR, sub, split), exist_ok=True)

    print("\n" + "="*60)
    print("  FasalAstra v3 — Dataset Rebuild & Remapper")
    print("="*60)

    # ── DELETE STALE MERGED CONTENT ─────────────────────────
    print("\n  [1/4] Clearing old merged dataset...")
    count_deleted = 0
    for sub in ['images', 'labels']:
        for split in ['train', 'val']:
            d = os.path.join(MERGED_DIR, sub, split)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
                    count_deleted += 1
    # Delete stale caches
    for cache in ['labels/train.cache', 'labels/val.cache']:
        p = os.path.join(MERGED_DIR, cache)
        if os.path.exists(p):
            os.remove(p)
    print(f"     Deleted {count_deleted:,} old files")

    # ── COLLECT ALL PAIRS ────────────────────────────────────
    print("\n  [2/4] Scanning source datasets...\n")
    all_pairs_with_cfg = []  # list of (img_path, lbl_path, ds_cfg)

    collectors = {
        'rf':        collect_pairs_rf,
        'rf_sub':    collect_pairs_rf_sub,
        'mixed_dir': collect_pairs_mixed_dir,
    }

    for cfg in DATASET_CONFIGS:
        collector = collectors.get(cfg['pattern'], collect_pairs_rf)
        pairs = collector(cfg)
        print(f"     {cfg['name']:<8}: {len(pairs):>5,} labeled pairs  [{cfg['path']}]")
        for img_p, lbl_p in pairs:
            all_pairs_with_cfg.append((img_p, lbl_p, cfg))

    print(f"\n     TOTAL: {len(all_pairs_with_cfg):,} image-label pairs")

    if len(all_pairs_with_cfg) == 0:
        print("\n  [ERROR] No labeled pairs found! Check dataset paths.")
        return

    # ── SPLIT TRAIN/VAL ─────────────────────────────────────
    print(f"\n  [3/4] Splitting {int((1-VAL_RATIO)*100)}/{int(VAL_RATIO*100)} train/val...")
    random.shuffle(all_pairs_with_cfg)
    val_count = max(50, int(len(all_pairs_with_cfg) * VAL_RATIO))
    val_set   = all_pairs_with_cfg[:val_count]
    train_set = all_pairs_with_cfg[val_count:]
    print(f"     Train : {len(train_set):,}")
    print(f"     Val   : {len(val_set):,}")

    # ── COPY & REMAP ──────────────────────────────────────────
    print("\n  [4/4] Copying and remapping labels...")
    stats = {'copied': 0, 'skipped_empty': 0}

    # Use a GLOBAL index to guarantee unique filenames across all datasets
    global_idx = 0

    def copy_pair(img_path, lbl_path, cfg, split, gidx):
        ext      = os.path.splitext(img_path)[1].lower()
        basename = f"{gidx:06d}_{cfg['name']}{ext}"
        lbl_name = f"{gidx:06d}_{cfg['name']}.txt"

        dst_img = os.path.join(MERGED_DIR, 'images', split, basename)
        dst_lbl = os.path.join(MERGED_DIR, 'labels', split, lbl_name)

        lines = remap_label_file(lbl_path, cfg['remap'], cfg['skip'])
        if lines is None:
            stats['skipped_empty'] += 1
            return

        shutil.copy2(img_path, dst_img)
        with open(dst_lbl, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        stats['copied'] += 1

    for img_p, lbl_p, cfg in train_set:
        copy_pair(img_p, lbl_p, cfg, 'train', global_idx)
        global_idx += 1

    for img_p, lbl_p, cfg in val_set:
        copy_pair(img_p, lbl_p, cfg, 'val', global_idx)
        global_idx += 1

    # ── WRITE data.yaml ─────────────────────────────────────
    data = {
        'path':  os.path.abspath(MERGED_DIR),
        'train': 'images/train',
        'val':   'images/val',
        'nc':    3,
        'names': {0: 'weed', 1: 'crop', 2: 'soil'},
    }
    with open(os.path.join(MERGED_DIR, 'data.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)

    # ── FINAL REPORT ─────────────────────────────────────────
    train_imgs = len(os.listdir(os.path.join(MERGED_DIR, 'images', 'train')))
    val_imgs   = len(os.listdir(os.path.join(MERGED_DIR, 'images', 'val')))
    train_lbls = len(os.listdir(os.path.join(MERGED_DIR, 'labels', 'train')))
    val_lbls   = len(os.listdir(os.path.join(MERGED_DIR, 'labels', 'val')))

    coverage = train_lbls / max(1, train_imgs) * 100

    print(f"\n  {'='*56}")
    print(f"  REBUILD COMPLETE!")
    print(f"  {'='*56}")
    print(f"     Train images : {train_imgs:,}")
    print(f"     Train labels : {train_lbls:,}")
    print(f"     Val images   : {val_imgs:,}")
    print(f"     Val labels   : {val_lbls:,}")
    print(f"     Coverage     : {coverage:.1f}%")
    print(f"     Skipped      : {stats['skipped_empty']:,} (empty/invalid label files)")
    print()
    print(f"  Classes  : 0=weed | 1=crop | 2=soil")
    print(f"  YAML     : datasets/merged/data.yaml")
    print()

    if coverage >= 99.0:
        print("  STATUS: PERFECT — 100% label coverage. Ready for training!")
    elif coverage >= 95.0:
        print("  STATUS: EXCELLENT — dataset is ready for training.")
    else:
        print(f"  STATUS: WARNING — {100-coverage:.1f}% images have no labels.")
        print("          This usually means some source label files were missing.")

    print()
    print("  Next step: python 03_train.py")
    print()


if __name__ == '__main__':
    rebuild()
