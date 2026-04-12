# FasalAstra v3 — MAXIMUM DATASET COLLECTOR
# Strategy: Real datasets + Smart augmentation = 1 lakh images
#
# SOURCES:
# ┌─────────────────────────────────────────────────┐
# │ SOURCE 1: Roboflow  (~15,000 labeled images)   │
# │ SOURCE 2: Kaggle    (~27,000 labeled images)   │
# │ SOURCE 3: GitHub    (~5,000  labeled images)   │
# │ SOURCE 4: MH-Weed16 ~25,972 (manual download) │
# │ TOTAL REAL          ~70,000 labeled images     │
# │ After augmentation  ~1,00,000+ effective imgs  │
# └─────────────────────────────────────────────────┘

from roboflow import Roboflow
import os, time, shutil, requests, zipfile
import subprocess, sys

# ── API KEYS ─────────────────────────────────────────────
API_KEY        = "wE02Bi4maLDvZMWlishj"        # Roboflow
KAGGLE_TOKEN   = "KGAT_ba692935d95b459536f8d91d671720f0"  # Kaggle
# ─────────────────────────────────────────────────────────

# Auto-configure Kaggle environment variable so subprocess picks it up
os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN

rf      = Roboflow(api_key=API_KEY)
results = []
failed  = []


def count_images(folder):
    count = 0
    exts  = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    if not os.path.exists(folder):
        return 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                count += 1
    return count


def download_roboflow(workspace, project, version,
                      location, name, expected):
    print(f"\n{'─'*55}")
    print(f"  [{name}]")
    print(f"  Expected: ~{expected} images")
    try:
        rf.workspace(workspace) \
          .project(project) \
          .version(version) \
          .download("yolov8", location=location)
        count = count_images(location)
        results.append({'name': name, 'path': location,
                        'images': count, 'source': 'roboflow'})
        print(f"  ✅ {count} images downloaded")
        return True
    except Exception as e:
        failed.append({'name': name, 'error': str(e)})
        print(f"  ❌ FAILED: {str(e)[:100]}")
        return False


def download_kaggle(dataset_slug, location, name, expected):
    """
    Download from Kaggle using kaggle CLI 2.0+.
    Uses KAGGLE_API_TOKEN env var (new bearer token format KGAT_...)
    which is supported by kaggle CLI 2.0 without needing kaggle.json.
    """
    print(f"\n{'─'*55}")
    print(f"  [{name}] — Kaggle")
    print(f"  Expected: ~{expected} images")
    os.makedirs(location, exist_ok=True)

    # Build env with token injected — kaggle 2.0 reads this env var
    kaggle_env = os.environ.copy()
    kaggle_env["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN

    # Find kaggle executable next to current python (works in venv)
    python_dir = os.path.dirname(sys.executable)
    kaggle_exe = os.path.join(python_dir, "kaggle.exe")
    if not os.path.exists(kaggle_exe):
        kaggle_exe = os.path.join(python_dir, "kaggle")   # Linux/Mac
    if not os.path.exists(kaggle_exe):
        kaggle_exe = "kaggle"                              # fallback PATH

    try:
        result = subprocess.run(
            [kaggle_exe, 'datasets', 'download',
             '-d', dataset_slug,
             '-p', location,
             '--unzip'],
            capture_output=True, text=True,
            timeout=1800, env=kaggle_env
        )

        if result.returncode == 0:
            count = count_images(location)
            results.append({'name': name, 'path': location,
                            'images': count, 'source': 'kaggle'})
            print(f"  ✅ {count} images downloaded")
            return True
        else:
            raise Exception(result.stderr or result.stdout)
    except Exception as e:
        failed.append({'name': name, 'error': str(e)})
        print(f"  ❌ FAILED: {str(e)[:120]}")
        return False


def download_direct(url, location, name, expected):
    """Download direct zip from GitHub/research sources."""
    print(f"\n{'─'*55}")
    print(f"  [{name}] — Direct Download")
    print(f"  Expected: ~{expected} images")
    os.makedirs(location, exist_ok=True)
    zip_path = f"{location}/data.zip"
    try:
        print(f"  Downloading from: {url[:60]}...")
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        total      = int(resp.headers.get('content-length', 0))
        downloaded = 0
        with open(zip_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}%", end='', flush=True)
        print()
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(location)
        os.remove(zip_path)
        count = count_images(location)
        results.append({'name': name, 'path': location,
                        'images': count, 'source': 'direct'})
        print(f"  ✅ {count} images downloaded")
        return True
    except Exception as e:
        failed.append({'name': name, 'error': str(e)})
        print(f"  ❌ FAILED: {str(e)[:100]}")
        return False


# ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  FasalAstra v3 — MAXIMUM Dataset Collector")
print("  Target: 50,000+ real labeled images")
print("  After augmentation: 1,00,000+ effective images")
print("="*55)

# ══════════════════════════════════════════════════════════
# BLOCK A — ROBOFLOW DATASETS (~15,000 images)
# All workspace/project IDs verified working
# ══════════════════════════════════════════════════════════
print("\n\n🟦 BLOCK A — ROBOFLOW (~15,000 images)")

download_roboflow(
    "augmented-startups", "weeds-nxe1w", 1,
    "datasets/rf_01_augmented_weeds",
    "Augmented Startups Weeds — Complex BG", 4200)
time.sleep(2)

download_roboflow(
    "new-workspace-csmgu", "weedcrop-waifl", 1,
    "datasets/rf_02_weedcrop",
    "WeedCrop — 6 Crops 8 Weed Species", 1118)
time.sleep(2)

download_roboflow(
    "orbibarobotics", "crop-and-weed-datase", 1,
    "datasets/rf_03_crop_weed",
    "Orbibarobotics Crop and Weed", 2436)
time.sleep(2)

download_roboflow(
    "new-workspace-pysgt", "crop-and-weeds-detection", 1,
    "datasets/rf_04_canopy",
    "Crop Weeds Dense Canopy", 1300)
time.sleep(2)

download_roboflow(
    "sanghyun-ryu", "weed-and-crop", 1,
    "datasets/rf_05_mixed",
    "Weed Crop Mixed Conditions", 1300)
time.sleep(2)

download_roboflow(
    "project-weeds", "weed-detection-5jm0z", 1,
    "datasets/rf_06_project_weeds",
    "Project Weeds General", 254)
time.sleep(2)

download_roboflow(
    "project-cvdsb", "cotton_weed_recognition", 9,
    "datasets/rf_08_cotton_recognition",
    "Cotton Weed Recognition 2569 imgs", 2569)
time.sleep(2)

download_roboflow(
    "weed-detection-lxcu2", "weed-detection-e62qr", 1,
    "datasets/rf_09_weed_2025",
    "Weed Detection 2025 Latest", 800)
time.sleep(2)

# ══════════════════════════════════════════════════════════
# BLOCK B — KAGGLE DATASETS (~27,000 images)
# WHY KAGGLE: Largest single weed datasets live here
# Token configured automatically via KAGGLE_API_TOKEN env var
# ══════════════════════════════════════════════════════════
print("\n\n🟩 BLOCK B — KAGGLE (~27,000 images)")
print("   WHY: DeepWeeds alone = 17,509 images")

# Kaggle DS1 — DeepWeeds (17,509 images — BIGGEST SINGLE DATASET)
# WHY: Research-grade labeled dataset
# Has Parthenium weed = most common Indian invasive weed
download_kaggle(
    "imsparsh/deepweeds",
    "datasets/kg_01_deepweeds",
    "DeepWeeds — 17,509 Research Grade Images",
    17509)

# Kaggle DS2 — Weed Detection (general)
download_kaggle(
    "jaidalmotra/weed-detection",
    "datasets/kg_02_weed_detection",
    "Weed Detection Kaggle General",
    5000)

# Kaggle DS3 — Crop and Weed with Bounding Boxes
# WHY: Already in YOLO format with bounding boxes
# 1300 sesame crop + weed images from Indian fields
download_kaggle(
    "ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes",
    "datasets/kg_03_crop_weed_bbox",
    "Crop Weed Detection BBox — Indian Sesame Fields",
    1300)

# Kaggle DS4 — Plant Seedlings (V2)
# WHY: 5500 images of plant seedlings including weeds
# Teaches model: baby weed detection = catches early stage
download_kaggle(
    "vbookshelf/v2-plant-seedlings-dataset",
    "datasets/kg_04_seedlings",
    "Plant Seedlings V2 — Early Stage Weeds",
    5539)

# ══════════════════════════════════════════════════════════
# BLOCK C — DIRECT GITHUB DOWNLOADS (~5,000 images)
# Research datasets not on Roboflow or Kaggle
# ══════════════════════════════════════════════════════════
print("\n\n🟨 BLOCK C — GITHUB RESEARCH DATASETS (~5,000)")

# Direct DS1 — CWFID (Crop/Weed Field Image Dataset)
# WHY: Classic academic benchmark dataset
# Used in 100+ research papers — very well labeled
download_direct(
    "https://github.com/cwfid/dataset/archive/refs/heads/master.zip",
    "datasets/gh_01_cwfid",
    "CWFID — Classic Academic Crop Weed Dataset",
    60)

# Direct DS2 — Crop Weed Detection (ravirajsinh45 GitHub)
download_direct(
    "https://github.com/ravirajsinh45/Crop_and_weed_detection/archive/refs/heads/master.zip",
    "datasets/gh_02_crop_weed_github",
    "Crop Weed Detection GitHub",
    1300)

# ══════════════════════════════════════════════════════════
# BLOCK D — MH-WEED16 INDIAN DATASET (25,972 images)
# MOST IMPORTANT FOR OUR PROJECT
# Maharashtra region weeds — EXACT target area
# ══════════════════════════════════════════════════════════
print("\n\n🟥 BLOCK D — MH-WEED16 MAHARASHTRA DATASET")
print("   25,972 images from Maharashtra fields!")
print("   16 weed species from Maharashtra region")
print("   This is EXACTLY our target geography")
print()

download_kaggle(
    "sayalis069/mh-weed16",
    "datasets/mh_weed16",
    "MH-Weed16 Maharashtra Dataset",
    25972)

# ══════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════
print("\n\n" + "="*55)
print("  FasalAstra v3 — Download Complete")
print("="*55)

grand_total = 0
print("\n  ✅ DOWNLOADED SUCCESSFULLY:")
for r in results:
    print(f"     {r['source'].upper():10s} | "
          f"{r['images']:6,} imgs | {r['name']}")
    grand_total += r['images']

if failed:
    print(f"\n  ❌ FAILED ({len(failed)}):")
    for f in failed:
        print(f"     ✗ {f['name']}")

print(f"""
{'─'*55}
  DOWNLOADED NOW     : {grand_total:,} images
  TOTAL POTENTIAL    : ~{grand_total:,} images
{'─'*55}

  📈 AUGMENTATION PLAN:
     Real images         : ~{grand_total:,}
     After 3x augment    : ~{grand_total*3:,}
     After 5x augment    : ~{grand_total*5:,}

     Target 1,00,000 ✅ achievable!

  🚀 NEXT STEPS:
     1. python 02_merge.py
     2. python 03_train.py
""")
