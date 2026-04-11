# PURPOSE: Pull all 4 weed datasets from Roboflow cloud
# WHY ROBOFLOW: Largest free agricultural AI dataset hub
# WHY THESE 4 DATASETS:
#   ds1 → General weed detection (core foundation)
#   ds2 → Crop AND weed labeled (teaches crop safety)
#   ds3 → Dense canopy weeds (fixes our occlusion problem)
#   ds4 → Paddy/rice weeds (Indian farming context)

from roboflow import Roboflow
import os

API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()

if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY is not set. Set it in terminal, then rerun this script."
    )

rf = Roboflow(api_key=API_KEY)

DATASETS = [
  ("krishibot-mwqke", "weed-detection-txujb", 1, "datasets/ds1", "General Weed Dataset"),
  ("loki-orzgg", "weed-detection-de08c", 1, "datasets/ds2", "Dense Weed Dataset"),
  ("testws-32zuu", "weed-detection-rlbal", 1, "datasets/ds3", "Indian Field Weed Dataset"),
  ("paddy-weed", "paddy-weed-detection", 1, "datasets/ds4", "Paddy / Rice Weed Dataset"),
]

print("=" * 50)
print("FasalAstra — Downloading Datasets from Cloud")
print("=" * 50)

ok = 0
for i, (workspace, project, version, out_dir, title) in enumerate(DATASETS, start=1):
    print(f"\n[{i}/4] {title}...")
    try:
        rf.workspace(workspace) \
            .project(project) \
            .version(version) \
            .download("yolov8", location=out_dir)
        ok += 1
    except Exception as e:
        print(f"   Skipped {workspace}/{project}: {e}")

if ok == 0:
    raise RuntimeError("No datasets were downloaded. Check API key or dataset availability.")

print(f"\n✅ Downloaded {ok}/4 datasets to /datasets folder")
