# PURPOSE: Runs YOLOv8 inference + extracts centroids
# WHY CENTROIDS:
#   We don't need the full bounding box on ESP32
#   Just the center point (cx, cy) is enough
#   for homography mapping and kinematic calculation

from ultralytics import YOLO

CONF_THRESHOLD = 0.75   # only fire if 75%+ confident
                        # WHY 75%: Low confidence = risky
                        # Better to miss a weed than spray a crop

WEED_CLASS = 0
CROP_CLASS = 1

# Class name mapping (must match data.yaml)
CLASS_NAMES = {0: 'weed', 1: 'crop'}

class FasalAstraDetector:
    def __init__(self, model_path='runs/detect/runs/fasal_astra_v23/weights/best.pt'):
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("✅ Model loaded")

    def detect(self, frame):
        """
        Run inference on a single frame
        Returns list of detections with centroids
        """
        results = self.model(
            frame,
            imgsz=320,
            conf=CONF_THRESHOLD,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Calculate centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append({
                'class_id'  : class_id,
                'class_name': CLASS_NAMES.get(class_id, f'unknown_{class_id}'),
                'confidence': confidence,
                'box'       : [x1, y1, x2, y2],
                'centroid'  : (cx, cy)
            })

        return detections

    def get_weeds_only(self, detections):
        return [d for d in detections if d['class_id'] == WEED_CLASS]

    def get_crops_only(self, detections):
        return [d for d in detections if d['class_id'] == CROP_CLASS]
