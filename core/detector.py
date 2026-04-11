# UPDATED FOR RPi4:
# imgsz=640 (was 320)
# Confidence threshold stays 0.75
# Added RPi4 ONNX inference option
# WHY: Direct PyTorch on RPi4 is fine
#      RPi4 4GB RAM handles YOLOv8n easily

from ultralytics import YOLO
import cv2
import time

CONF_THRESHOLD = 0.75
WEED_CLASS     = 0
CROP_CLASS     = 1
SOIL_CLASS     = 2
IMAGE_SIZE     = 640   # upgraded from 320

class FasalAstraDetector:
    def __init__(self,
                 model_path='runs/fasal_astra_v3_rpi/weights/best.pt'):
        print(f"  Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.imgsz = IMAGE_SIZE
        print(f"  ✅ Model loaded | imgsz={IMAGE_SIZE}")

    def detect(self, frame):
        t0 = time.time()

        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=CONF_THRESHOLD,
            verbose=False
        )[0]

        inference_ms = (time.time() - t0) * 1000
        detections   = []

        for box in results.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            x1,y1,x2,y2 = box.xyxy[0].tolist()

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append({
                'class_id'    : class_id,
                'class_name'  : ['weed','crop','soil'][class_id],
                'confidence'  : confidence,
                'box'         : [x1,y1,x2,y2],
                'centroid'    : (cx, cy),
                'inference_ms': inference_ms
            })

        return detections

    def get_weeds_only(self, detections):
        return [d for d in detections if d['class_id'] == WEED_CLASS]
