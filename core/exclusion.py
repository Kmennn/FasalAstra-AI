# PURPOSE: Crop Exclusion Zone — Protect Cash Crops
# WHY THIS IS CRITICAL:
#   If a weed grows tangled in crop roots, spraying it
#   also sprays the crop with herbicide = crop dies
#   = farmer loses entire harvest = project causes harm
#
# HOW IT WORKS:
#   After YOLO detection, we check ALL bounding boxes
#   If any CROP box overlaps significantly with a WEED box
#   → ABORT spray for that weed
#
# OVERLAP THRESHOLD: 15%
#   If weed box overlaps crop box by >15% → too risky → skip
#   This was chosen because weeds near crop base
#   typically show 20-40% overlap

OVERLAP_THRESHOLD = 0.15   # 15% overlap = abort spray
WEED_CLASS_ID     = 0
CROP_CLASS_ID     = 1

class CropExclusionZone:
    def __init__(self, threshold=OVERLAP_THRESHOLD):
        self.threshold = threshold

    def _calc_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two boxes
        box format: [x1, y1, x2, y2]
        """
        # Intersection
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])

        if ix2 < ix1 or iy2 < iy1:
            return 0.0

        intersection = (ix2-ix1) * (iy2-iy1)

        # Union
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _overlap_ratio(self, weed_box, crop_box):
        """
        What fraction of weed_box is inside crop_box?
        This is more relevant than IoU for our case
        """
        ix1 = max(weed_box[0], crop_box[0])
        iy1 = max(weed_box[1], crop_box[1])
        ix2 = min(weed_box[2], crop_box[2])
        iy2 = min(weed_box[3], crop_box[3])

        if ix2 < ix1 or iy2 < iy1:
            return 0.0

        intersection = (ix2-ix1) * (iy2-iy1)
        weed_area    = ((weed_box[2]-weed_box[0]) *
                        (weed_box[3]-weed_box[1]))

        if weed_area <= 0:
            return 0.0

        return intersection / weed_area

    def is_safe_to_spray(self, weed_box, all_detections):
        """
        Check if spraying this weed is safe
        
        weed_box       : [x1,y1,x2,y2] of detected weed
        all_detections : list of {class_id, box} for all detections
        
        Returns: (is_safe, overlap_ratio, reason)
        """
        for det in all_detections:
            if det['class_id'] != CROP_CLASS_ID:
                continue

            overlap = self._overlap_ratio(weed_box, det['box'])

            if overlap > self.threshold:
                return False, overlap, "crop_overlap_detected"

        return True, 0.0, "clear_to_spray"


# Quick test
if __name__ == '__main__':
    zone = CropExclusionZone()

    # Weed box
    weed = [140, 180, 200, 240]

    # Scenario 1: No crop nearby
    detections_clear = [
        {'class_id': 2, 'box': [0, 0, 50, 50]}   # soil
    ]
    safe, overlap, reason = zone.is_safe_to_spray(weed, detections_clear)
    print(f"  No crop nearby   : {'✅ SAFE' if safe else '❌ ABORT'} ({reason})")

    # Scenario 2: Crop overlapping weed
    detections_overlap = [
        {'class_id': 1, 'box': [130, 170, 210, 250]}  # crop overlapping
    ]
    safe, overlap, reason = zone.is_safe_to_spray(weed, detections_overlap)
    print(f"  Crop overlapping : {'✅ SAFE' if safe else '❌ ABORT'} overlap={overlap:.0%}")
