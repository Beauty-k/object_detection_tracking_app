import math
import cv2

class DistanceCalculator:
    def __init__(self, reference_label="scale", reference_mm=300):
        self.reference_label = reference_label
        self.reference_mm = reference_mm
        self.pixel_per_mm = None

    def update_pixel_mm_ratio(self, detections):
        for detection in detections:
            if detection["label"] == self.reference_label:
                _, _, w, _ = detection["box"]
                self.pixel_per_mm = w / self.reference_mm
                return True
        return True
    
    def get_center(self, box):
        x, y, _, _ = box
        return int(x), int(y)
    
    def calculate_distance(self, box1, box2):
        x1, y1 = self.get_center(box1)
        x2, y2 = self.get_center(box2)
        pixel_distance = math.hypot(x2-x1, y2-y1)
        if self.pixel_per_mm:
            return round(pixel_distance * 1.0 / self.pixel_per_mm, 2), (x1, y1), (x2, y2)
        else:
            return None, (x1, y1), (x2, y2)
        
    def annotate_distance(self, frame, box1, box2, label1, label2):
        distance_mm, p1, p2 = self.calculate_distance(box1, box2)
        if distance_mm is not None:
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
            mid = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            cv2.putText(frame, f"{distance_mm} mm", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.putText(frame, label1, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, label2, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


