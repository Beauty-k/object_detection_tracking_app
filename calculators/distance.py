import math
import cv2
from typing import Tuple
from calculators.base import ICalculator

class _DistanceCalculator(ICalculator):
    def __init__(self, reference_label:str, reference_mm: float = 300):
        self.reference_label = reference_label
        self.reference_mm = reference_mm
        self.pixel_per_mm = None
        self.reference_widths = []

    def update_pixel_mm_ratio(self, detections):
        for detection in detections:
            if detection["label"] == self.reference_label:
                _,_,w,_ = detection["box"]
                self.reference_widths.append(w)
                if len(self.reference_widths) > 10:
                    self.reference_widths.pop(0)
                    avg_width = sum(self.reference_widths) / len(self.reference_widths)
                self.pixel_per_mm = w / self.reference_mm
                # print(f"Pixel/mm ratio: {self.pixel_per_mm:.4f}, Detected Distance: {self.distance_mm} mm")
                return True
        return False
    
    def get_center(self, box: Tuple[int, int, int, int]):
        x_center, y_center, _, _ = box
        return x_center, y_center
    
    def calculate(self, box1, box2):
        x1, y1 = self.get_center(box1)
        x2, y2 = self.get_center(box2)
        pixel_distance = math.hypot(x2 - x1, y2 - y1)
        if self.pixel_per_mm:
            return round(pixel_distance / self.pixel_per_mm, 2), (x1, y1), (x2, y2)
        return None, (x1, y1), (x2, y2)
    
    def annotate_distance(self, frame, box1, box2, label1, label2):
        distance_mm, p1, p2 = self.calculate(box1, box2)
        if distance_mm is None or not all(isinstance(i, int) for i in p1 + p2):
            return
        if distance_mm is not None:
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(frame, f"{distance_mm} mm", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, label1, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, label2, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def get_distance_calculator(reference_label: str, reference_mm: float = 300) -> ICalculator:
        return _DistanceCalculator(reference_label, reference_mm)
    