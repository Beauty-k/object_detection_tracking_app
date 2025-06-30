import math
from math import radians, tan
import cv2
from typing import Tuple
from calculators.base import ICalculator

class _DistanceCalculator(ICalculator):
    def __init__(self, reference_label:str, reference_mm: float = 300):
    #   self.pixel_per_mm = self.estimate_pixel_per_mm(fov_deg, distance_mm, frame_width_px)
        self.reference_label = reference_label
        self.reference_mm = reference_mm
        self.pixel_per_mm = None
        self.distance_mm = None
        # self.reference_widths = []

    def update_pixel_mm_ratio(self, detections):

    #   fov_rad = radians(fov_deg)
    #   scene_width_mm = 2 * distance_mm * tan(fov_rad / 2)
    #   return frame_width_px / scene_width_mm

        for detection in detections:
            if detection["label"] == self.reference_label:
                _,_,w,_ = detection["box"]
                print("observed pixel width", w)
                self.pixel_per_mm = w / self.reference_mm
                return True
        return False
    
    def get_center(self, box):
        x_center, y_center, _, _ = box
        return x_center, y_center
    
    def calculate(self, box1, box2):
        x1, y1 = self.get_center(box1)
        x2, y2 = self.get_center(box2)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        pixel_distance = math.hypot(x2 - x1, y2 - y1)
        if self.pixel_per_mm:
            distance_mm = round(pixel_distance / self.pixel_per_mm, 2)
            return distance_mm, (x1, y1), (x2, y2)
        return None, (x1, y1), (x2, y2)
    
    def annotate_distance(self, frame, box1, box2, label1, label2):
        print(f"[DEBUG] Annotating distance between {label1} and {label2}")
        distance_mm, p1, p2 = self.calculate(box1, box2)
        if distance_mm is not None:
            cv2.line(frame, p1, p2, (0, 0, 255), 2)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(frame, f"{distance_mm} mm", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, label1, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, label2, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    

    