from ultralytics import YOLO
import cv2
import torch
import os
import tempfile

# ----------Object detection-----------
class ObjectDetector:

    def __init__(self, model_path):

        self.model = YOLO(model_path)
        self.setup_device()

    def setup_device(self):
        if torch.cuda.is_available():
            self.model.to('cuda:0')
            print("Running on GPU")
        else:
            print("Running on CPU")

    def get_detection(self, frame):
        results = self.model(frame)[0]
        annotated_frame = results.plot()
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            label = results.names[cls_id]
            conf = float(box.conf[0].item())
            x_center, y_center, w, h = box.xywh[0].tolist()
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)

            coord_text = f"XYWH: {round(x_center)}, {round(y_center)}, {round(w)}, {round(h)}"

            cv2.putText(
                annotated_frame,
                coord_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "box": [round(x, 2) for x in [x_center, y_center, w, h]]
            })

        return annotated_frame, detections

# ----------Video processing---------



    