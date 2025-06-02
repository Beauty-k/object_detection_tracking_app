from ultralytics import YOLO
import cv2
import torch
import os
from pytube import YouTube
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

class VideoProcessor:

    def __init__(self, video_source):
        self.video_source = video_source
        self.cap = self.open_video_source()
        self.width, self.height, self.fps = self.get_video_properties()

    def open_video_source(self):
        source = self.video_source.get_video_source()
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            print("Video source opened successfully")
            return cap
        else:
            print(f"[ERROR] Could not open video source: {video_source}")

    def get_video_properties(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Resolution: {width}*{height}, FPS: {fps}")
        return width, height, fps
    
    
    @staticmethod
    def ensure_output_directory(path):
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

    def setup_video_writer(self,output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        return out
    
    def process_video(self, detector, output_path, display=True):
        print("Processing frames....")
        self.ensure_output_directory(output_path)
        writer = self.setup_video_writer(output_path)
        frame_count = 0
        all_detections = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            annotated_frame, detections = detector.get_detection(frame)


            if writer:
                writer.write(annotated_frame)

            if display:
                cv2.imshow("Live Detection", annotated_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                print("[INFO] Stream stopped by user.")
                break

            all_detections.append({
                "frame": frame_count,
                "detections": detections
            })

        self.cap.release()
        if writer:
            writer.release()

        print("[INFO] Video processing complete.")
        return all_detections


    