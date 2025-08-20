from object_detection_app.calculators.distance_calculator import DistanceCalculator
# from calculators.test_sample_distance import GeometryDistanceCalculator
from deep_sort_realtime.deepsort_tracker import DeepSort
from object_detection_app.tracker.deep_sort_tracker import DeepSortTracker
import cv2
import os

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
            raise FileNotFoundError(f"[ERROR] Could not open video source: {source}")

    def get_video_properties(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # if fps == 0:
        #      fps = 30 
        return width, height, fps
    
    
    @staticmethod
    def ensure_output_directory(path):
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

    def setup_video_writer(self,output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        return out
    
    def process_video(self, detector, output_path, display, target_labels):
        print("Processing frames....")
        self.ensure_output_directory(output_path)
        writer = self.setup_video_writer(output_path)
        frame_count = 0
        all_detections = []
        
        distance_calculator = DistanceCalculator("blessing_card", reference_mm=85)

        tracker = DeepSort(max_age=30)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            annotated_frame, detections = detector.get_detection(frame)

            tracking_inputs = []
            for d in detections:
                xc, yc, w, h = d["box"]
                x = xc - w / 2
                y = yc - h / 2
                tracking_inputs.append(([x, y, w, h], d["confidence"], d["label"]))

            tracks = tracker.update_tracks(tracking_inputs, frame=frame)
            tracked_detections = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                l, t, r, b = track.to_ltrb()
                label = track.det_class
                box = [l, t, r - l, b - t]

                cv2.putText(annotated_frame, f"{label}-{track_id}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                tracked_detections.append({
                    "id": track_id,
                    "label": label,
                    "box": box
                })

            if distance_calculator.pixel_per_mm is None:
                distance_calculator.update_pixel_mm_ratio(detections)

            measured_distance_mm = None
            target_boxes = [d for d in detections if d["label"] in target_labels]
            if len(target_boxes) == 2:
                box1 = target_boxes[0]["box"]
                label1 = target_boxes[0]["label"]
                box2 = target_boxes[1]["box"]
                label2 = target_boxes[1]["label"]
                measured_distance_mm , _, _ = distance_calculator.calculate(box1, box2)
                distance_calculator.annotate_distance(annotated_frame, box1, box2, label1, label2)

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
        return measured_distance_mm, all_detections


