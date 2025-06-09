from distance_calculator import DistanceCalculator
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
            print(f"[ERROR] Could not open video source: {source}")

    def get_video_properties(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
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
        
        distance_calculator = DistanceCalculator(reference_label="scale", reference_mm=300)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # frame_count += 1
            annotated_frame, detections = detector.get_detection(frame)

            if distance_calculator.pixel_per_mm is None:
                distance_calculator.update_pixel_mm_ratio(detections)

            target_boxes = [d for d in detections if d["label"] in ["Bottle", "Book"]]
            if len(target_boxes) == 2:
                box1 = target_boxes[0]["box"]
                label1 = target_boxes[0]["label"]
                box2 = target_boxes[1]["box"]
                label2 = target_boxes[1]["label"]
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
        return all_detections
