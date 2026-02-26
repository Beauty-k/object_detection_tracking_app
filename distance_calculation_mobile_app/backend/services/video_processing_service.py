from pathlib import Path
from fastapi import UploadFile
from object_detection_app.video_sources.local_file_source import LocalFileSource
from object_detection_app.detector.object_detector import ObjectDetector
from object_detection_app.processors.video_processor import VideoProcessor
from object_detection_app.calculators.distance_calculator import DistanceCalculator
from services.uploaded_file_saver import UploadedFileSaver
import os

class VideoProcessingService:
    def __init__(self,model_path: str,output_path: str,reference_label: str = "blessing_card"):
        self.model_path = model_path
        self.output_path = output_path
        self.reference_label = reference_label

    def process(self,uploaded_file: UploadFile,label1: str,label2: str) -> str:

        file_saver = UploadedFileSaver(uploaded_file)
        input_video_path = file_saver.save("temp")

        video_source = LocalFileSource(str(input_video_path))
        detector = ObjectDetector(self.model_path)
        video_processor = VideoProcessor(video_source)
        distance_calculator = DistanceCalculator(reference_label=self.reference_label)

        static_output_dir = Path("static")
        static_output_dir.mkdir(exist_ok=True)
        output_video_path = static_output_dir / "output.mp4"

        video_processor._ensure_output_directory(self.output_path)
        output_video_path = os.path.join("static", "output.mp4")

        distance_mm, _ = video_processor.process_video(
            detector=detector,
            output_path=self.output_path,
            display=False,
            target_labels=(label1, label2)
        )
        
        print("Saving video to:", os.path.abspath(self.output_path))

        return distance_mm, self.output_path

    