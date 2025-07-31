import shutil
from pathlib import Path
from fastapi import UploadFile
from object_detection_app.video_sources.video_source_interface import LocalFileSource
from object_detection_app.detector.object_detector import ObjectDetector
from object_detection_app.processors.video_processor import VideoProcessor
from object_detection_app.calculators.distance_calculator import DistanceCalculator

def process_uploaded_video(file: UploadFile, label1: str, label2: str):
    # Save uploaded file to a temp location
    temp_input_path = Path("temp") / file.filename
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Setup components
    video_source = LocalFileSource(str(temp_input_path))
    detector = ObjectDetector("runs/detect/train14/weights/best.pt")
    video_processor = VideoProcessor(video_source)
    distance_calculator = DistanceCalculator(reference_label="blessing_card")

    # Process video
    output_path = "static/output.mp4"
    VideoProcessor.ensure_output_directory(output_path)
    video_processor.process_video(
        detector=detector,
        output_path=output_path,
        display=True,
        target_labels=(label1, label2)
    )
    
    return output_path
