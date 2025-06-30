from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import shutil

from detector.object_detector import ObjectDetector
from tracker.deep_sort_tracker import DeepSortTracker
from processors.file_handler import FileHandler
from calculators.distance_calculator import _DistanceCalculator


router = APIRouter()
detector = ObjectDetector('training/runs/detect/train12/weights/best.pt')
tracker = DeepSortTracker()
file_handler = FileHandler()
calculator = _DistanceCalculator

@router.post("/detect/")
async def detect_objects(
    file: UploadFile = File(...),
    label1: str = Form("scale"),
    label2: str = Form("diary")
):
    input_path = os.path.join("temp", file.filename)
    output_path = os.path.join("static", "annotated_output.mp4")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Call run_detection which internally uses VideoProcessor
    all_detections = detector.run_detection(input_path, output_path, label1, label2)

    return JSONResponse({
        "video_url": "/static/annotated_output.mp4",
        "total_frames": len(all_detections),
        "target_labels": [label1, label2],
        "message": "Detection complete!"
    })