from fastapi import APIRouter, UploadFile, File
from services.video_service import process_uploaded_video

video_router = APIRouter()

@video_router.post("/calculate-distance")
def calculate_distance(
    file: UploadFile = File("temp/sample_video_002"),
    label1: str = "blessing_card",
    label2: str = "wallet"
):
    result_path = process_uploaded_video(file, label1, label2)
    return {
        "message": "Distance measured successfully",
        "output_video_path": result_path
    }
