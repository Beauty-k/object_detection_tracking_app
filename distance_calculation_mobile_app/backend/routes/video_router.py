from fastapi import APIRouter, UploadFile, File
from services.video_service import process_uploaded_video

video_router = APIRouter()

@video_router.post("/calculate-distance")
async def calculate_distance(
    file: UploadFile = File(...),
    label1: str = "blessing_card",
    label2: str = "wallet"
):
    result_path = await process_uploaded_video(file, label1, label2)
    return {
        "message": "Distance measured successfully",
        "output_video_path": result_path
    }
