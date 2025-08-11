from fastapi import UploadFile
from services.video_service import VideoProcessingService  # You can alias or reimport differently if needed

class VideoProcessingController:
    def calculate_distance(self, file: UploadFile, label1: str, label2: str) -> dict:
        model_path = "runs/detect/train14/weights/best.pt" 
        output_path = "static/output.mp4"
        Video_service = VideoProcessingService(model_path,output_path)
        result_path = Video_service.process(file, label1, label2)
        return {
            "message": "Distance measured successfully",
            "output_video_path": result_path
        }
