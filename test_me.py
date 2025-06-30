from video_sources.video_source_interface import WebcamSource, LocalFileSource, YouTubeSource
from detector.object_detector import ObjectDetector
from processors.video_processor import VideoProcessor
from calculators.distance_calculator import _DistanceCalculator

# video_source = WebcamSource()
video_source = LocalFileSource("temp/wall_video.mp4")
detector = ObjectDetector("training/runs/detect/train12/weights/best.pt")
video_processor = VideoProcessor(video_source)

