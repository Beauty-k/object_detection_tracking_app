from video_sources.video_source_interface import WebcamSource, LocalFileSource, YouTubeSource
from detector.object_detector import ObjectDetector
from video_processor import VideoProcessor
from calculators.distance import _DistanceCalculator

# video_source = WebcamSource()
video_source = LocalFileSource("temp/wall_video.mp4")
# youtube_url = "https://www.youtube.com/shorts/nLXBinY7BwI" 
# video_source = YouTubeSource(youtube_url)
detector = ObjectDetector("runs/detect/train12/weights/best.pt")
# detector = ObjectDetector("models/yolov8s.pt")

video_processor = VideoProcessor(video_source)
output_path = "static/output.mp4"
VideoProcessor.ensure_output_directory(output_path)
detections = video_processor.process_video(detector, output_path, True)


