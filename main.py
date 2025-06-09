from video_source_interface import WebcamSource, LocalFileSource, YouTubeSource
from object_detector import ObjectDetector
from video_processor import VideoProcessor
from distance_calculator import DistanceCalculator

# video_source = WebcamSource()
video_source = LocalFileSource("temp/final_test_video.mp4")
# youtube_url = "https://www.youtube.com/shorts/nLXBinY7BwI" 
# video_source = YouTubeSource(youtube_url)
detector = ObjectDetector("runs/detect/train11/weights/best.pt")
video_processor = VideoProcessor(video_source)
output_path = "static/output.mp4"
VideoProcessor.ensure_output_directory(output_path)
detections = video_processor.process_video(detector, output_path, True)


