from abc import ABC, abstractmethod
import os
import uuid
import yt_dlp
import tempfile

class VideoSourceInterface(ABC):
    @abstractmethod
    def get_video_source(self):
        pass

class WebcamSource(VideoSourceInterface):
    def get_video_source(self):
        return 0

class LocalFileSource(VideoSourceInterface):
    def __init__(self, file_path):
        self.file_path = file_path
    
    def get_video_source(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} not found")
        return self.file_path
    
class YouTubeSource(VideoSourceInterface):
    def __init__(self, yt_url):
        self.yt_url = yt_url

    def get_video_source(self):
       temp_filename = f"youtube_video_{uuid.uuid4().hex}.mp4"
       temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
       ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': temp_path,
            'quiet': True,
        }

       try:
            print(f"[INFO] Downloading video using yt-dlp: {self.yt_url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.yt_url])
            return temp_path

       except Exception as e:
            print(f"[ERROR] yt-dlp failed: {e}")
            raise
