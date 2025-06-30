
class FileSource:
    def __init__(self, path):
        self.path = path

    def get_video_source(self):
        return self.path