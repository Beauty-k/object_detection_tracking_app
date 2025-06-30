import os
from fastapi import UploadFile

class FileHandler:
    def __init__(self, upload_dir="temp"):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_uploaded_file(self, file: UploadFile) -> str:
        file_path = os.path.join(self.upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return file_path
