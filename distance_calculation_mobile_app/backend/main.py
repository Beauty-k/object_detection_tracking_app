import sys
import os

# Add the 'Project' folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI
from routes.video_router import video_router

app = FastAPI(title="Distance Measurement API")
app.include_router(video_router, prefix="/video", tags=["Video Distance"])
