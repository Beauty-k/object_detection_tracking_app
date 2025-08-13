import sys
import os
from fastapi.middleware.cors import CORSMiddleware


# Add the 'Project' folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI
from routes.video_router import video_router

app = FastAPI(title="Distance Measurement API")
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Server is alive!"}
app.include_router(video_router, prefix="/video", tags=["Video Distance"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your Flutter IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
