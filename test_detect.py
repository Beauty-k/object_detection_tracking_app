from detect import detect_from_video
from detect_c import detect_from_webcam


# Run detection
detections = detect_from_video("temp/person-bicycle-car-detection.mp4")
# detections = detect_from_webcam()

print(detections[:2])