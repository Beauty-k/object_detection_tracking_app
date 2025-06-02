from ultralytics import YOLO
import cv2
import torch
import os
from pytube import YouTube

def load_model(model_path="yolov8s.pt"):
    print("Loading model...")
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to("cuda")
        print("Using GPU")
    else:
        print("Using CPU")
    return model

def annotate_and_show(model, frame, save_frame=None, writer=None):
    results = model(frame)[0]
    annotated_frame = results.plot()

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = results.names[cls_id]
        conf = float(box.conf[0].item())
        xywh = box.xywh[0].tolist()
        x_center, y_center, w, h = xywh
        x1 = int(x_center - w/2)
        y1 = int(y_center - h/2)

        coord_text = f"XYWH: {round(x_center)}, {round(y_center)}, {round(w)}, {round(h)}"

        cv2.putText(
            annotated_frame,
            coord_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
    if writer:
        writer.write(annotated_frame)
    if save_frame:
        cv2.imshow("Detection", annotated_frame)

def process_video(model, video_path, output_path="static/output.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotate_and_show(model, frame, writer=out)
    cap.release()
    out.release()
    print(f"[DONE] Output saved to: {output_path}")

def process_webcam(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not accessible")
        return

    print("Webcam started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        annotate_and_show(model, frame, save_frame=True)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def download_youtube_video(youtube_url):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    video_path = stream.download(output_path="static", filename="youtube_input.mp4")
    print(f"Downloaded YouTube video to {video_path}")
    return video_path

def main():
    print("Choose input source:")
    print("1. Static Video File")
    print("2. Live Webcam")
    print("3. YouTube Video")

    choice = input("Enter choice (1/2/3): ").strip()
    model = load_model("yolov8m.pt")  # Change to 'yolov8s.pt' if needed

    if choice == "1":
        video_path = input("Enter video file path: ").strip()
        process_video(model, video_path)

    elif choice == "2":
        process_webcam(model)

    elif choice == "3":
        yt_url = input("Enter YouTube video URL: ").strip()
        video_path = download_youtube_video(yt_url)
        process_video(model, video_path)

    else:
        print("Invalid choice. Please enter 1, 2 or 3.")

if __name__ == "__main__":
    main()


