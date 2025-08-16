import cv2
import os

def dummy_video_writer_test(input_path, output_path="dummy_output.mp4"):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Cannot open input video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # fallback if FPS is invalid
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # safe codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Dummy video saved: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    dummy_video_writer_test("temp\sample_video_002.mp4")
