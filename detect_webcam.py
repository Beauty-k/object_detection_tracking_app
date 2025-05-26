from ultralytics import YOLO
import cv2
import torch

def detect_from_webcam(model_path="yolov8s.pt"):
    print("Starting live webcam detection...")

    # Load YOLO model
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to("cuda")
    else:
        print("Running on CPU")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access the webcam.")
        return

    print("Webcam feed started. Press 'q' to exit.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame_count += 1
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
                0.5,
                (255,0,0),
                1,
                cv2.LINE_AA
            )

        cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")


