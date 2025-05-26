from ultralytics import YOLO
import cv2
import torch
import os

def detect_from_video(video_path, output_path="static/output.mp4", model_path="yolov8m.pt"):
    print("Starting Detection...")

    # Load model
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to("cuda")
        # print("Running on GPU")
    else:
        print("Running on CPU")

    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] could not open video: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Ensure output folder exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Processing frames....")

    frame_count = 0
    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count +=1
        results = model(frame)[0]
        annotated_frame = results.plot()
        out.write(annotated_frame)
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            label = results.names[cls_id]
            conf = float(box.conf[0].item())
            # x_centre, y_centre, w, h 
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
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "box": [round(x,2) for x in xywh]
            })

        all_detections.append({
            "frame": frame_count,
            "detections": detections
        })


        # write frame to output 
        out.write(annotated_frame)

    # Clean up
    cap.release
    out.release
    print(f"[SUCCESSFUL] Detection complete! Output saved to {output_path}")
    return all_detections
        





