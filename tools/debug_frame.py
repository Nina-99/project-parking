import cv2
from pathlib import Path
from ultralytics import YOLO

video_path = next(Path("/mnt/PARKING").glob("*.mp4"))
cap = cv2.VideoCapture(str(video_path))

# Cargar YOLO26 Medium descargado
model = YOLO("yolo26m.pt")

ret, frame = cap.read()
if ret:
    results = model(frame, imgsz=640, device=0)
    annotated_frame = results[0].plot()
    cv2.imwrite("test_output_medium.jpg", annotated_frame)

    print("\n--- RESULTS WITH YOLO26 MEDIUM (yolo26m.pt) ---")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls_id]
        print(f" -> Object: [{name}] (Clase {cls_id}) | Trust: {conf:.2f}")

cap.release()
