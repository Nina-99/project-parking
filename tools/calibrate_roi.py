# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

import json
from pathlib import Path

import cv2
from ultralytics import YOLO

INPUT_VIDEOS_DIR = Path("/mnt/PARKING/ver/")
CONFIG_FILE = Path("roi_config.json")
MODEL_NAME = "yolo26m.pt"

VEHICLE_CLASSES = [2, 3, 5, 7]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".MP4", ".AVI"}


def nothing(x):
    pass


def main():
    if not INPUT_VIDEOS_DIR.exists():
        print(f"[ERROR] La ruta no existe: {INPUT_VIDEOS_DIR}")
        return

    video_files = sorted(
        [f for f in INPUT_VIDEOS_DIR.iterdir() if f.is_file() and f.suffix in VIDEO_EXTENSIONS]
    )

    if not video_files:
        print(f"[ERROR] No se encontraron videos en: {INPUT_VIDEOS_DIR}")
        return

    first_video = video_files[0]
    print(f"Video: {first_video.name}")
    print("SPACE pausa, A/D ±5 s, W/S velocidad (1x/2x/4x), "
          "R reinicia, ENTER guarda el ROI, Q sale")

    cap = cv2.VideoCapture(str(first_video))
    if not cap.isOpened():
        print(f"[ERROR] no se pudo abrir el video: {first_video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    window_name = "Calibrator Zone ROI - Parking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    cv2.createTrackbar("Y_MIN (Up)", window_name, 35, 100, nothing)
    cv2.createTrackbar("Y_MAX (Down)", window_name, 95, 100, nothing)
    cv2.createTrackbar("X_MIN (Left)", window_name, 15, 100, nothing)
    cv2.createTrackbar("X_MAX (Right)", window_name, 85, 100, nothing)
    cv2.createTrackbar("Confidence %", window_name, 25, 100, nothing)

    model = YOLO(MODEL_NAME)

    paused = False
    playback_speed = 1.0  # 1.0x, 2.0x, 4.0x
    current_frame_img = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            current_frame_img = frame.copy()
        else:
            frame = current_frame_img.copy() if current_frame_img is not None else None
            if frame is None:
                continue

        h, w, _ = frame.shape

        y_min_pct = cv2.getTrackbarPos("Y_MIN (Up)", window_name) / 100.0
        y_max_pct = cv2.getTrackbarPos("Y_MAX (Down)", window_name) / 100.0
        x_min_pct = cv2.getTrackbarPos("X_MIN (Left)", window_name) / 100.0
        x_max_pct = cv2.getTrackbarPos("X_MAX (Right)", window_name) / 100.0
        conf_thresh = max(5, cv2.getTrackbarPos("Confidence %", window_name)) / 100.0

        rx1 = int(w * x_min_pct)
        ry1 = int(h * y_min_pct)
        rx2 = int(w * x_max_pct)
        ry2 = int(h * y_max_pct)

        rx1, rx2 = min(rx1, rx2 - 10), max(rx2, rx1 + 10)
        ry1, ry2 = min(ry1, ry2 - 10), max(ry2, ry1 + 10)
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)

        # INFERNCE IN THE BOX
        roi_crop = frame[ry1:ry2, rx1:rx2]

        display_frame = frame.copy()

        if roi_crop.size > 0:
            results = model.predict(roi_crop, verbose=False, device=0, conf=conf_thresh)

            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id in VEHICLE_CLASSES:
                        # Mapear coordenadas locales del Crop -> Coordenadas globales del Frame
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                        global_x1 = rx1 + bx1
                        global_y1 = ry1 + by1
                        global_x2 = rx1 + bx2
                        global_y2 = ry1 + by2

                        class_label = "vehicle" if cls_id == 72 else model.names[cls_id]
                        label = f"{class_label} {conf:.2f} [ROI]"

                        cv2.rectangle(
                            display_frame,
                            (global_x1, global_y1),
                            (global_x2, global_y2),
                            (0, 255, 255),
                            2,
                        )
                        cv2.putText(
                            display_frame,
                            label,
                            (global_x1, max(20, global_y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                        )

        cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
        cv2.putText(
            display_frame,
            f"EXCLUSIVE AREA ROI ({rx2 - rx1}x{ry2 - ry1}px)",
            (rx1 + 10, ry1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        status_str = f"| SPACE: {'PAUSA' if paused else 'PLAY'} | Vel: {playback_speed:.1f}x | Frame: {current_frame_pos}/{total_frames}"
        cv2.putText(
            display_frame,
            status_str,
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window_name, display_frame)

        wait_ms = int(max(1, (1000.0 / fps) / playback_speed)) if not paused else 50
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == 32:  # Space bar -> Pause / Resume
            paused = not paused

        elif key == ord("d") or key == ord("D"):  # Skip forward 5 seconds
            curr = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, curr + int(fps * 5)))

        elif key == ord("a") or key == ord("A"):  # backward 5 seconds
            curr = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr - int(fps * 5)))

        elif key == ord("w") or key == ord("W"):  # Increase speed (1x -> 2x -> 4x)
            playback_speed = min(8.0, playback_speed * 2.0)
            print(f"[PLAYER] Speed: {playback_speed}x")

        elif key == ord("s") or key == ord("S"):  # Decrease speed
            playback_speed = max(0.5, playback_speed / 2.0)
            print(f"[PLAYER] speed: {playback_speed}x")

        elif key == ord("r") or key == ord("R"):  # Restart video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        elif key == 13:  # Saved JSON
            roi_config = {
                "roi_x_min": x_min_pct,
                "roi_x_max": x_max_pct,
                "roi_y_min": y_min_pct,
                "roi_y_max": y_max_pct,
                "confidence_threshold": conf_thresh,
                "resolution": {"width": w, "height": h},
            }
            with open(CONFIG_FILE, "w") as f:
                json.dump(roi_config, f, indent=4)

            print(f"\n[SUCCESS] Settings saved in: {CONFIG_FILE.resolve()}")
            print(f"Values: {roi_config}")
            break

        elif key == ord("q") or key == ord("Q") or key == 27:  # Salir sin guardar
            print("\n[INFO] Exit without saving changes.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
