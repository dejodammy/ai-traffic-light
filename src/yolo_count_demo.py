"""DEFENSE DEMO - live YOLO vehicle DETECTION + COUNTING.

Opens a webcam (or a video/image file), runs YOLOv8 on each frame, draws a green
box on every detected vehicle, shows a running COUNT on the video, AND prints the
count to the terminal so the panel can see it working live.

Usage (from the src/ folder):
    ..\\venv\\Scripts\\python.exe yolo_count_demo.py --source 0
    ..\\venv\\Scripts\\python.exe yolo_count_demo.py --source path\\to\\traffic.mp4
    ..\\venv\\Scripts\\python.exe yolo_count_demo.py --source ..\\sumo_scenarios1\\ikeja_trafficlight.png

Press 'q' in the video window to quit.
"""
from __future__ import annotations

import argparse
import time

import cv2
from ultralytics import YOLO

# COCO class ids for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Live YOLO vehicle detection + counting demo")
    ap.add_argument("--source", default="0",
                    help="Webcam index (e.g. 0), or a path to a video/image file")
    ap.add_argument("--model-name", default="../yolov8n.pt")
    ap.add_argument("--confidence", type=float, default=0.35)
    ap.add_argument("--no-display", action="store_true",
                    help="Headless: print counts only, do not open a window")
    args = ap.parse_args()

    print("Loading YOLOv8 model... (first run may download/initialise)")
    model = YOLO(args.model_name)

    src = args.source
    is_image = str(src).lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))

    # ---- single image mode (guaranteed to work without a camera) ----
    if is_image:
        frame = cv2.imread(src)
        if frame is None:
            print(f"ERROR: could not read image '{src}'")
            return 1
        _process_and_report(model, frame, args, frame_no=0)
        if not args.no_display:
            print("Press any key on the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    # ---- live video / webcam mode ----
    cap_src = int(src) if str(src).isdigit() else src
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"ERROR: could not open source '{src}'. Try a different webcam index "
              f"(0, 1, 2) or pass a video file path.")
        return 1

    print("Running LIVE vehicle counting - press 'q' in the window to quit.\n")
    frame_no = 0
    last_print = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            # throttle terminal printing to ~4x/sec so it stays readable
            do_print = (time.time() - last_print) > 0.25
            annotated, total, by_type = _detect(model, frame, args.confidence)
            if do_print:
                breakdown = ", ".join(f"{k}={v}" for k, v in by_type.items() if v) or "none"
                print(f"frame {frame_no:5d} | VEHICLES IN VIEW: {total:2d}  ({breakdown})")
                last_print = time.time()
            if not args.no_display:
                try:
                    cv2.imshow("YOLO Vehicle Counting - press q to quit", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    print("\n[!] This OpenCV build has no display window support. "
                          "Showing counts in the terminal only.\n"
                          "    To get the video window: in F:\\ai-traffic-light run\n"
                          "    venv\\Scripts\\python.exe -m pip uninstall -y opencv-python opencv-python-headless\n"
                          "    venv\\Scripts\\python.exe -m pip install opencv-python\n")
                    args.no_display = True
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    print(f"\nStopped after {frame_no} frames.")
    return 0


def _detect(model, frame, conf):
    """Run YOLO, draw vehicle boxes, return (annotated_frame, total, by_type)."""
    results = model.predict(frame, conf=conf, verbose=False)
    annotated = frame.copy()
    by_type = {name: 0 for name in VEHICLE_CLASSES.values()}
    total = 0
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        if cls_id not in VEHICLE_CLASSES:
            continue
        total += 1
        by_type[VEHICLE_CLASSES[cls_id]] += 1
        c = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{VEHICLE_CLASSES[cls_id]} {c:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    # big count banner top-left
    cv2.rectangle(annotated, (0, 0), (360, 46), (0, 0, 0), -1)
    cv2.putText(annotated, f"VEHICLES: {total}", (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return annotated, total, by_type


def _process_and_report(model, frame, args, frame_no):
    annotated, total, by_type = _detect(model, frame, args.confidence)
    breakdown = ", ".join(f"{k}={v}" for k, v in by_type.items() if v) or "none"
    print(f"\nVEHICLES DETECTED: {total}  ({breakdown})")
    out = "yolo_count_result.png"
    cv2.imwrite(out, annotated)
    print(f"Annotated image saved to src\\{out}")
    if not args.no_display:
        cv2.imshow("YOLO Vehicle Counting", annotated)


if __name__ == "__main__":
    raise SystemExit(main())
