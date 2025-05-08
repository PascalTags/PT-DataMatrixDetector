#this script shows the camera feed, and a counter window


import os
import time
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# === CONFIGURATION ===
WEIGHTS      = "/home/pascal/runs/detect/datamatrix_v8n_synth_lowp/weights/best.pt"
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"
CONF_T       = 0.25        # confidence threshold
TIMEOUT      = 120         # seconds to run before exiting

# Load YOLO model
model = YOLO(WEIGHTS)
model.fuse()
model.conf = CONF_T
model.to(DEVICE)

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create display windows
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.namedWindow("Counter", cv2.WINDOW_NORMAL)
print(f"[ℹ] Counting Data Matrices for up to {TIMEOUT}s. Press ESC to quit.")

start_time      = time.time()
matrix_count    = 0
prev_detected   = False

try:
    while True:
        # Exit on timeout
        if time.time() - start_time > TIMEOUT:
            print(f"[⚠] {TIMEOUT}s elapsed, final count: {matrix_count}")
            break

        ret, frame = cap.read()
        if not ret:
            print("[!] Camera read failed, exiting.")
            break

        # Run detection
        results = model(frame)
        dets    = results[0].boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

        # Determine if a matrix is currently detected
        currently_detected = bool(dets.size)

        # If it has just appeared (was out, now in), increment
        if currently_detected and not prev_detected:
            matrix_count += 1
            print(f"[🔢] Data Matrix count: {matrix_count}")

        # Draw box if detected
        if currently_detected:
            # pick highest-confidence box
            x1, y1, x2, y2, conf, _ = max(dets, key=lambda x: x[4])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame,
                        f"Count: {matrix_count}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

        # Update state
        prev_detected = currently_detected

        # Show detection window
        cv2.imshow("Detection", frame)

        # Create and show counter window
        counter_img = np.zeros((150, 400, 3), dtype=np.uint8)
        cv2.putText(
            counter_img,
            f"Count: {matrix_count}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3
        )
        cv2.imshow("Counter", counter_img)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            print(f"[ℹ] ESC pressed, final count: {matrix_count}")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[ℹ] Done.")

