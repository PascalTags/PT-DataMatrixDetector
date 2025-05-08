#this shows just a counter

import time
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# --- CONFIGURATION ---
WEIGHTS = '/home/pascal/runs/detect/datamatrix_v8n_synth_lowp/weights/best.pt'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CONF_T = 0.25
TIMEOUT = 120

# Load model
model = YOLO(WEIGHTS)
model.fuse()
model.conf = CONF_T
model.to(DEVICE)

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Only counter window
cv2.namedWindow('Counter', cv2.WINDOW_NORMAL)
print(f'[ℹ] Counting Data Matrices for up to {TIMEOUT}s. Press ESC to quit.')

start_time = time.time()
matrix_count = 0
prev_detected = False

try:
    while True:
        # Timeout check
        if time.time() - start_time > TIMEOUT:
            print(f'[⚠] {TIMEOUT}s elapsed, final count: {matrix_count}')
            break

        ret, frame = cap.read()
        if not ret:
            print('[!] Camera read failed, exiting.')
            break

        # Run detection
        results = model(frame)
        dets = results[0].boxes.data.cpu().numpy()

        currently_detected = bool(dets.size)
        if currently_detected and not prev_detected:
            matrix_count += 1
            print(f'[🔢] Data Matrix count: {matrix_count}')

        prev_detected = currently_detected

        # Build counter image
        counter_img = np.zeros((150, 400, 3), dtype=np.uint8)
        cv2.putText(counter_img, f'Count: {matrix_count}', (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow('Counter', counter_img)

        # Exit on ESC
        if cv2.waitKey(1) == 27:
            print(f'[ℹ] ESC pressed, final count: {matrix_count}')
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print('[ℹ] Done.')
