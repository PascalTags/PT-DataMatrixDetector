import os
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# === CONFIGURATION ===
WEIGHTS        = "/home/pascal/runs/detect/datamatrix_v8n_synth_lowp/weights/best.pt"
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"
CONF_T         = 0.25        # confidence threshold
STABLE_PIX     = 5           # max center shift (px) between frames
REQUIRED_STEPS = 7           # consecutive “stable” frames needed
PAD            = 10          # padding around the box for the crop
OUT_DIR        = "snapshots" # where to save steady snapshots
TIMEOUT        = 30          # total seconds to run before exiting

os.makedirs(OUT_DIR, exist_ok=True)

# Load model
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
cv2.namedWindow("Snapshot",  cv2.WINDOW_NORMAL)

print(f"[ℹ] Scanning for up to {TIMEOUT} seconds. Press ESC to quit.")
start_time   = time.time()
stable_count = 0
prev_center  = None

while True:
    # Exit on timeout
    if time.time() - start_time > TIMEOUT:
        print(f"[⚠] {TIMEOUT}s elapsed, exiting.")
        break

    ret, frame = cap.read()
    if not ret:
        print("[!] Camera read failed, exiting.")
        break

    # 1) Run detection
    results = model(frame)
    dets    = results[0].boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

    if dets.size:
        # pick highest-confidence box
        x1, y1, x2, y2, conf, _ = max(dets, key=lambda x: x[4])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # update stability counter
        if prev_center is None:
            stable_count = 1
        else:
            dx = abs(cx - prev_center[0])
            dy = abs(cy - prev_center[1])
            stable_count = stable_count + 1 if dx <= STABLE_PIX and dy <= STABLE_PIX else 1

        prev_center = (cx, cy)

        # draw detection box + counter
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame,
                    f"{stable_count}/{REQUIRED_STEPS}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

        # when stable, save & show snapshot
        if stable_count >= REQUIRED_STEPS:
            w, h    = int(x2 - x1), int(y2 - y1)
            pw, ph  = int(w * 0.4) + PAD, int(h * 0.4) + PAD
            xa, ya  = max(0, int(x1) - pw), max(0, int(y1) - ph)
            xb       = min(frame.shape[1], int(x2) + pw)
            yb       = min(frame.shape[0], int(y2) + ph)
            crop     = frame[ya:yb, xa:xb]

            ts       = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(OUT_DIR, f"datamatrix_{ts}.png")
            cv2.imwrite(out_path, crop)
            print(f"[✅] Saved snapshot to {out_path}")

            # display the cropped snapshot
            cv2.imshow("Snapshot", crop)

            # reset for next capture
            stable_count = 0
            prev_center  = None
    else:
        # no detection → reset stability
        prev_center  = None
        stable_count = 0

    # always show the live detection frame
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        print("[ℹ] ESC pressed, exiting.")
        break

cap.release()
cv2.destroyAllWindows()

