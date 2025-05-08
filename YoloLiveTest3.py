import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
from pylibdmtx.pylibdmtx import decode
from ultralytics import YOLO

# === CONFIGURATION ===
WEIGHTS = "/home/pascal/runs/detect/datamatrix_v8n_synth/weights/best.pt"
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
PAD         = 10        # padding around each detected box for decoding
CONF_T      = 0.25      # confidence threshold
DURATION    = 30        # seconds to run
SKIP_DECODE = 3         # only run Data Matrix decode every N frames

# Sanity-check weights
if not os.path.isfile(WEIGHTS):
    print(f"[!] Weights not found at {WEIGHTS}")
    exit(1)

# Load YOLOv8 model
print(f"[ℹ] Loading YOLOv8 model from {WEIGHTS} → device {DEVICE}")
model = YOLO(WEIGHTS)
model.fuse()                # fuse Conv+BN
model.conf = CONF_T         # set confidence threshold
model.to(DEVICE)            # move model to GPU/CPU

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[!] Cannot open camera")
    exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"[🎥] Running live detection for {DURATION}s…")
start_time = time.time()
frame_idx  = 0

while True:
    # timeout
    if time.time() - start_time > DURATION:
        break

    ret, frame = cap.read()
    if not ret:
        break

    # ── 1) inference ──
    # YOLOv8 will handle resizing internally
    results = model(frame)               # returns a list with one Results
    res     = results[0]
    # boxes.data: tensor[N,6] = (x1,y1,x2,y2,conf,cls)
    dets    = res.boxes.data.cpu().numpy()

    # ── 2) decode & draw ──
    if frame_idx % SKIP_DECODE == 0:
        for x1, y1, x2, y2, conf, cls in dets:
            x1i, y1i = int(x1), int(y1)
            x2i, y2i = int(x2), int(y2)
            w, h     = x2i - x1i, y2i - y1i

            # padded crop around detection
            pw, ph = int(w * 0.4) + PAD, int(h * 0.4) + PAD
            xa, ya = max(0, x1i - pw), max(0, y1i - ph)
            xb      = min(frame.shape[1], x2i + pw)
            yb      = min(frame.shape[0], y2i + ph)
            crop    = frame[ya:yb, xa:xb]

            # preprocess for decoding
            gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
            _, bw    = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # decode Data Matrix
            decoded = decode(Image.fromarray(bw))
            text    = decoded[0].data.decode('utf-8') if decoded else None

            # draw result
            color = (0,255,0) if text else (0,0,255)
            cv2.rectangle(frame, (x1i,y1i), (x2i,y2i), color, 2)
            label = text or f"{conf:.2f}"
            cv2.putText(frame, label, (x1i, y1i-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        # just draw boxes + confidences
        for x1, y1, x2, y2, conf, cls in dets:
            x1i, y1i = int(x1), int(y1)
            x2i, y2i = int(x2), int(y2)
            cv2.rectangle(frame, (x1i,y1i), (x2i,y2i), (255,200,0), 1)
            cv2.putText(frame, f"{conf:.2f}", (x1i, y1i-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)

    # ── 3) display ──
    cv2.imshow("YOLOv8 Live DataMatrix", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

    frame_idx += 1

# cleanup
cap.release()
cv2.destroyAllWindows()
