import os
import time
import cv2
import torch
import numpy as np
from PIL import Image
from pylibdmtx.pylibdmtx import decode

# === CONFIGURATION ===
WEIGHTS     = "runs/train/datamatrix_from_scratch_retrain/weights/best.pt"
DEVICE      = "0" if torch.cuda.is_available() else "cpu"  # GPU index or 'cpu'
PAD         = 10        # padding around detected box
CONF_T      = 0.25      # YOLO confidence threshold
DURATION    = 30        # total seconds to run
IN_W, IN_H  = 640, 360  # resize frames to this before inference
SKIP_DECODE = 3         # only run decode every N frames

# — sanity check weights file —
if not os.path.isfile(WEIGHTS):
    print(f"[!] Weights not found at {WEIGHTS}")
    exit(1)

print(f"[ℹ] Loading model from {WEIGHTS} onto device {DEVICE}")
model = torch.hub.load(
    'ultralytics/yolov5', 'custom', WEIGHTS, device=DEVICE
)
model.conf = CONF_T
model.half()  # use FP16 for speed

# — open camera —
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[!] Cannot open camera")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"[🎥] Running live detection for {DURATION}s (resize {IN_W}×{IN_H}, skip-decode every {SKIP_DECODE} frames)…")
start_time = time.time()
frame_idx  = 0

while True:
    # timeout
    if time.time() - start_time > DURATION:
        print(f"[⏱] {DURATION}s elapsed. Exiting.")
        break

    ret, frame = cap.read()
    if not ret:
        print("[!] Frame grab failed, exiting.")
        break

    # 1) downscale
    small = cv2.resize(frame, (IN_W, IN_H))
    # 2) inference under AMP
    with torch.amp.autocast('cuda' if DEVICE != 'cpu' else 'cpu'):
        preds = model(small)
    dets = preds.xyxy[0].cpu().numpy()  # shape (N,6)

    # 3) process detections
    if frame_idx % SKIP_DECODE == 0:
        # full pipeline: crop → CLAHE → thresh → decode
        for x1, y1, x2, y2, conf, cls in dets:
            # scale back coords
            sx = frame.shape[1] / IN_W
            sy = frame.shape[0] / IN_H
            x1i = int(x1 * sx); y1i = int(y1 * sy)
            x2i = int(x2 * sx); y2i = int(y2 * sy)
            w = x2i - x1i; h = y2i - y1i

            # padded crop
            pw = int(w * 0.4) + PAD
            ph = int(h * 0.4) + PAD
            xa = max(0, x1i - pw); ya = max(0, y1i - ph)
            xb = min(frame.shape[1], x2i + pw)
            yb = min(frame.shape[0], y2i + ph)
            crop = frame[ya:yb, xa:xb]

            # CLAHE + threshold
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
            _, bw = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # decode Data Matrix
            decoded = decode(Image.fromarray(bw))
            text = decoded[0].data.decode('utf-8') if decoded else None

            # draw
            color = (0,255,0) if text else (0,0,255)
            cv2.rectangle(frame, (x1i,y1i), (x2i,y2i), color, 2)
            label = text or f"{conf:.2f}"
            cv2.putText(frame, label, (x1i, y1i-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        # lighter: just draw the boxes + confidences
        sx = frame.shape[1] / IN_W
        sy = frame.shape[0] / IN_H
        for x1, y1, x2, y2, conf, cls in dets:
            x1i = int(x1 * sx); y1i = int(y1 * sy)
            x2i = int(x2 * sx); y2i = int(y2 * sy)
            cv2.rectangle(frame, (x1i,y1i), (x2i,y2i), (255,200,0), 1)
            cv2.putText(frame, f"{conf:.2f}", (x1i, y1i-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)

    # 4) display
    cv2.imshow("Fast Live DM", frame)
    cv2.waitKey(1)

    frame_idx += 1

# cleanup
cap.release()
cv2.destroyAllWindows()

