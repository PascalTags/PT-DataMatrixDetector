import os
import time
import cv2
import torch
import threading
from PIL import Image
from pylibdmtx.pylibdmtx import decode
from ultralytics import YOLO
import numpy as np

# === CONFIGURATION ===
WEIGHTS        = "/home/pascal/runs/detect/datamatrix_v8n_synth_lowp/weights/best.pt"
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"
CONF_T         = 0.25        # confidence threshold
STABLE_PIX     = 5           # max center shift (px) between frames
REQUIRED_STEPS = 7           # consecutive “stable” frames needed
PAD            = 10          # padding around the box for the crop
OUT_DIR        = "snapshots" # where to save steady snapshots
TIMEOUT        = 30          # seconds to run before exiting

RESULTS_FILE   = "results.txt"  # where decoded results go
os.makedirs(OUT_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(WEIGHTS)
model.fuse()
model.conf = CONF_T
model.to(DEVICE)

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Prepare decode thread
stop_event = threading.Event()
processed = set()

# If there's an existing results file, mark those as done
if os.path.isfile(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as rf:
        for line in rf:
            fname = line.split(" -> ")[0]
            processed.add(fname)

def decode_worker():
    """Continuously look for new snapshots, decode them, and append to results."""
    while not stop_event.is_set():
        for fname in os.listdir(OUT_DIR):
            if fname in processed:
                continue
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(OUT_DIR, fname)
            img  = cv2.imread(path)
            if img is None:
                processed.add(fname)
                continue

            # CLAHE + threshold (same as YoloLiveTest.py)
            gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            _, bw    = cv2.threshold(enhanced, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Attempt Data Matrix decode
            decoded = decode(Image.fromarray(bw))
            text    = decoded[0].data.decode('utf-8') if decoded else None
            result  = text if text else "Cannot Read Data Matrix"

            # Append to results file
            with open(RESULTS_FILE, "a") as rf:
                rf.write(f"{fname} -> {result}\n")

            print(f"[🔍] {fname} decoded: {result}")
            processed.add(fname)

        time.sleep(1)

# Launch decoder in the background
threading.Thread(target=decode_worker, daemon=True).start()

# Create display windows
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.namedWindow("Snapshot",  cv2.WINDOW_NORMAL)
print(f"[ℹ] Scanning for up to {TIMEOUT}s. Press ESC to quit.")

start_time   = time.time()
stable_count = 0
prev_center  = None

try:
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

                cv2.imshow("Snapshot", crop)

                stable_count = 0
                prev_center  = None
        else:
            prev_center  = None
            stable_count = 0

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC key
            print("[ℹ] ESC pressed, exiting.")
            break

finally:
    # Cleanup
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    print("[ℹ] Shutdown complete. All decoded results in", RESULTS_FILE)

