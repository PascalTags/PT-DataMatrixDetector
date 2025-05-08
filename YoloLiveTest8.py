import os
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pylibdmtx.pylibdmtx import decode
from PIL import Image

# === CONFIGURATION ===
WEIGHTS        = "/home/pascal/runs/detect/datamatrix_v8n_synth_lowp/weights/best.pt"
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"
CONF_T         = 0.25         # YOLO confidence threshold
STABLE_PIX     = 5            # max center shift (px) between frames
REQUIRED_STEPS = 2            # consecutive “stable” frames needed
PAD            = 10           # padding around the box for the crop
OUT_DIR        = "snapshots"  # where to save steady snapshots
RESULTS_FILE   = "results.txt"# decoding results log
TIMEOUT        = 60           # seconds to run live detection

os.makedirs(OUT_DIR, exist_ok=True)

# === LOAD YOLO MODEL ===
model = YOLO(WEIGHTS)
model.fuse()
model.conf = CONF_T
model.to(DEVICE)

# === OPEN CAMERA ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.namedWindow("Snapshot",   cv2.WINDOW_NORMAL)
print(f"[ℹ] Live detection for {TIMEOUT}s. Press ESC to quit early.")

# === DETECTION + SNAPSHOT LOOP ===
start_time   = time.time()
stable_count = 0
prev_center  = None

while True:
    # exit on timeout
    if time.time() - start_time > TIMEOUT:
        print(f"[⚠] {TIMEOUT}s elapsed; ending live view.")
        break

    ret, frame = cap.read()
    if not ret:
        print("[!] Camera read failed; exiting.")
        break

    # YOLO inference
    res  = model(frame)[0]
    dets = res.boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

    if dets.size:
        # pick highest-confidence box
        x1,y1,x2,y2,_,_ = max(dets, key=lambda d: d[4])
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)

        # update stability
        if prev_center is None:
            stable_count = 1
        else:
            dx = abs(cx - prev_center[0])
            dy = abs(cy - prev_center[1])
            stable_count = stable_count + 1 if dx <= STABLE_PIX and dy <= STABLE_PIX else 1

        prev_center = (cx, cy)

        # draw box + counter
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame,
                    f"{stable_count}/{REQUIRED_STEPS}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

        # when stable, save snapshot
        if stable_count >= REQUIRED_STEPS:
            w, h = int(x2-x1), int(y2-y1)
            pw   = int(w*0.4) + PAD
            ph   = int(h*0.4) + PAD
            xa   = max(0, int(x1)-pw)
            ya   = max(0, int(y1)-ph)
            xb   = min(frame.shape[1], int(x2)+pw)
            yb   = min(frame.shape[0], int(y2)+ph)
            crop = frame[ya:yb, xa:xb]

            ts    = time.strftime("%Y%m%d_%H%M%S")
            fname = f"datamatrix_{ts}.png"
            path  = os.path.join(OUT_DIR, fname)
            cv2.imwrite(path, crop)
            print(f"[✅] Saved snapshot: {fname}")
            cv2.imshow("Snapshot", crop)

            stable_count = 0
            prev_center  = None
    else:
        prev_center  = None
        stable_count = 0

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) == 27:
        print("[ℹ] ESC pressed; ending live view.")
        break

# cleanup live view
cap.release()
cv2.destroyAllWindows()

# === POST-PROCESS SAVED SNAPSHOTS ===
print("[ℹ] Post-processing snapshots for decoding...")

# stronger contrast enhancer
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

with open(RESULTS_FILE, "a") as rf:
    for fname in sorted(os.listdir(OUT_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(OUT_DIR, fname)
        img  = cv2.imread(path)
        if img is None:
            continue

        # 1) filter out incomplete crops
        res_c = model(img)[0]
        if res_c.boxes.data.cpu().numpy().size == 0:
            print(f"[✂️] Deleting incomplete crop: {fname}")
            os.remove(path)
            continue

        # 2) stronger contrast pipeline
        gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist_eq   = cv2.equalizeHist(gray)                                              # global equalization
        clahe_eq  = clahe.apply(hist_eq)                                                # CLAHE
        stretched = cv2.normalize(clahe_eq, None, 0, 255, cv2.NORM_MINMAX)              # stretch full range
        contrast  = cv2.convertScaleAbs(stretched, alpha=4.0, beta=0)                   # amplify contrast
        _, bw     = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3) try decoding at multiple rotations
        decoded = None
        for angle, bw_img in zip(
            [0, 90, 180, 270],
            [bw,
             cv2.rotate(bw, cv2.ROTATE_90_CLOCKWISE),
             cv2.rotate(bw, cv2.ROTATE_180),
             cv2.rotate(bw, cv2.ROTATE_90_COUNTERCLOCKWISE)]
        ):
            decs = decode(Image.fromarray(bw_img))
            if decs:
                decoded = decs[0].data.decode("utf-8")
                print(f"[🔄] {fname} decoded at {angle}°")
                break

        result = decoded if decoded else "not decodeable"

        # 4) log result
        rf.write(f"{fname} -> {result}\n")
        print(f"[🔍] {fname} -> {result}")

print(f"[ℹ] Done. Results in `{RESULTS_FILE}`.")

