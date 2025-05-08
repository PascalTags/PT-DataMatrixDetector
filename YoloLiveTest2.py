import os
import time
import cv2
from ultralytics import YOLO
from PIL import Image
from pylibdmtx.pylibdmtx import decode as dm_decode

# === CONFIGURATION ===
ENGINE    = "runs/train/datamatrix_from_scratch_retrain/weights/best.engine"
DEVICE    = "cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
CONF_T    = 0.25       # confidence threshold
DURATION  = 30         # seconds to run
IN_W, IN_H = 320, 180  # inference resolution

if not os.path.isfile(ENGINE):
    print(f"[!] Cannot find engine file: {ENGINE}")
    exit(1)

def decode_datamatrix(frame):
    # your existing BW-conversion + pylibdmtx decode logic here…
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pil = Image.fromarray(gray)
    res = dm_decode(pil)
    return res[0].data.decode("utf-8") if res else None

# ——— LOAD TensorRT MODEL ———
model = YOLO(ENGINE, task="detect", device=DEVICE)  # inject device here
model.conf = CONF_T

# ——— CAMERA LOOP ———
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[!] Cannot open camera")
    exit(1)

print(f"[🎥] Running detection-only for {DURATION}s on {DEVICE}…")
start = time.time()

while time.time() - start < DURATION:
    ret, frame = cap.read()
    if not ret:
        break

    # resize for speed
    small = cv2.resize(frame, (IN_W, IN_H))

    # run inference (predict & NMS) on your TensorRT engine
    results = model(small)

    for r in results:
        for box in r.boxes:
            # get box coords (on small), scale back to original
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            h, w = frame.shape[:2]
            x1o = int(x1 * w / IN_W)
            y1o = int(y1 * h / IN_H)
            x2o = int(x2 * w / IN_W)
            y2o = int(y2 * h / IN_H)

            conf = float(box.conf[0])
            crop = frame[y1o:y2o, x1o:x2o]
            text = decode_datamatrix(crop)

            color = (0,255,0) if text else (0,0,255)
            label = text or f"{conf:.2f}"
            cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), color, 2)
            cv2.putText(frame, label, (x1o, y1o - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detection Only", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit early
        break

cap.release()
cv2.destroyAllWindows()

