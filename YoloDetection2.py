import torch
import cv2
import numpy as np
from PIL import Image
from pylibdmtx.pylibdmtx import decode

# === Load YOLOv5 model ===
model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/datamatrix_from_scratch_retrain/weights/best.pt')

# === Load and detect ===
img_path = 'Barrel1.jpeg'
img = cv2.imread(img_path)
results = model(img_path)

# === Get detections ===
detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, class]

# === Loop through detections ===
for i, det in enumerate(detections):
    x1, y1, x2, y2, conf, cls = det.tolist()
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # === Add padding safely ===
    pad = 6
    h, w, _ = img.shape
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)
    crop = img[y1p:y2p, x1p:x2p]

    # === Preview crop before decoding ===
    cv2.imshow(f"Detection {i} - Pre-Rotation Preview", crop)
    print(f"[•] Showing Detection {i} before rotation... (10s)")
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    # === Convert to grayscale for decoding ===
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # === Try rotations up to 360 degrees ===
    decoded = None
    for angle in range(0, 360, 5):  # every 5 degrees
        center = (gray.shape[1] // 2, gray.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR)

        decoded = decode(Image.fromarray(rotated))
        if decoded:
            for d in decoded:
                value = d.data.decode('utf-8')
                print(f"[✓] Detection {i}: Decoded at {angle}° → {value}")

                # Show the successful rotated image for 10s
                cv2.imshow(f"Detection {i} - Decoded at {angle}°", rotated)
                print(f"[•] Showing successful decode at {angle}° (10s)")
                cv2.waitKey(10000)
                cv2.destroyAllWindows()
            break

    if not decoded:
        print(f"[x] Detection {i}: No valid code found after 360° rotation.")
