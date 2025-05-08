import torch
import cv2
import numpy as np
from PIL import Image
from pylibdmtx.pylibdmtx import decode
import os
# === Load YOLOv5 model ===
model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/datamatrix_from_scratch_retrain/weights/best.pt')

# === Load and detect ===
img_path = 'Barrel8.jpeg'
img = cv2.imread(img_path)
results = model(img_path)

# === Get detections ===
detections = results.xyxy[0]  # tensor of [x1, y1, x2, y2, conf, class]

# === Loop through detections ===
for i, det in enumerate(detections):
    x1, y1, x2, y2, conf, cls = det.tolist()
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # === Add padding safely ===
    pad = 20
    h, w, _ = img.shape
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)
    crop = img[y1p:y2p, x1p:x2p]

    # === Convert to grayscale ===
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # === Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) ===
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)


    # === Create output folder if it doesn't exist
    output_dir = "screenshots"
    os.makedirs(output_dir, exist_ok=True)

    # === Save the enhanced crop image
    screenshot_path = os.path.join(output_dir, f"det_{i}_enhanced.png")
    cv2.imwrite(screenshot_path, enhanced)
    print(f"[📸] Saved: {screenshot_path}")

    # === Preview enhanced crop before decoding ===
    #cv2.imshow(f"Detection {i} - Enhanced Preview", enhanced)
    #print(f"[•] Showing enhanced Detection {i} (10s)...")
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()

    # === Decode using enhanced image ===
    decoded = decode(Image.fromarray(enhanced))


    # === Try decoding with pylibdmtx ===
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    decoded = decode(Image.fromarray(gray))

    if decoded:
        for d in decoded:
            value = d.data.decode('utf-8')
            print(f"[✓] Detection {i}: Decoded → {value}")

            # Show successful decode region for 10s then exit
            cv2.imshow(f"Decoded: {value}", crop)
            print(f"[•] Showing successful decode result (10s)...")
            #cv2.waitKey(10000)
            cv2.destroyAllWindows()
            exit(0)  # Stop after the first successful decode

    print(f"[x] Detection {i} could not be decoded.")

# === If loop completes with no success
print("⚠️ No Data Matrix codes could be decoded in this image.")
