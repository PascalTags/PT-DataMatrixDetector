import torch
import cv2
import os
from PIL import Image
from pylibdmtx.pylibdmtx import decode

# === CONFIGURATION ===
image_folder = "."  # Directory with images
image_files = [f"Barrel{i}.jpeg" for i in range(1, 9)]
output_txt = "decodes.txt"
failed_dir = "failed"
pad = 10  # base padding

# === PREP ===
os.makedirs(failed_dir, exist_ok=True)

# === LOAD YOLO MODEL ===
model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/datamatrix_from_scratch_retrain/weights/best.pt')

# === PROCESS IMAGES ===
with open(output_txt, "w") as log:
    for img_file in image_files:
        print(f"\n[🔍] Processing {img_file}")
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Could not read {img_file}, skipping.")
            continue

        h, w, _ = img.shape
        results = model(img_path)
        detections = results.xyxy[0]
        success = False

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # === Extra padding proportional to box size ===
            x_pad = int((x2 - x1) * 0.4) + pad
            y_pad = int((y2 - y1) * 0.4) + pad

            x1p = max(0, x1 - x_pad)
            y1p = max(0, y1 - y_pad)
            x2p = min(w, x2 + x_pad)
            y2p = min(h, y2 + y_pad)
            crop = img[y1p:y2p, x1p:x2p]

            # === Enhance crop with CLAHE ===
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
 
	    # === Apply threshold to make PURE black and white ===
            _, bw = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

 
            # === Attempt decode ===
            decoded = decode(Image.fromarray(bw))
            if decoded:
                value = decoded[0].data.decode('utf-8')
                print(f"[✓] {img_file} → {value}")
                log.write(f"{img_file} -> {value}\n")
                success = True
                break

        if not success:
            fail_path = os.path.join(failed_dir, img_file)
            cv2.imwrite(fail_path, img)
            print(f"[x] No decode. Saved full image to: {fail_path}")

