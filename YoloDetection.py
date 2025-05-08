import torch
import cv2
import os
from PIL import Image
from pylibdmtx.pylibdmtx import decode

# === Load YOLOv5 model ===
model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/datamatrix_from_scratch_retrain/weights/best.pt')

# === Load and detect ===
img_path = 'Barrel1.jpeg'
img = cv2.imread(img_path)
results = model(img_path)

# === Get detections ===
detections = results.xyxy[0]  # tensor of [x1, y1, x2, y2, conf, class]

# === Loop through each detection ===
for i, det in enumerate(detections):
    x1, y1, x2, y2, conf, cls = det.tolist()
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # === Crop and save temp file ===
    crop = img[y1:y2, x1:x2]
    temp_path = f"temp_crop_{i}.png"
    cv2.imwrite(temp_path, crop)

  # === Show cropped image ===
    cv2.imshow(f"Detection {i}", crop)
    print(f"\n[•] Showing Detection {i}: Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # === Decode with pylibdmtx ===
    decoded = decode(Image.open(temp_path))
    if decoded:
        for d in decoded:
            value = d.data.decode('utf-8')
            print(f"[✓] Detection {i}: {value}")
    else:
        print(f"[x] Detection {i} could not be decoded.")

    # === Delete temp file ===
    os.remove(temp_path)

