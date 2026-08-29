import csv
import glob
import os

import cv2
import numpy as np

HK = r"dataset\hong kong vehicle.v1i.yolo26"
HK_NAMES = ['bus', 'car', 'minibus', 'motorcycle', 'taxi', 'truck', 'van']
OUT = r"dataset\_v3_staging\hk_audit"
SUS = os.path.join(OUT, "suspects")
os.makedirs(SUS, exist_ok=True)

dec = list(csv.DictReader(open(os.path.join(OUT, "decisions.csv"), encoding="utf-8")))
suspects = [d for d in dec if d["verdict"] == "suspect"]
print("suspects:", len(suspects))

idx = 0
tiles_by_cls = {}
mapping = []
for d in suspects:
    stem = d["frame"]
    p = None
    for split in ["train", "valid", "test"]:
        for ext in [".jpg", ".jpeg", ".png"]:
            q = os.path.join(HK, split, "images", stem + ext)
            if os.path.exists(q):
                p = q
                break
        if p:
            break
    if p is None:
        continue
    im = cv2.imread(p)
    if im is None:
        continue
    H, W = im.shape[:2]
    lab = os.path.join(HK, split, "labels", stem + ".txt")
    with open(lab, "r", encoding="utf-8") as f:
        boxes = [l.split() for l in f if len(l.split()) >= 5]
    for i, parts in enumerate(boxes):
        hk_cls = HK_NAMES[int(float(parts[0]))]
        if hk_cls != d["cls"]:
            continue
        xc, yc, w, h = map(float, parts[1:5])
        x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
        x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
        crop = im[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
            continue
        key = f"{idx:04d}_{hk_cls}"
        cv2.imwrite(os.path.join(SUS, key + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        mapping.append({"idx": idx, "frame": stem, "box_line": i, "cls": hk_cls, "detail": d["detail"]})
        tiles_by_cls.setdefault(hk_cls, []).append((crop, f"#{idx} {hk_cls} {d['detail'][:20]}"))
        idx += 1

with open(os.path.join(OUT, "suspects_index.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["idx", "frame", "box_line", "cls", "detail"])
    w.writeheader()
    w.writerows(mapping)

def grid(tiles, path, cols=8, cell=175):
    tiles = [t for t in tiles if t[0].size]
    if not tiles:
        return
    rows_n = (len(tiles) + cols - 1) // cols
    g = np.full((rows_n * cell, cols * cell, 3), 255, np.uint8)
    for i, (im, t) in enumerate(tiles):
        h, w = im.shape[:2]
        s = min(cell / w, cell / h)
        c = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        r, cc = i // cols, i % cols
        g[r * cell:r * cell + c.shape[0], cc * cell:cc * cell + c.shape[1]] = c
        cv2.putText(g, t, (cc * cell + 3, r * cell + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 2)
    cv2.imwrite(path, g, [cv2.IMWRITE_JPEG_QUALITY, 85])

for cls in ["minibus", "bus", "van", "motorcycle", "taxi"]:
    tiles = tiles_by_cls.get(cls, [])
    for k in range(0, len(tiles), 96):
        grid(tiles[k:k + 96], os.path.join(OUT, f"sus_{cls}_{k // 96}.jpg"))
    print(cls, len(tiles))
