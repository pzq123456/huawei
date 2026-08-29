import csv
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

STAGE = r"dataset\_v3_staging"
CROPS = os.path.join(STAGE, "crops")
AUD = os.path.join(STAGE, "hk_audit")
os.makedirs(AUD, exist_ok=True)

B12 = r"dataset\batch_12.v9i.yolov11"
B12_NAMES = ['Coach', 'Franchised Bus', 'HGV', 'LGV', 'Light Bus', 'MGV', 'Motorcycle', 'PLB GMB', 'Private Car', 'Taxi', 'Van']
B12_WANT = {10: "b12_van", 6: "b12_moto"}

VANSET = r"dataset\VAN.v1-van.yolo26"
HK = r"dataset\hong kong vehicle.v1i.yolo26"
HK_NAMES = ['bus', 'car', 'minibus', 'motorcycle', 'taxi', 'truck', 'van']
HK_MAP = {'bus': [9, 10, 11], 'car': [1], 'minibus': [8, 9], 'motorcycle': [0], 'taxi': [2], 'truck': [4, 5, 6, 7], 'van': [3, 1]}
YW_NAMES = ["van", "minibus", "car", "truck", "bus", "motorcycle", "taxi", "scooter"]

model = YOLO(r"runs\detect\yolo26m_traffic_20260828_0339\weights\best.pt")
yw = YOLO("yolov8s-worldv2.pt")
yw.set_classes(["van", "minibus", "car", "truck", "bus", "motorcycle", "taxi", "scooter"])

rows = []


def find_img(root, split, stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(root, split, "images", stem + ext)
        if os.path.exists(p):
            return p
    return None


def save_crop(im, x1, y1, x2, y2, lib, tag, extra):
    pad = int(0.06 * max(x2 - x1, y2 - y1)) + 2
    H, W = im.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    crop = im[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
        return False
    name = f"{lib}_{tag}.jpg"
    cv2.imwrite(os.path.join(CROPS, lib, name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    rows.append({"lib": lib, "file": name, **extra})
    return True


for split in ["train"]:
    for lf in glob.glob(os.path.join(B12, split, "labels", "*.txt")):
        stem = os.path.splitext(os.path.basename(lf))[0]
        p = find_img(B12, split, stem)
        if p is None:
            continue
        im = cv2.imread(p)
        if im is None:
            continue
        H, W = im.shape[:2]
        with open(lf, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                if c not in B12_WANT:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
                x2, y2 = (xc + w / 2) * W, (yc + h / 2) * H
                bw, bh = x2 - x1, y2 - y1
                m = 0.01 * max(W, H)
                if bw < 28 or bh < 28 or x1 < m or y1 < m or x2 > W - m or y2 > H - m:
                    continue
                if save_crop(im, int(x1), int(y1), int(x2), int(y2), B12_WANT[c], f"{stem[:20]}_{i}", {"src": "batch_12", "w": int(bw), "h": int(bh)}):
                    pass

van_kept = 0
for split in ["train", "valid", "test"]:
    for lf in glob.glob(os.path.join(VANSET, split, "labels", "*.txt")):
        stem = os.path.splitext(os.path.basename(lf))[0]
        p = find_img(VANSET, split, stem)
        if p is None:
            continue
        im = cv2.imread(p)
        if im is None:
            continue
        H, W = im.shape[:2]
        ry = yw.predict(im, conf=0.15, imgsz=640, device=0, verbose=False)[0]
        yd = []
        if ry.boxes is not None and len(ry.boxes):
            for b in ry.boxes:
                yd.append((b.xyxy[0].tolist(), YW_NAMES[int(b.cls[0])], float(b.conf[0])))
        with open(lf, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.split()
                if len(parts) < 5:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
                x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
                bw, bh = x2 - x1, y2 - y1
                if bw < 90 or bh < 70:
                    continue
                ok = False
                for yb, yc_, cf in yd:
                    if yc_ not in ("van", "car", "minibus"):
                        continue
                    ix1, iy1 = max(x1, yb[0]), max(y1, yb[1])
                    ix2, iy2 = min(x2, yb[2]), min(y2, yb[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    a2 = (yb[2] - yb[0]) * (yb[3] - yb[1])
                    if a2 <= 0:
                        continue
                    cov = inter / a2
                    inside = yb[0] > x1 + 0.03 * bw and yb[1] > y1 + 0.03 * bh and yb[2] < x2 - 0.03 * bw and yb[3] < y2 - 0.03 * bh
                    if cov > 0.8 and inside:
                        ok = True
                        break
                if ok and save_crop(im, x1, y1, x2, y2, "van_ext", f"{split}_{stem[:16]}_{i}", {"src": "vandset", "w": bw, "h": bh}):
                    van_kept += 1

hk_stats = {"pass": 0, "suspect": 0, "skip": 0}
hk_tiles = {"suspect": [], "pass": []}
hk_dec = []
for split in ["train", "valid", "test"]:
    for lf in glob.glob(os.path.join(HK, split, "labels", "*.txt")):
        stem = os.path.splitext(os.path.basename(lf))[0]
        p = find_img(HK, split, stem)
        if p is None:
            continue
        im = cv2.imread(p)
        if im is None:
            continue
        H, W = im.shape[:2]
        with open(lf, "r", encoding="utf-8") as f:
            boxes = [l.split() for l in f if len(l.split()) >= 5]
        r = model.predict(im, conf=0.25, imgsz=640, device=0, verbose=False)[0] if boxes else None
        md = []
        if r is not None and r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                md.append((b.xyxy[0].tolist(), int(b.cls[0]), float(b.conf[0])))
        for i, parts in enumerate(boxes):
            hk_cls = HK_NAMES[int(float(parts[0]))]
            xc, yc, w, h = map(float, parts[1:5])
            x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
            x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
            bw, bh = x2 - x1, y2 - y1
            if bw < 30 or bh < 30:
                hk_stats["skip"] += 1
                continue
            best_iou, best = 0.0, None
            for yb, mc, mcf in md:
                ix1, iy1 = max(x1, yb[0]), max(y1, yb[1])
                ix2, iy2 = min(x2, yb[2]), min(y2, yb[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                a1 = max(1, bw * bh)
                iou_v = inter / a1
                if iou_v > best_iou:
                    best_iou, best = iou_v, (mc, mcf)
            cls_ok = best is not None and best[0] in HK_MAP[hk_cls]
            if best is not None and best_iou >= 0.4 and cls_ok and best[1] >= 0.4:
                verdict = "pass"
            else:
                verdict = "suspect"
            hk_stats[verdict] += 1
            det = f"iou={best_iou:.2f} m={'?' if best is None else str(best[0])+' '+format(best[1],'.2f')}"
            hk_dec.append([stem, hk_cls, verdict, det])
            crop = im[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            tile = (crop, f"{hk_cls}|{det}")
            if verdict == "pass" and len(hk_tiles["pass"]) < 48:
                hk_tiles["pass"].append(tile)
            elif verdict == "suspect" and len(hk_tiles["suspect"]) < 96:
                hk_tiles["suspect"].append(tile)
            if verdict == "pass":
                save_crop(im, x1, y1, x2, y2, f"hk_{hk_cls}", f"{split}_{stem[:16]}_{i}", {"src": "hk", "w": bw, "h": bh})

with open(os.path.join(CROPS, "library.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["lib", "file", "src", "w", "h", "yw", "sub"])
    w.writeheader()
    for r2 in rows:
        w.writerow({k: r2.get(k, "") for k in w.fieldnames})
with open(os.path.join(AUD, "decisions.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["frame", "cls", "verdict", "detail"])
    w.writerows(hk_dec)


def grid(tiles, path, cols=8, cell=180):
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
        cv2.putText(g, t[:38], (cc * cell + 3, r * cell + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    cv2.imwrite(path, g, [cv2.IMWRITE_JPEG_QUALITY, 85])


grid(hk_tiles["pass"], os.path.join(AUD, "hk_pass_sample.jpg"))
grid(hk_tiles["suspect"], os.path.join(AUD, "hk_suspect.jpg"))

libs = {}
for r2 in rows:
    libs[r2["lib"]] = libs.get(r2["lib"], 0) + 1
print("crop library:", libs)
print("van_ext kept:", van_kept)
print("hk stats:", hk_stats)
