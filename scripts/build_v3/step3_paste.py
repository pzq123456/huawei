import csv
import glob
import os
import random

import cv2
import numpy as np
from ultralytics import YOLO

random.seed(42)

STAGE = r"dataset\_v3_staging"
V2 = r"dataset\dataset_merged_v2"
HK = r"dataset\hong kong vehicle.v1i.yolo26"
CROPS = os.path.join(STAGE, "crops")
PASTE_IMG = os.path.join(STAGE, "pasted", "images")
PASTE_LBL = os.path.join(STAGE, "pasted", "labels")
os.makedirs(PASTE_IMG, exist_ok=True)
os.makedirs(PASTE_LBL, exist_ok=True)

NAMES = ['Motorcycle', 'Private Car', 'Taxi', 'Light Van', 'LGV', 'MGV', 'HGV', 'Container', 'Public Light Bus/GMB', 'Light Bus', 'Franchised Bus', 'Coach']
HK_NAMES = ['bus', 'car', 'minibus', 'motorcycle', 'taxi', 'truck', 'van']
HK_MAP = {'bus': [9, 10, 11], 'car': [1], 'minibus': [8, 9], 'motorcycle': [0], 'taxi': [2], 'truck': [4, 5, 6, 7], 'van': [3, 1]}

MINIBUS_OK = {3, 142, 243, 247, 252, 279, 290, 388, 289}
VAN_OK = {70, 104, 139, 267, 360, 402, 468}
TAXI_OK = {481}

model = YOLO(r"runs\detect\yolo26m_traffic_20260828_0339\weights\best.pt")

items = []


def add_item(crop, cls, scale_mode, source, count):
    items.append({"crop": crop, "cls": cls, "scale_mode": scale_mode, "source": source, "count": count})


def find_img(split, stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(HK, split, "images", stem + ext)
        if os.path.exists(p):
            return p
    return None


sidx = {int(r["idx"]): (r["frame"], int(r["box_line"]), r["cls"]) for r in csv.DictReader(open(os.path.join(STAGE, "hk_audit", "suspects_index.csv"), encoding="utf-8"))}

hk_pass_m = {}
for row in csv.DictReader(open(os.path.join(STAGE, "hk_audit", "decisions.csv"), encoding="utf-8")):
    if row["verdict"] == "pass":
        d = row["detail"]
        try:
            m = int(d.split("m=")[1].split()[0])
        except (IndexError, ValueError):
            m = -1
        hk_pass_m[(row["frame"], row["cls"])] = m

for split in ["train", "valid", "test"]:
    for lf in glob.glob(os.path.join(HK, split, "labels", "*.txt")):
        stem = os.path.splitext(os.path.basename(lf))[0]
        p = find_img(split, stem)
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
                continue
            best_iou, best = 0.0, None
            for yb, mc, mcf in md:
                ix1, iy1 = max(x1, yb[0]), max(y1, yb[1])
                ix2, iy2 = min(x2, yb[2]), min(y2, yb[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                iou_v = inter / max(1, bw * bh)
                if iou_v > best_iou:
                    best_iou, best = iou_v, (mc, mcf)
            cls_ok = best is not None and best[0] in HK_MAP[hk_cls]
            passed = best is not None and best_iou >= 0.4 and cls_ok and best[1] >= 0.4
            pad = int(0.06 * max(bw, bh)) + 2
            crop = im[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
            if crop.size == 0:
                continue
            sidx_key = [k for k, v in sidx.items() if v[0] == stem and v[1] == i]
            rescued = sidx_key and sidx_key[0] in (MINIBUS_OK | VAN_OK | TAXI_OK)
            if passed:
                if hk_cls == "minibus":
                    add_item(crop, 8, "median", f"hk_{stem}_{i}", 3)
                elif hk_cls == "van":
                    m = hk_pass_m.get((stem, "van"), -1)
                    if m in (3, 4):
                        add_item(crop, 3, "median", f"hk_{stem}_{i}", 2)
                elif hk_cls == "bus":
                    add_item(crop, 10, "median", f"hk_{stem}_{i}", 2)
                elif hk_cls == "taxi":
                    add_item(crop, 2, "median", f"hk_{stem}_{i}", 1)
            elif rescued:
                k0 = sidx_key[0]
                if k0 in MINIBUS_OK:
                    add_item(crop, 8, "median", f"hkres_{stem}_{i}", 3)
                elif k0 in VAN_OK:
                    add_item(crop, 3, "median", f"hkres_{stem}_{i}", 2)
                elif k0 in TAXI_OK:
                    add_item(crop, 2, "median", f"hkres_{stem}_{i}", 2)
            elif hk_cls == "taxi" and best is not None and best[0] == 1 and best_iou >= 0.5 and best[1] >= 0.5:
                add_item(crop, 2, "median", f"hkres_{stem}_{i}", 1)

for f in glob.glob(os.path.join(CROPS, "b12_van", "*.jpg")):
    add_item(cv2.imread(f), 3, "native", f"b12_{os.path.basename(f)}", 1)
for f in glob.glob(os.path.join(CROPS, "b12_moto", "*.jpg")):
    add_item(cv2.imread(f), 0, "native", f"b12_{os.path.basename(f)}", 2)

median_h = {}
hnorm = {}
for lf in glob.glob(os.path.join(V2, "train", "labels", "*.txt")):
    with open(lf, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                hnorm.setdefault(int(float(parts[0])), []).append(float(parts[4]))
for c, arr in hnorm.items():
    median_h[c] = float(np.median(arr))
print("median h_norm:", {NAMES[c]: round(v, 4) for c, v in sorted(median_h.items())})

canvases = []
all_stems = [os.path.splitext(os.path.basename(lf))[0] for lf in glob.glob(os.path.join(V2, "train", "labels", "*.txt"))]


def has_black_patch(stem):
    p = None
    for ext in [".jpg", ".jpeg", ".png"]:
        q = os.path.join(V2, "train", "images", stem + ext)
        if os.path.exists(q):
            p = q
            break
    if p is None:
        return True
    im = cv2.imread(p)
    if im is None:
        return True
    im = cv2.resize(im, (320, 320))
    mask = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 6).astype(np.uint8)
    if mask.sum() == 0:
        return False
    n, lab, stats_, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        if stats_[i, cv2.CC_STAT_AREA] > 0.004 * 320 * 320:
            return True
    return False


random.shuffle(all_stems)
canvases = [(s, True) for s in all_stems if not has_black_patch(s)]
print("clean canvases:", len(canvases), "/", len(all_stems))
canvas_pool = canvases

canvas_cache = {}


def load_canvas(stem):
    if stem not in canvas_cache:
        p = None
        for ext in [".jpg", ".jpeg", ".png"]:
            q = os.path.join(V2, "train", "images", stem + ext)
            if os.path.exists(q):
                p = q
                break
        if p is None:
            return None, None, None
        im = cv2.imread(p)
        boxes = []
        with open(os.path.join(V2, "train", "labels", stem + ".txt"), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    c = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                    boxes.append((c, xc, yc, w, h))
        canvas_cache[stem] = (im, boxes, p)
    return canvas_cache[stem]


manifest = []
op_count = 0
img_count = 0
ci = 0
random.shuffle(items)
for it in items:
    for rep in range(it["count"]):
        tries = 0
        placed = False
        while tries < 12 and not placed:
            tries += 1
            if ci >= len(canvas_pool):
                break
            stem, is_bg = canvas_pool[ci]
            im, boxes, p = load_canvas(stem)
            if im is None:
                ci += 1
                continue
            H, W = im.shape[:2]
            crop = it["crop"]
            ch, cw = crop.shape[:2]
            s = random.uniform(0.7, 1.4) * (median_h.get(it["cls"], 0.06) * H) / ch
            nw, nh = max(12, int(cw * s)), max(12, int(ch * s))
            if nw >= W - 4 or nh >= H - 4:
                ci += 1
                continue
            if random.random() < 0.5:
                crop = cv2.flip(crop, 1)
            crop_j = crop.copy()
            hsv = cv2.cvtColor(crop_j, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(-6, 6), 0, 179)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.85, 1.15), 0, 255)
            crop_j = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            crop_j = cv2.resize(crop_j, (nw, nh), interpolation=cv2.INTER_AREA)
            for _ in range(8):
                px = random.randint(2, max(2, W - nw - 2))
                py = random.randint(2, max(2, H - nh - 2))
                bx1, by1, bx2, by2 = px, py, px + nw, py + nh
                ok = True
                for c, xc, yc, w2, h2 in boxes:
                    ox1, oy1 = (xc - w2 / 2) * W, (yc - h2 / 2) * H
                    ox2, oy2 = (xc + w2 / 2) * W, (yc + h2 / 2) * H
                    ix1, iy1 = max(bx1, ox1), max(by1, oy1)
                    ix2, iy2 = min(bx2, ox2), min(by2, oy2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    a2 = (ox2 - ox1) * (oy2 - oy1)
                    if inter / max(1e-6, a2) > 0.25:
                        ok = False
                        break
                if ok:
                    break
            if not ok:
                continue
            feather = np.ones((nh, nw), np.float32)
            b = max(3, int(0.12 * min(nw, nh)))
            feather[:b, :] *= np.linspace(0, 1, b)[:, None]
            feather[-b:, :] *= np.linspace(1, 0, b)[:, None]
            feather[:, :b] *= np.linspace(0, 1, b)[None, :]
            feather[:, -b:] *= np.linspace(1, 0, b)[None, :]
            feather = cv2.GaussianBlur(feather, (0, 0), 2)[..., None]
            region = im[by1:by2, bx1:bx2].astype(np.float32)
            im[by1:by2, bx1:bx2] = (crop_j.astype(np.float32) * feather + region * (1 - feather)).astype(np.uint8)
            boxes.append((it["cls"], (bx1 + bx2) / 2 / W, (by1 + by2) / 2 / H, nw / W, nh / H))
            manifest.append({"out": f"paste_{img_count:06d}.jpg", "canvas": stem, "crop": it["source"], "cls": NAMES[it["cls"]], "xyxy": f"{bx1},{by1},{bx2},{by2}"})
            placed = True
            op_count += 1
        if placed:
            out_im = canvas_cache[stem][0].copy()
            out_boxes = list(canvas_cache[stem][1])
            canvas_cache.pop(stem, None)
            cv2.imwrite(os.path.join(PASTE_IMG, f"paste_{img_count:06d}.jpg"), out_im, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(os.path.join(PASTE_LBL, f"paste_{img_count:06d}.txt"), "w", encoding="utf-8") as f:
                for c, xc, yc, w2, h2 in out_boxes:
                    f.write(f"{c} {xc:.6f} {yc:.6f} {w2:.6f} {h2:.6f}\n")
            img_count += 1
            ci += 1

with open(os.path.join(STAGE, "paste_manifest.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["out", "canvas", "crop", "cls", "xyxy"])
    w.writeheader()
    w.writerows(manifest)

cls_count = {}
for m in manifest:
    cls_count[m["cls"]] = cls_count.get(m["cls"], 0) + 1
print("paste ops:", op_count, "| pasted images:", img_count)
print("per class:", cls_count)
