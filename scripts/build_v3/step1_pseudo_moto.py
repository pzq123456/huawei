import csv
import glob
import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

SRC = r"dataset\Motorcycle.v3-base_no_resize.yolo26"
STAGE = r"dataset\_v3_staging"
OUT_IMG = os.path.join(STAGE, "moto", "images")
OUT_LBL = os.path.join(STAGE, "moto", "labels")
AUD = os.path.join(STAGE, "moto_audit")
for d in [OUT_IMG, OUT_LBL, AUD]:
    os.makedirs(d, exist_ok=True)

CLASSES = ['Motorcycle', 'Private Car', 'Taxi', 'Light Van', 'LGV', 'MGV', 'HGV', 'Container', 'PLB', 'Light Bus', 'Franchised Bus', 'Coach']
YW_NAMES = ["car", "van", "minibus", "truck", "bus", "motorcycle"]
COMPAT = {
    0: {5}, 1: {0, 1, 2, 3}, 2: {0, 1, 2}, 3: {1, 2, 3}, 4: {1, 3}, 5: {1, 3},
    6: {3}, 7: {3}, 8: {2, 4}, 9: {2, 4}, 10: {4}, 11: {4},
}

KEEP_CONF = 0.5
STRONG_CONF = 0.65
MIN_CONF = 0.35
MOTO_ADD_CONF = 0.7

model = YOLO(r"runs\detect\yolo26m_traffic_20260828_0339\weights\best.pt")
yw = YOLO("yolov8s-worldv2.pt")
yw.set_classes(YW_NAMES)


def xyxy_norm(b, W, H):
    return [b[0] / W, b[1] / H, b[2] / W, b[3] / H]


def yolo_line(c, bb, W, H):
    return f"{c} {(bb[0] + bb[2]) / 2 / W:.6f} {(bb[1] + bb[3]) / 2 / H:.6f} {(bb[2] - bb[0]) / W:.6f} {(bb[3] - bb[1]) / H:.6f}"


SIZE_CAP = {0: 0.06, 1: 0.06, 2: 0.06, 3: 0.06, 4: 0.10, 5: 0.10, 6: 0.12, 7: 0.20, 8: 0.12, 9: 0.20, 10: 0.25, 11: 0.25}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (a[2] - a[0]) * (a[3] - a[1])
    a2 = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-6, a1 + a2 - inter)


stats = {"gt_moto": 0, "kept": 0, "audit": 0, "dropped": 0, "moto_addback": 0, "frames": 0}
per_cls = {}
decisions = []
audit_tiles = {"conflict": [], "weak": []}
keep_tiles = []
addback_tiles = []

frames = []
for split in ["train", "valid", "test"]:
    for lf in glob.glob(os.path.join(SRC, split, "labels", "*.txt")):
        stem = os.path.splitext(os.path.basename(lf))[0]
        for ext in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(SRC, split, "images", stem + ext)
            if os.path.exists(p):
                frames.append((p, lf))
                break

for p, lf in frames:
    stem = os.path.splitext(os.path.basename(p))[0]
    im = cv2.imread(p)
    if im is None:
        continue
    H, W = im.shape[:2]
    stats["frames"] += 1

    gt = []
    with open(lf, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                xc, yc, w, h = map(float, parts[1:5])
                gt.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    stats["gt_moto"] += len(gt)

    r = model.predict(im, conf=MIN_CONF, imgsz=640, device=0, verbose=False)[0]
    dets = []
    if r.boxes is not None and len(r.boxes):
        for b in r.boxes:
            dets.append((int(b.cls[0]), float(b.conf[0]), b.xyxy[0].tolist()))
    ry = yw.predict(im, conf=0.12, imgsz=640, device=0, verbose=False)[0]
    yd = []
    if ry.boxes is not None and len(ry.boxes):
        for b in ry.boxes:
            yd.append((b.xyxy[0].tolist(), int(b.cls[0])))

    kept, audit = [], []
    for c, conf, bb in dets:
        if c == 0:
            best = max([iou(bb, g) for g in gt], default=0.0)
            if conf >= MOTO_ADD_CONF and best < 0.25:
                kept.append((c, conf, bb, "yw_addback"))
                stats["moto_addback"] += 1
            continue
        if any(iou(bb, g) >= 0.3 for g in gt):
            continue
        yw_hit = any(iou(bb, yb) >= 0.35 and yc_ in COMPAT.get(c, set()) for yb, yc_ in yd)
        if conf >= KEEP_CONF and (yw_hit or conf >= STRONG_CONF):
            kept.append((c, conf, bb, f"yw{int(yw_hit)}"))
        elif conf >= MIN_CONF:
            audit.append((c, conf, bb, f"yw{int(yw_hit)}"))

    ded = []
    for c, conf, bb, src in sorted(kept, key=lambda t: -t[1]):
        bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        if (bw * bh) / (W * H) > SIZE_CAP.get(c, 0.06) or bw / W > 0.45 or bh / H > 0.65:
            continue
        if all(iou(bb, d[2]) < 0.6 for d in ded):
            ded.append((c, conf, bb, src))
    for c, conf, bb, src in ded:
        xn = xyxy_norm(bb, W, H)
        if src == "yw_addback":
            addback_tiles.append((im.copy(), bb, f"moto {conf:.2f}"))
        elif len(keep_tiles) < 48:
            keep_tiles.append((im.copy(), bb, f"{CLASSES[c]} {conf:.2f} yw={src}"))
    stats["kept"] += len(ded)
    for c, conf, bb, src in audit:
        stats["audit"] += 1
        per_cls[c] = per_cls.get(c, 0) + 1
        bucket = "conflict"
        audit_tiles[bucket].append((im.copy(), bb, f"{CLASSES[c]} {conf:.2f}"))
        decisions.append([stem, CLASSES[c], f"{conf:.3f}", src, bucket])
    stats["dropped"] += len(dets) - len(kept) - len(audit)

    out_lines = [f"{0} " + " ".join(f"{v:.6f}" for v in [(g[0] + g[2]) / 2, (g[1] + g[3]) / 2, g[2] - g[0], g[3] - g[1]]) for g in gt]
    out_lines += [yolo_line(c, bb, W, H) for c, conf, bb, src in ded]
    with open(os.path.join(OUT_LBL, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
    shutil.copy(p, os.path.join(OUT_IMG, os.path.basename(p)))


def render(tile, tag, color):
    im, bb, txt = tile
    H, W = im.shape[:2]
    x1, y1 = max(0, int(bb[0]) - 30), max(0, int(bb[1]) - 30)
    x2, y2 = min(W, int(bb[2]) + 30), min(H, int(bb[3]) + 30)
    crop = im[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None
    cv2.rectangle(crop, (int(max(0, bb[0]) - x1), int(max(0, bb[1]) - y1)), (int(min(W, bb[2]) - x1), int(min(H, bb[3]) - y1)), color, 2)
    s = max(1.0, 160 / max(crop.shape[1], crop.shape[0]))
    crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.putText(crop, txt[:34], (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    return crop


def grid(tiles, path, color, cols=8, cell=170):
    crops = [t for t in (render(x, "", color) for x in tiles) if t is not None]
    if not crops:
        return
    rows = (len(crops) + cols - 1) // cols
    g = np.full((rows * cell, cols * cell, 3), 255, np.uint8)
    for i, c in enumerate(crops):
        h, w = c.shape[:2]
        s = min(cell / w, cell / h)
        c = cv2.resize(c, (max(1, int(w * s)), max(1, int(h * s))))
        r, cc = i // cols, i % cols
        g[r * cell:r * cell + c.shape[0], cc * cell:cc * cell + c.shape[1]] = c
    cv2.imwrite(path, g, [cv2.IMWRITE_JPEG_QUALITY, 85])


grid(keep_tiles[:48], os.path.join(AUD, "auto_keep_sample.jpg"), (0, 160, 0))
grid(addback_tiles[:48], os.path.join(AUD, "moto_addback_sample.jpg"), (255, 0, 0))
for name, tiles in audit_tiles.items():
    grid(tiles[:96], os.path.join(AUD, f"audit_{name}.jpg"), (0, 0, 255))

with open(os.path.join(AUD, "decisions.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["frame", "cls", "conf", "yw", "bucket"])
    w.writerows(decisions)

print("frames:", stats["frames"])
print("gt_moto:", stats["gt_moto"])
print("pseudo kept:", stats["kept"], "| moto_addback:", stats["moto_addback"])
print("audit:", stats["audit"], dict(sorted(per_cls.items())))
print("dropped:", stats["dropped"])
