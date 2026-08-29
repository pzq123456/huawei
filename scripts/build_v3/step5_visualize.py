import csv
import glob
import os
import random

import cv2
import numpy as np

V3 = r"dataset\dataset_merged_v3"
STAGE = r"dataset\_v3_staging"
OUT = r"tmp\crops"
os.makedirs(OUT, exist_ok=True)
random.seed(23)

NAMES = ['Motorcycle', 'Private Car', 'Taxi', 'Light Van', 'LGV', 'MGV', 'HGV', 'Container', 'PLB/GMB', 'Light Bus', 'Franchised Bus', 'Coach']
COLORS = [
    (255, 60, 60), (90, 90, 90), (0, 220, 255), (0, 90, 255), (200, 0, 180), (160, 60, 0),
    (0, 160, 160), (255, 0, 255), (0, 255, 0), (60, 120, 255), (255, 130, 0), (120, 0, 255),
]
CELL_W, CELL_H = 640, 540
COLS = 2


def draw(im, lf, only=None, thick=2, fs=0.55):
    H, W = im.shape[:2]
    for line in open(lf, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        if only is not None and c not in only:
            continue
        xc, yc, w, h = map(float, parts[1:5])
        x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
        x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
        cv2.rectangle(im, (x1, y1), (x2, y2), COLORS[c], thick)
        t = NAMES[c]
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        yy = y1 - 6 if y1 - 6 > th else y2 + th + 4
        cv2.rectangle(im, (x1, yy - th - 4), (x1 + tw + 4, yy + 2), COLORS[c], -1)
        cv2.putText(im, t, (x1 + 2, yy - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)
    return im


def mosaic(items, path, caption=""):
    tiles = []
    for im, tag in items:
        H, W = im.shape[:2]
        s = min((CELL_W - 8) / W, (CELL_H - 30) / H)
        im2 = cv2.resize(im, (int(W * s), int(H * s)))
        cell = np.full((CELL_H, CELL_W, 3), 245, np.uint8)
        cell[0:im2.shape[0], 0:im2.shape[1]] = im2
        cv2.putText(cell, tag, (6, CELL_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        tiles.append(cell)
    while len(tiles) % COLS:
        tiles.append(np.full((CELL_H, CELL_W, 3), 245, np.uint8))
    rows = [np.hstack(tiles[r * COLS:(r + 1) * COLS]) for r in range(len(tiles) // COLS)]
    g = np.vstack(rows)
    if caption:
        bar = np.full((44, g.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        g = np.vstack([bar, g])
    cv2.imwrite(path, g, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("saved", path, g.shape)


def lbl(im_path):
    return os.path.join(V3, "train", "labels", os.path.splitext(os.path.basename(im_path))[0] + ".txt")


rows = list(csv.DictReader(open(os.path.join(STAGE, "paste_manifest.csv"), encoding="utf-8")))
by_out = {}
for r in rows:
    by_out.setdefault(r["out"], []).append(r)

lv_outs = [o for o, rs in by_out.items() if any(x["cls"] == "Light Van" for x in rs)]
random.shuffle(lv_outs)
items = []
for o in lv_outs[:6]:
    p = os.path.join(V3, "train", "images", o)
    im = draw(cv2.imread(p), lbl(p))
    crops = ",".join(sorted({x["crop"].split("_")[0] + ("+hk" if x["crop"].startswith("hk") else "") for x in by_out[o]}))
    items.append((im, f"{o} <- {by_out[o][0]['canvas'][:26]}"))
mosaic(items, os.path.join(OUT, "v3z_paste_lightvan.jpg"), "PASTE: Light Van boosted (b12_van self-crop -> HK canvas), all boxes drawn")

plb_outs = [o for o, rs in by_out.items() if any(x["cls"] in ("PLB/GMB", "Franchised Bus") for x in rs)]
random.shuffle(plb_outs)
items = []
for o in plb_outs[:6]:
    p = os.path.join(V3, "train", "images", o)
    im = cv2.imread(p)
    im = draw(im, lbl(p))
    items.append((im, f"{o} <- {by_out[o][0]['canvas'][:26]}"))
mosaic(items, os.path.join(OUT, "v3z_paste_plb_bus.jpg"), "PASTE: PLB/GMB + Franchised Bus rescues (HK web crops -> canvas)")

moto_imgs = glob.glob(os.path.join(V3, "train", "images", "moto_*.jpg"))
random.shuffle(moto_imgs)
items = []
for p in moto_imgs[:6]:
    im = draw(cv2.imread(p), lbl(p))
    items.append((im, os.path.basename(p)))
mosaic(items, os.path.join(OUT, "v3z_moto_pseudo.jpg"), "MOTO: Vietnam frames, GT + dual-model pseudo boxes (all classes)")

moto_dense = []
for p in moto_imgs:
    n = sum(1 for _ in open(lbl(p), encoding="utf-8"))
    moto_dense.append((n, p))
moto_dense.sort(reverse=True)
items = []
for _, p in moto_dense[:4]:
    im = draw(cv2.imread(p), lbl(p))
    items.append((im, os.path.basename(p)))
mosaic(items, os.path.join(OUT, "v3z_moto_dense.jpg"), "MOTO: densest pseudo-labeled frames (top-4 by box count)")

hk_outs = [o for o, rs in by_out.items() if any(x["crop"].startswith("hk") for x in rs)]
random.shuffle(hk_outs)
items = []
for o in hk_outs[:4]:
    p = os.path.join(V3, "train", "images", o)
    im = draw(cv2.imread(p), lbl(p))
    items.append((im, f"{o} <- {by_out[o][0]['crop'][:30]}"))
if items:
    mosaic(items, os.path.join(OUT, "v3z_paste_hkrescue.jpg"), "PASTE: HK-dataset rescued crops on canvas")
