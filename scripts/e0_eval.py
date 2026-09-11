import collections
import glob
import json
import os

import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
DATA = "dataset/batch_12.v10i.merge8.yolov11/data.yaml"
NAMES = ["Bus", "Franchised Bus", "Truck", "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van"]
nc = 8
W = "runs/detect/yolo26m_merge8_20260902_0916/weights/best.pt"
BUCKETS = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 2.0)]
BUCKET_NAMES = ["<0.01", "0.01-0.02", "0.02-0.05", ">0.05"]
CONF_EVAL = 0.25   # confusion-matrix operating point
CONF_PRED = 0.05   # keep low-conf dets, filter during matching


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def run_split(model, split):
    cm = np.zeros((nc, nc), dtype=int)          # [gt, pred]
    fp = np.zeros(nc, dtype=int)                # background -> class (unmatched dets)
    fn = np.zeros(nc, dtype=int)                # unmatched gt
    gt_conf_ok, wrong_car_conf = [], []
    van_buckets = [dict(n=0, ok=0, to_car=0, to_other=0, miss=0, miss_conf=[], car_conf=[]) for _ in BUCKETS]
    n_img = 0
    for img in sorted(glob.glob(f"dataset/batch_12.v10i.merge8.yolov11/{split}/images/*")):
        stem = os.path.splitext(os.path.basename(img))[0]
        gts = []
        with open(f"dataset/batch_12.v10i.merge8.yolov11/{split}/labels/{stem}.txt") as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    gts.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        r = model.predict(img, conf=CONF_PRED, imgsz=640, verbose=False)[0]
        H_, W_ = r.orig_img.shape[:2]
        dets = []
        for c, cf, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            if cf >= CONF_EVAL:
                dets.append((int(c), float(cf), float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])))
        n_img += 1
        used = [False] * len(dets)
        for c, gx, gy, gw, gh in gts:
            box = ((gx - gw / 2) * W_, (gy - gh / 2) * H_, (gx + gw / 2) * W_, (gy + gh / 2) * H_)
            best, biou = None, 0.5
            for j, d in enumerate(dets):
                if used[j]:
                    continue
                v = iou(box, d[2:])
                if v > biou:
                    best, biou = j, v
            if best is None:
                fn[c] += 1
                if c == 7:
                    a = gw * gh
                    bi = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= a < hi)
                    van_buckets[bi]["miss"] += 1
                    van_buckets[bi]["miss_conf"].append(0.0)
            else:
                used[best] = True
                dc, dcf = dets[best][0], dets[best][1]
                cm[c, dc] += 1
                if dc == c:
                    gt_conf_ok.append(dcf)
                if c == 7:
                    a = gw * gh
                    bi = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= a < hi)
                    s = van_buckets[bi]
                    if dc == 7:
                        s["ok"] += 1
                    else:
                        s["to_other"] += 1
                        if dc == 5:
                            s["to_car"] += 1
                            s["car_conf"].append(dcf)
        for j, d in enumerate(dets):
            if not used[j]:
                fp[d[0]] += 1
    return cm, fp, fn, van_buckets, n_img


model = YOLO(W)
report = {}
for split in ["valid", "test"]:
    cm, fp, fn, vb, n_img = run_split(model, split)
    print(f"\n===== E0 (0916 best.pt) on {split}  [{n_img} images, conf>={CONF_EVAL}, IoU 0.5] =====")
    print(f"{'class':16s} {'gt':>5s} {'TP':>5s} {'FN':>5s} {'FP(bg)':>6s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    rows = {}
    for c in range(nc):
        tp = cm[c, c]
        p = tp / max(1, tp + fp[c])
        rr = tp / max(1, tp + fn[c])
        f1 = 2 * p * rr / max(1e-9, p + rr)
        rows[NAMES[c]] = dict(gt=int(cm[c].sum()), tp=int(tp), fn=int(fn[c]), fp=int(fp[c]),
                              P=round(p, 4), R=round(rr, 4), F1=round(f1, 4))
        print(f"{NAMES[c]:16s} {int(cm[c].sum()):5d} {tp:5d} {fn[c]:5d} {fp[c]:6d} {p:6.3f} {rr:6.3f} {f1:6.3f}")
    van_gt = int(cm[7].sum())
    print(f"\nVan->Car: {cm[7,5]}/{van_gt} = {100*cm[7,5]/max(1,van_gt):.1f}%   background->Car: {fp[5]}")
    print("\nVan size buckets:")
    for i, s in enumerate(vb):
        mcc = np.mean(s["car_conf"]) if s["car_conf"] else float("nan")
        print(f"  {BUCKET_NAMES[i]:>10s}: n={s['n']:4d} ok={s['ok']:4d} Van->Car={s['to_car']:3d} "
              f"({100*s['to_car']/max(1,s['n']):5.1f}%) ->other={s['to_other']:3d} missed={s['miss']:3d} wrongCarConf={mcc:.3f}")
    report[split] = dict(cm=cm.tolist(), fp=fp.tolist(), fn=fn.tolist(), per_class=rows, van_buckets=vb,
                         n_img=n_img)

with open("/tmp/opencode/e0_confusion.json", "w") as f:
    json.dump(report, f, indent=1, default=float)
print("\nsaved /tmp/opencode/e0_confusion.json")
