import glob
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
DATA = "dataset/batch_12.v10i.merge8.yolov11"
NAMES = ["Bus", "Franchised Bus", "Truck", "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van"]
CAR = 5
CONF_EVAL = 0.25
CONF_PRED = 0.05
IMGSZ = 640
MATCH_IOU = 0.5
OVERLAP_IOU = 0.1
CROP_DIR = "runs/car_fp_audit"
DEVICE = 2
W = "runs/detect/yolo26m_merge8_20260902_0916/weights/best.pt"

SAVE_BG = {"train": 150, "valid": 10**9, "test": 10**9}
SAVE_OVL = {"train": 60, "valid": 60, "test": 60}
SAVE_MISC = {"train": 30, "valid": 60, "test": 60}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def save_crop(img, box, conf, tag, split, stem, idx, gt_note=""):
    H, W_ = img.shape[:2]
    x1, y1, x2, y2 = box
    mw, mh = 0.4 * (x2 - x1), 0.4 * (y2 - y1)
    x1, y1 = max(0, int(x1 - mw)), max(0, int(y1 - mh))
    x2, y2 = min(W_, int(x2 + mw)), min(H, int(y2 + mh))
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    crop = img[y1:y2, x1:x2].copy()
    cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (0, 0, 255), 2)
    label = f"Car {conf:.2f} {tag} {gt_note}".strip()
    cv2.putText(crop, label, (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    fn = f"{CROP_DIR}/{split}/{tag}/{split}_{stem}_c{conf:.2f}_{idx}.jpg"
    cv2.imwrite(fn, crop)
    return fn


def main():
    model = YOLO(W)
    report, fp_records = {}, []
    for split in ["train", "valid", "test"]:
        os.makedirs(f"{CROP_DIR}/{split}/bg_fp", exist_ok=True)
        os.makedirs(f"{CROP_DIR}/{split}/ovl_fp", exist_ok=True)
        os.makedirs(f"{CROP_DIR}/{split}/misclass", exist_ok=True)
        saved = dict(bg=0, ovl=0, misc=0)
        stats = dict(n_img=0, car_fp=0, car_fp_by_overlap={}, car_fp_bg_area=[], misclass={}, fn=0, fp_all={})
        bg_cand, ovl_cand, misc_cand = [], [], []
        for img_path in sorted(glob.glob(f"{DATA}/{split}/images/*")):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            gts = []
            lbl = f"{DATA}/{split}/labels/{stem}.txt"
            if os.path.isfile(lbl):
                with open(lbl) as f:
                    for line in f:
                        p = line.split()
                        if len(p) >= 5:
                            gts.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
            r = model.predict(img_path, conf=CONF_PRED, imgsz=IMGSZ, verbose=False, device=DEVICE)[0]
            H, W_ = r.orig_img.shape[:2]
            dets = [(int(c), float(cf), *map(float, xy)) for c, cf, xy in
                    zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
            dets25 = [d for d in dets if d[1] >= CONF_EVAL]
            stats["n_img"] += 1
            used = [False] * len(dets25)
            gt_boxes = [(c, ((gx - gw / 2) * W_, (gy - gh / 2) * H, (gx + gw / 2) * W_, (gy + gh / 2) * H)) for c, gx, gy, gw, gh in gts]
            for c, gx, gy, gw, gh in gts:
                box = ((gx - gw / 2) * W_, (gy - gh / 2) * H, (gx + gw / 2) * W_, (gy + gh / 2) * H)
                best, biou = None, MATCH_IOU
                for j, d in enumerate(dets25):
                    if used[j]:
                        continue
                    v = iou(box, d[2:])
                    if v > biou:
                        best, biou = j, v
                if best is None:
                    stats["fn"] += 1
                else:
                    used[best] = True
                    dc, dcf = dets25[best][0], dets25[best][1]
                    if dc != c:
                        key = f"{NAMES[c]}->{NAMES[dc]}"
                        stats["misclass"][key] = stats["misclass"].get(key, 0) + 1
                        if dc == CAR and c in (7, 2, 0):
                            misc_cand.append(dict(conf=dcf, box=dets25[best][2:], stem=stem,
                                                  gt=NAMES[c], note=f"GT:{NAMES[c]} iou={biou:.2f}"))
            for j, d in enumerate(dets25):
                if not used[j]:
                    dc, dcf, *box = d
                    stats["fp_all"][NAMES[dc]] = stats["fp_all"].get(NAMES[dc], 0) + 1
                    if dc != CAR:
                        continue
                    stats["car_fp"] += 1
                    bo, bio = None, OVERLAP_IOU
                    for gc, gb in gt_boxes:
                        v = iou(box, gb)
                        if v > bio:
                            bo, bio = gc, v
                    key = f"overlap:{NAMES[bo]}@{bio:.1f}" if bo is not None else "background"
                    if bo is None:
                        key = "background"
                    elif bo == CAR:
                        key = "overlap:Car(near-miss)"
                    else:
                        key = f"overlap:{NAMES[bo]}"
                    stats["car_fp_by_overlap"][key] = stats["car_fp_by_overlap"].get(key, 0) + 1
                    a = ((box[2] - box[0]) * (box[3] - box[1])) / (H * W_)
                    rec = dict(split=split, img=stem, conf=round(dcf, 4), box=[round(v, 1) for v in box],
                               area_frac=round(a, 5), what=key)
                    fp_records.append(rec)
                    if bo is None:
                        stats["car_fp_bg_area"].append(a)
                        bg_cand.append(dict(conf=dcf, box=box, stem=stem, gt="", note=f"area={a:.4f}"))
                    else:
                        ovl_cand.append(dict(conf=dcf, box=box, stem=stem, gt=NAMES[bo], note=f"GT:{NAMES[bo]} iou={bio:.2f}"))
        bg_cand.sort(key=lambda r: -r["conf"])
        ovl_cand.sort(key=lambda r: -r["conf"])
        misc_cand.sort(key=lambda r: -r["conf"])
        stats["car_fp_bg_area"] = dict(n=len(stats["car_fp_bg_area"]),
                                       median=round(float(np.median(stats["car_fp_bg_area"])), 5) if stats["car_fp_bg_area"] else None,
                                       p75=round(float(np.percentile(stats["car_fp_bg_area"], 75)), 5) if stats["car_fp_bg_area"] else None,
                                       p95=round(float(np.percentile(stats["car_fp_bg_area"], 95)), 5) if stats["car_fp_bg_area"] else None)
        for kind, cands, limit, tag in [("bg", bg_cand, SAVE_BG[split], "bg_fp"),
                                        ("ovl", ovl_cand, SAVE_OVL[split], "ovl_fp"),
                                        ("misc", misc_cand, SAVE_MISC[split], "misclass")]:
            img_cache = {}
            for i, rec in enumerate(cands[:limit]):
                if rec["stem"] not in img_cache:
                    img_cache[rec["stem"]] = cv2.imread(f"{DATA}/{split}/images/{rec['stem']}.jpg")
                    if img_cache[rec["stem"]] is None:
                        for ext in (".png", ".jpeg"):
                            im = cv2.imread(f"{DATA}/{split}/images/{rec['stem']}{ext}")
                            if im is not None:
                                img_cache[rec["stem"]] = im
                                break
                im = img_cache[rec["stem"]]
                if im is None:
                    continue
                if save_crop(im, rec["box"], rec["conf"], tag, split, rec["stem"], i, rec["note"]):
                    saved[kind] += 1
        stats["saved_crops"] = saved
        report[split] = stats
        print(f"\n===== {split} ({stats['n_img']} imgs) =====")
        print(f"Car FP (conf>=0.25, IoU0.5 unmatched): {stats['car_fp']}")
        for k, v in sorted(stats["car_fp_by_overlap"].items(), key=lambda kv: -kv[1]):
            print(f"   {k:28s} {v:4d} ({100*v/max(1,stats['car_fp']):5.1f}%)")
        print(f"   bg FP area: {stats['car_fp_bg_area']}")
        print(f"all-class FP breakdown: {stats['fp_all']}")
        print(f"misclass (IoU0.5 matched, wrong cls): {stats['misclass']}")
        print(f"FN total: {stats['fn']}   crops saved: {saved}")
    with open("/tmp/opencode/car_fp_audit.json", "w") as f:
        json.dump(dict(report=report, fp_records=fp_records), f, indent=1)
    print(f"\nsaved /tmp/opencode/car_fp_audit.json ({len(fp_records)} FP records); crops in {CROP_DIR}/")


if __name__ == "__main__":
    main()
