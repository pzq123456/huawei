import argparse
import glob
import json
import os
import re
import shutil
from collections import Counter, defaultdict

import cv2
import yaml
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
DATA = "dataset/batch_12.v10i.merge8.yolov11"
DST = "dataset/batch_12.v13i.merge8.plrepair2.yolov11"
W_E0 = "runs/detect/yolo26m_merge8_20260902_0916/weights/best.pt"
W_V2X = "runs/detect/yolo26m_van2x_20260908_0030/weights/best.pt"
AUDIT = "/tmp/opencode/car_fp_audit.json"
NAMES = ["Bus", "Franchised Bus", "Truck", "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van"]
DEVICE = 2
CONF_PRED = 0.6
GT_IOU = 0.5
A_IOU = 0.7
A_CONF = 0.7
B_CONF = 0.85


def src_key(stem):
    s = stem.split("_jpg")[0]
    m = re.match(r"^(ANMR\d+|Motorcycle_\d+_ANMR\d+|frame_\d{8})_", s)
    if m:
        return m.group(1)
    return s.rsplit("_frame", 1)[0].rsplit("_f0", 1)[0][:24]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def load_gt(path):
    gts = []
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    gts.append([float(v) for v in p[:5]])
    return gts


def main():
    global DST, DATA, W_E0, W_V2X, AUDIT, NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default="dataset/batch_12.v13i.merge8.plrepair2.yolov11")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--weights", nargs=2, metavar=("E0", "AUX"), default=[W_E0, W_V2X])
    ap.add_argument("--audit", default=AUDIT)
    ap.add_argument("--ratio", type=float, default=None, help="min bg-Car-FP ratio; default = v1 conservative rule")
    ap.add_argument("--min-bgfp", type=int, default=None)
    args = ap.parse_args()
    DST, DATA, W_E0, W_V2X, AUDIT = args.dst, args.data, args.weights[0], args.weights[1], args.audit
    with open(f"{DATA}/data.yaml") as f:
        NAMES = yaml.safe_load(f)["names"]
    if isinstance(NAMES, dict):
        NAMES = [NAMES[k] for k in sorted(NAMES)]
    # 1. scope
    d = json.load(open(AUDIT))
    bg = Counter()
    for r in d["fp_records"]:
        if r["split"] == "train" and r["what"] == "background":
            bg[src_key(r["img"])] += 1
    all_imgs = Counter()
    for f in glob.glob(f"{DATA}/train/images/*"):
        all_imgs[src_key(os.path.basename(f).rsplit(".rf", 1)[0])] += 1
    if args.ratio is not None:
        scope = {k for k in all_imgs if bg[k] >= args.min_bgfp and bg[k] / max(1, all_imgs[k]) >= args.ratio}
    else:
        scope = {k for k in all_imgs
                 if (bg[k] >= 5 and bg[k] / max(1, all_imgs[k]) >= 0.20) or (bg[k] >= 20 and bg[k] / max(1, all_imgs[k]) >= 0.12)}
    scope_files = sorted(f for f in glob.glob(f"{DATA}/train/images/*")
                         if src_key(os.path.basename(f).rsplit(".rf", 1)[0]) in scope)
    print(f"scope: {len(scope)} sources, {len(scope_files)} train images")

    # 2. mine with both models
    m_e0, m_v2 = YOLO(W_E0), YOLO(W_V2X)
    provenance, stats = {}, dict(tierA=0, tierB=0, imgs_touched=0, per_class=Counter(), per_tier_src=Counter())
    for i, img_path in enumerate(scope_files):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        H, W_ = cv2.imread(img_path).shape[:2]
        gt = load_gt(f"{DATA}/train/labels/{stem}.txt")
        gt_xy = [((g[1] - g[3] / 2) * W_, (g[2] - g[4] / 2) * H, (g[1] + g[3] / 2) * W_, (g[2] + g[4] / 2) * H) for g in gt]

        def cand(model):
            r = model.predict(img_path, conf=CONF_PRED, imgsz=640, verbose=False, device=DEVICE)[0]
            out = []
            for c, cf, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                box = tuple(float(v) for v in xy)
                if max((iou(box, g) for g in gt_xy), default=0.0) < GT_IOU:
                    out.append((int(c), float(cf), box))
            return out

        c_e0, c_v2 = cand(m_e0), cand(m_v2)
        used_v2 = [False] * len(c_v2)
        inject = []
        for c, cf, box in sorted(c_e0, key=lambda x: -x[1]):
            best, bio = None, A_IOU
            if cf >= A_CONF:
                for j, (c2, cf2, b2) in enumerate(c_v2):
                    if used_v2[j] or c2 != c:
                        continue
                    v = iou(box, b2)
                    if v > bio:
                        best, bio = j, v
            if best is not None:
                used_v2[best] = True
                inject.append(dict(cls=c, box=box, conf=round((cf + c_v2[best][1]) / 2, 4),
                                   tier="A", conf_e0=round(cf, 4), conf_v2x=round(c_v2[best][1], 4)))
            elif cf >= B_CONF:
                if all(iou(box, p["box"]) < A_IOU for p in inject):
                    inject.append(dict(cls=c, box=box, conf=round(cf, 4), tier="B", conf_e0=round(cf, 4), conf_v2x=None))
        if inject:
            provenance[stem] = inject
            stats["imgs_touched"] += 1
            for p in inject:
                stats[f"tier{p['tier']}"] += 1
                stats["per_class"][NAMES[p["cls"]]] += 1
        if (i + 1) % 500 == 0:
            print(f"  mined {i + 1}/{len(scope_files)}, tierA={stats['tierA']} tierB={stats['tierB']}")

    # 3. build v12
    if os.path.exists(DST):
        raise SystemExit(f"target exists: {DST}")
    for split in ["train", "valid", "test"]:
        os.makedirs(f"{DST}/{split}/images")
        os.makedirs(f"{DST}/{split}/labels")
        for f in glob.glob(f"{DATA}/{split}/images/*"):
            os.link(f, f"{DST}/{split}/images/{os.path.basename(f)}")
        for f in glob.glob(f"{DATA}/{split}/labels/*.txt"):
            shutil.copy2(f, f"{DST}/{split}/labels/{os.path.basename(f)}")
    for stem, inject in provenance.items():
        with open(f"{DST}/train/labels/{stem}.txt", "a") as f:
            for p in inject:
                x1, y1, x2, y2 = p["box"]
                f.write(f"{p['cls']} {(x1 + x2) / 2 / W_:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W_:.6f} {(y2 - y1) / H:.6f}\n")
    shutil.copy2(f"{DATA}/data.yaml", f"{DST}/data.yaml")
    with open(f"{DST}/pseudo_provenance.json", "w") as f:
        json.dump(dict(
            models=dict(e0=W_E0, aux=W_V2X),
            rules=dict(tierA=dict(consensus=True, conf=A_CONF, iou=A_IOU),
                       tierB=dict(e0_conf=B_CONF, gt_iou_lt=GT_IOU),
                       discarded="everything else (tier C)"),
            scope_sources=sorted(scope), scope_images=len(scope_files),
            injected=provenance), f, indent=1)

    # 4. report
    print(f"\nimages touched: {stats['imgs_touched']} / {len(scope_files)} in scope")
    print(f"tier A (consensus): {stats['tierA']}   tier B (E0>=0.85): {stats['tierB']}")
    print("per-class injected:", dict(stats["per_class"]))
    src_cnt = Counter()
    src_cls = defaultdict(Counter)
    for stem, inj in provenance.items():
        k = src_key(stem)
        src_cnt[k] += len(inj)
        for p in inj:
            src_cls[k][NAMES[p["cls"]]] += 1
    print("top sources by injected boxes:", src_cnt.most_common(8))
    print("\nsource x pseudo-class crosstab (wholesale-deletion check, top 12 sources):")
    for k, _ in src_cnt.most_common(12):
        print(f"  {k:30s} " + " ".join(f"{c}:{n}" for c, n in src_cls[k].most_common()))
    print(f"\ndone: {DST}  (provenance: pseudo_provenance.json, original GT untouched)")


if __name__ == "__main__":
    main()
