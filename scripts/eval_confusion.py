import argparse
import glob
import json
import os

import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
DATA = "dataset/batch_12.v10i.merge8.yolov11/data.yaml"
import yaml

_ym = yaml.safe_load(open(DATA))
NAMES = _ym["names"]
if isinstance(NAMES, dict):
    NAMES = [NAMES[k] for k in sorted(NAMES)]
nc = len(NAMES)
VAN_ID = NAMES.index("Van")
CAR_ID = NAMES.index("Private Car")
BUCKETS = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 2.0)]
BUCKET_NAMES = ["<0.01", "0.01-0.02", "0.02-0.05", ">0.05"]
CONF_EVAL = 0.25
CONF_PRED = 0.05
IMGSZ = 640  # default, overridden by --imgsz


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def run_split(model, split):
    cm = np.zeros((nc, nc), dtype=int)
    fp = np.zeros(nc, dtype=int)
    fn = np.zeros(nc, dtype=int)
    ok_conf = [dict() for _ in range(nc)]
    van_buckets = [dict(n=0, ok=0, to_car=0, to_other=0, miss=0, car_conf=[]) for _ in BUCKETS]
    n_img = 0
    for img in sorted(glob.glob(f"dataset/batch_12.v10i.merge8.yolov11/{split}/images/*")):
        stem = os.path.splitext(os.path.basename(img))[0]
        gts = []
        with open(f"dataset/batch_12.v10i.merge8.yolov11/{split}/labels/{stem}.txt") as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    gts.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        r = model.predict(img, conf=CONF_PRED, imgsz=IMGSZ, verbose=False, device=DEVICE)[0]
        H_, W_ = r.orig_img.shape[:2]
        dets = []
        for c, cf, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            if cf >= CONF_EVAL:
                dets.append((int(c), float(cf), float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])))
        n_img += 1
        used = [False] * len(dets)
        for c, gx, gy, gw, gh in gts:
            box = ((gx - gw / 2) * W_, (gy - gh / 2) * H_, (gx + gw / 2) * W_, (gy + gh / 2) * W_)
            best, biou = None, 0.5
            for j, d in enumerate(dets):
                if used[j]:
                    continue
                v = iou(box, d[2:])
                if v > biou:
                    best, biou = j, v
            if best is None:
                fn[c] += 1
                if c == VAN_ID:
                    a = gw * gh
                    bi = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= a < hi)
                    van_buckets[bi]["miss"] += 1
            else:
                used[best] = True
                dc, dcf = dets[best][0], dets[best][1]
                cm[c, dc] += 1
                if dc == c:
                    ok_conf[c][dcf] = ok_conf[c].get(dcf, 0) + 1
                if c == VAN_ID:
                    a = gw * gh
                    bi = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= a < hi)
                    s = van_buckets[bi]
                    s["n"] += 1
                    if dc == VAN_ID:
                        s["ok"] += 1
                    else:
                        s["to_other"] += 1
                        if dc == CAR_ID:
                            s["to_car"] += 1
                            s["car_conf"].append(dcf)
        for j, d in enumerate(dets):
            if not used[j]:
                fp[d[0]] += 1
    return cm, fp, fn, van_buckets, ok_conf, n_img


def conf_summary(d):
    tot = sum(d.values())
    if tot == 0:
        return float("nan")
    return sum(k * v for k, v in d.items()) / tot


def evaluate(tag, weights):
    model = YOLO(weights)
    report = dict(weights=weights)
    for split, split_name in [("valid", "val"), ("test", "test")]:
        m = model.val(data=DATA, split=split_name, imgsz=IMGSZ, device=DEVICE, verbose=False, name=f"evalconf_{tag}_{split}")
        ap50 = m.box.ap50
        report[split] = dict(
            mAP50=float(m.box.map50), mAP50_95=float(m.box.map),
            P=float(m.box.mp), R=float(m.box.mr),
            per_class={
                NAMES[i]: dict(
                    AP50=float(ap50[i]) if i < len(ap50) else None,
                    n_gt=int((m.box.ap_class_index == i).sum()),
                )
                for i in range(nc) if i in list(m.box.ap_class_index)
            },
        )
        cm, fp, fn, vb, okc, n_img = run_split(model, split)
        rows = {}
        for c in range(nc):
            tp = int(cm[c, c])
            p = tp / max(1, tp + int(fp[c]))
            rr = tp / max(1, tp + int(fn[c]))
            rows[NAMES[c]] = dict(gt=int(cm[c].sum()), tp=tp, fn=int(fn[c]), fp=int(fp[c]),
                                  P=round(p, 4), R=round(rr, 4),
                                  ok_conf_mean=round(conf_summary(okc[c]), 4))
        van_gt = int(cm[VAN_ID].sum())
        report[split]["n_img"] = n_img
        report[split]["confusion"] = dict(
            per_class=rows,
            Van_to_Car_pct=round(100 * cm[VAN_ID, CAR_ID] / max(1, van_gt), 2),
            Van_to_Car=(int(cm[VAN_ID, CAR_ID]), van_gt),
            background_to_Car=int(fp[CAR_ID]),
            van_buckets=[dict(bucket=BUCKET_NAMES[i], **{k: int(vv) for k, vv in s.items() if k != "car_conf"},
                              to_car_pct=round(100 * s["to_car"] / max(1, s["n"]), 1),
                              wrongCarConf_mean=round(float(np.mean(s["car_conf"])), 4) if s["car_conf"] else None)
                         for i, s in enumerate(vb)],
        )
        print(f"\n===== {tag} on {split}  [{n_img} imgs, conf>={CONF_EVAL}, IoU 0.5] =====")
        print(f"mAP50={report[split]['mAP50']:.4f}  mAP50-95={report[split]['mAP50_95']:.4f}")
        print(f"Van->Car: {cm[VAN_ID,CAR_ID]}/{van_gt} = {report[split]['confusion']['Van_to_Car_pct']}%   background->Car: {fp[CAR_ID]}")
        print(f"{'class':16s} {'gt':>5s} {'P':>6s} {'R':>6s} {'AP50':>6s} {'conf(ok)':>8s}")
        for i, n in enumerate(NAMES):
            r_ = rows[n]
            ap = report[split]["per_class"].get(n, {}).get("AP50")
            ap = f"{ap:.3f}" if ap is not None else "-"
            print(f"{n:16s} {r_['gt']:5d} {r_['P']:6.3f} {r_['R']:6.3f} {ap:>6s} {r_['ok_conf_mean']:8.3f}")
        print("Van size buckets:")
        for b in report[split]["confusion"]["van_buckets"]:
            print(f"  {b['bucket']:>10s}: n={b['n']:4d} ok={b['ok']:4d} Van->Car={b['to_car']:3d} "
                  f"({b['to_car_pct']:5.1f}%) ->other={b['to_other']:3d} missed={b['miss']:3d} "
                  f"wrongCarConf={b['wrongCarConf_mean']}")
    return report


def main():
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True, help="one or more best.pt paths (tag = run dir name)")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--device", default="2")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="/tmp/opencode/eval_confusion.json")
    args = ap.parse_args()
    global DEVICE, IMGSZ, DATA, NAMES, nc, VAN_ID, CAR_ID
    DEVICE = args.device
    IMGSZ = args.imgsz
    DATA = args.data
    _ym = yaml.safe_load(open(DATA))
    NAMES = _ym["names"]
    if isinstance(NAMES, dict):
        NAMES = [NAMES[k] for k in sorted(NAMES)]
    nc = len(NAMES)
    VAN_ID = NAMES.index("Van")
    CAR_ID = NAMES.index("Private Car")

    reports = {}
    for w in args.weights:
        tag = os.path.basename(os.path.dirname(os.path.dirname(w)))
        reports[f"{tag}@{IMGSZ}"] = evaluate(tag, w)

    keys = ["mAP50", "mAP50_95"]
    print(f"\n===== side-by-side (valid) =====")
    tags = list(reports)
    print(f"{'metric':28s} " + " ".join(f"{t[:18]:>18s}" for t in tags))
    vals = [reports[t]["valid"] for t in tags]
    confs = [v["confusion"] for v in vals]

    def ap50(v, c):
        x = v["per_class"].get(c, {}).get("AP50")
        return f"{x:.4f}" if x is not None else "-"

    rows = [
        ("mAP50", [f"{v['mAP50']:.4f}" for v in vals]),
        ("mAP50-95", [f"{v['mAP50_95']:.4f}" for v in vals]),
        ("Van AP50", [ap50(v, "Van") for v in vals]),
        ("Van->Car %", [f"{c['Van_to_Car_pct']:.1f}" for c in confs]),
        ("<0.01 Van->Car %", [f"{c['van_buckets'][0]['to_car_pct']:.1f}" for c in confs]),
        ("background->Car", [str(c["background_to_Car"]) for c in confs]),
        ("Car AP50", [ap50(v, "Private Car") for v in vals]),
        ("PLB GMB AP50", [ap50(v, "PLB GMB") for v in vals]),
        ("Motorcycle AP50", [ap50(v, "Motorcycle") for v in vals]),
        ("Van P", [f"{c['per_class']['Van']['P']:.3f}" for c in confs]),
        ("Van R", [f"{c['per_class']['Van']['R']:.3f}" for c in confs]),
        ("Van conf(ok)", [f"{c['per_class']['Van']['ok_conf_mean']:.3f}" for c in confs]),
    ]
    for name, cells in rows:
        print(f"{name:28s} " + " ".join(f"{x:>18s}" for x in cells))

    with open(args.out, "w") as f:
        json.dump(reports, f, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
