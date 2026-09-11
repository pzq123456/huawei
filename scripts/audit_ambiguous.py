import collections
import csv
import glob
import os

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "dataset", "dataset_merged_v2")
NAMES = ["Motorcycle", "Private Car", "Taxi", "Light Van", "LGV", "MGV", "HGV", "Container",
         "Public Light Bus/GMB", "Light Bus", "Franchised Bus", "Coach"]
# ambiguous source classes (dropped as undecipherable)
AMBIG = {"Bus": None, "Minibus": None, "Truck": None, "van": None}

rows = list(csv.DictReader(open(os.path.join(V2, "manifest.csv"), encoding="utf-8-sig")))
name2src = {r["final_name"]: r["source"] for r in rows}

# 1) kept boxes per class per source (from final label files)
kept = collections.defaultdict(collections.Counter)   # source -> class -> n
for lf in glob.glob(os.path.join(V2, "*", "labels", "*.txt")):
    stem = os.path.splitext(os.path.basename(lf))[0]
    src = name2src.get(stem)
    with open(lf, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5 and src:
                kept[src][NAMES[int(float(p[0]))]] += 1

# 2) original boxes and dropped per source
orig = collections.Counter()
dropped = collections.Counter()
for r in rows:
    orig[r["source"]] += int(r["n_boxes"])
    dropped[r["source"]] += int(r["n_dropped_ambiguous"])

# 3) dropped per class per source (from report.md)
dropped_cls = {
    "vehicledetect": {"Bus": 389, "Minibus": 33, "Truck": 387, "van": 342},
    "yolov8": {"Bus": 1208, "Minibus": 1404, "Truck": 145, "van": 222},
}

print("== per-source box accounting ==")
print(f"{'source':14s} {'orig':>7s} {'dropped':>8s} {'kept':>7s} {'drop%':>7s}")
for src in sorted(orig):
    k = orig[src] - dropped[src]
    print(f"{src:14s} {orig[src]:7d} {dropped[src]:8d} {k:7d} {100*dropped[src]/max(1,orig[src]):6.1f}%")

print("\n== kept boxes per class per source (final labels) ==")
srcs = sorted(kept)
hdr = "class".ljust(20) + "".join(s.rjust(14) for s in srcs)
print(hdr)
for c in NAMES:
    print(c.ljust(20) + "".join(str(kept[s][c]).rjust(14) for s in srcs))

print("\n== per-class retention for ambiguous-prone external sources ==")
# original per class = kept + dropped(class); only vehicledetect & yolov8 had drops
for src in ["vehicledetect", "yolov8"]:
    print(f"-- {src} --")
    print(f"{'class':12s} {'kept':>6s} {'dropped':>8s} {'orig':>7s} {'retained%':>10s}")
    for cls in sorted(set(kept[src]) | set(dropped_cls.get(src, {}))):
        k = kept[src][cls]
        d = dropped_cls.get(src, {}).get(cls, 0)
        print(f"{cls:12s} {k:6d} {d:8d} {k+d:7d} {100*k/max(1, k+d):9.1f}%")

# 4) size proxy of dropped boxes: can't recover directly; but compare kept-box size
#    distribution of Van-family classes between batch12 (never dropped) vs external sources
def sizes_for(src, cls_ids):
    out = []
    for lf in glob.glob(os.path.join(V2, "*", "labels", "*.txt")):
        stem = os.path.splitext(os.path.basename(lf))[0]
        if name2src.get(stem) != src:
            continue
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            q = lf.replace("/labels/", "/images/").replace(".txt", ext)
            if os.path.exists(q):
                img = cv2.imread(q)
                break
        if img is None:
            continue
        H, W = img.shape[:2]
        with open(lf, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) >= 5 and int(float(p[0])) in cls_ids:
                    out.append(float(p[3]) * float(p[4]))
    return out

print("\n== bbox area (normalized w*h) of kept boxes, Van-family, batch12 vs external sources ==")
for src, cls in [("batch12", [3]), ("vehicledetect", [3, 4, 5, 6]), ("yolov8", [3, 4, 5, 6])]:
    import numpy as np
    a = sizes_for(src, cls)
    if a:
        a = np.array(a)
        print(f"{src:14s} n={len(a):5d} median_area={np.median(a):.5f}  p25={np.percentile(a,25):.5f} p75={np.percentile(a,75):.5f}  frac<0.01={np.mean(a<0.01):.2f}")
    else:
        print(f"{src:14s} n=0")
