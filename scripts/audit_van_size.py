import collections
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKETS = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 2.0)]
LBL = (0, "<0.01"), (1, "0.01-0.02"), (2, "0.02-0.05"), (3, ">0.05")


def area_dist(dataset, cls_id, cls_name):
    print(f"\n== {dataset} : {cls_name} bbox area buckets (normalized w*h) ==")
    for split in ["train", "valid", "test"]:
        c = collections.Counter()
        heights = collections.defaultdict(list)
        for lf in glob.glob(os.path.join(ROOT, "dataset", dataset, split, "labels", "*.txt")):
            with open(lf, encoding="utf-8") as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 5 and int(float(p[0])) == cls_id:
                        a = float(p[3]) * float(p[4])
                        for i, (lo, hi) in enumerate(BUCKETS):
                            if lo <= a < hi:
                                c[i] += 1
                                heights[i].append(float(p[4]))
                                break
        tot = sum(c.values())
        cells = "  ".join(f"{LBL[i][1]}:{c[i]:4d}({100*c[i]/max(1,tot):4.1f}%)" for i in range(4))
        print(f"  {split:6s} n={tot:5d}  {cells}")


area_dist("batch_12.v10i.merge8.yolov11", 7, "Van (merge8)")
area_dist("batch_12.v9i.yolov11", 10, "Van (v9 11cls)")
area_dist("dataset_merged_v2", 3, "Light Van (v2 benchmark)")

# distribution of Private Car for contrast (v10)
area_dist("batch_12.v10i.merge8.yolov11", 5, "Private Car (merge8, contrast)")
