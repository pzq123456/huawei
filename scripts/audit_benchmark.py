import collections
import glob
import hashlib
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "dataset")


def md5set(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*", "images", "*")):
        split = p.replace("\\", "/").split("/")[-3]
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        out.setdefault(split, {})[h] = os.path.basename(p)
    return out


def boxes_per_split(d):
    per = {}
    for lf in glob.glob(os.path.join(d, "*", "labels", "*.txt")):
        split = lf.replace("\\", "/").split("/")[-3]
        with open(lf, encoding="utf-8") as f:
            n = sum(1 for line in f if len(line.split()) >= 5)
        per.setdefault(split, [0, 0])
        per[split][0] += 1
        per[split][1] += n
    return per


def seqkey(stem):
    s = re.sub(r"(_jpg)?\.rf\..*$", "", stem)
    s = re.sub(r"\.(jpg|jpeg|png)$", "", s)
    m = re.search(r"(.*?[_-])(?:frame[_-])?f?(\d+)$", s)
    if m:
        return m.group(1), int(m.group(2))
    return s, -1


def seq_audit(d, name):
    idx = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in glob.glob(os.path.join(d, "*", "images", "*")):
        split = p.replace("\\", "/").split("/")[-3]
        pre, fi = seqkey(os.path.splitext(os.path.basename(p))[0])
        idx[split][pre].append(fi)
    print(f"  sequence prefix groups: " + ", ".join(f"{s}={len(idx[s])}" for s in ["train", "valid", "test"]))
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        shared = set(idx[a]) & set(idx[b])
        n_img = sum(len(idx[b][k]) for k in shared)
        if shared:
            # temporal adjacency: min |frame idx| distance between val img and nearest train img in same prefix
            close = 0
            for k in shared:
                for fi in idx[b][k]:
                    if fi >= 0 and min(abs(fi - t) for t in idx[a][k] if t >= 0) <= 20:
                        close += 1
            print(f"  {a}∩{b}: {len(shared)} seq-prefixes shared, {n_img} {b}-images affected, {close} within ±20 frames")


def class_counts(d, names):
    per = collections.defaultdict(collections.Counter)
    for lf in glob.glob(os.path.join(d, "*", "labels", "*.txt")):
        split = lf.replace("\\", "/").split("/")[-3]
        with open(lf, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    per[split][int(float(parts[0]))] += 1
    for s in ["train", "valid", "test"]:
        print("   ", s, {names[c]: per[s][c] for c in sorted(per[s])} if names else dict(sorted(per[s].items())))


names12 = ["Motorcycle", "Private Car", "Taxi", "Light Van", "LGV", "MGV", "HGV", "Container",
           "Public Light Bus/GMB", "Light Bus", "Franchised Bus", "Coach"]
names11 = ["Coach", "Franchised Bus", "HGV", "LGV", "Light Bus", "MGV", "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van"]
names8 = ["Bus", "Franchised Bus", "Truck", "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van"]

for d, names in [
    ("batch_12.v5i.yolov11", names11),
    ("batch_12.v9i.yolov11", names11),
    ("batch_12.v10i.merge8.yolov11", names8),
    ("dataset_merged_v2", names12),
    ("dataset_merged_v3", names12),
    ("dataset_merged_lite", names12),
]:
    path = os.path.join(DS, d)
    print(f"\n================ {d} ================")
    per = boxes_per_split(path)
    for s in ["train", "valid", "test"]:
        print(f"  {s:6s} images={per.get(s, [0, 0])[0]:5d}  boxes={per.get(s, [0, 0])[1]:6d}")
    print("  per-class boxes:")
    class_counts(path, names)
    m = md5set(path)
    print("  byte-identical duplicates (md5):")
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        ov = set(m.get(a, {})) & set(m.get(b, {}))
        dup_in = 0
        for s in ["train", "valid", "test"]:
            hs = collections.Counter(m.get(s, {}).keys())
            dup_in += sum(v - 1 for v in hs.values())
        print(f"    {a}∩{b}: {len(ov)}")
    if d == "batch_12.v10i.merge8.yolov11":
        seq_audit(path, d)

# cross-dataset contamination vs v2 benchmark (valid+test)
print("\n================ cross-dataset byte-identical overlap vs v2 benchmark valid ================")
bench = {}
for s in ["valid", "test"]:
    for p in glob.glob(os.path.join(DS, "dataset_merged_v2", s, "images", "*")):
        bench[hashlib.md5(open(p, "rb").read()).hexdigest()] = (s, os.path.basename(p))
for d in ["batch_12.v5i.yolov11", "batch_12.v9i.yolov11", "batch_12.v10i.merge8.yolov11", "dataset_merged_v3", "dataset_merged_lite"]:
    for s in ["train", "valid", "test"]:
        ov = collections.Counter()
        for p in glob.glob(os.path.join(DS, d, s, "images", "*")):
            h = hashlib.md5(open(p, "rb").read()).hexdigest()
            if h in bench:
                ov[bench[h][0]] += 1
        if ov:
            print(f"  {d}/{s}  ∩ v2-benchmark: {dict(ov)}")
