import csv
import glob
import os
import shutil

import cv2
import numpy as np

V2 = r"dataset\dataset_merged_v2"
STAGE = r"dataset\_v3_staging"
V3 = r"dataset\dataset_merged_v3"

HAMMING = 3


def ahash(path):
    im = cv2.imread(path)
    if im is None:
        return None
    g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (8, 8))
    return (g > g.mean()).flatten()


def pack(bits):
    return np.packbits(bits.astype(bool))


def hamming(a, b):
    return int(np.count_nonzero(np.unpackbits(a ^ b)))


if os.path.exists(V3):
    shutil.rmtree(V3)
shutil.copytree(V2, V3)

moto_imgs = sorted(glob.glob(os.path.join(STAGE, "moto", "images", "*.jpg")))
v2_train_imgs = glob.glob(os.path.join(V3, "train", "images", "*"))

ref = []
for p in v2_train_imgs:
    h = ahash(p)
    if h is not None:
        ref.append((os.path.basename(p), pack(h)))

kept, dropped = [], 0
seen = []
for p in moto_imgs:
    h = ahash(p)
    if h is None:
        dropped += 1
        continue
    ph = pack(h)
    dup = False
    for _, q in ref:
        if hamming(ph, q) <= HAMMING:
            dup = True
            break
    if not dup:
        for _, q in seen:
            if hamming(ph, q) <= HAMMING:
                dup = True
                break
    if dup:
        dropped += 1
        continue
    seen.append((os.path.basename(p), ph))
    ref.append((os.path.basename(p), ph))
    kept.append(p)

print(f"moto frames: {len(moto_imgs)} | kept: {len(kept)} | dropped(dup/corrupt): {dropped}")

for p in kept:
    stem = os.path.splitext(os.path.basename(p))[0]
    shutil.copy(p, os.path.join(V3, "train", "images", f"moto_{os.path.basename(p)}"))
    shutil.copy(os.path.join(STAGE, "moto", "labels", stem + ".txt"), os.path.join(V3, "train", "labels", f"moto_{stem}.txt"))

n_paste = 0
for src in glob.glob(os.path.join(STAGE, "pasted", "images", "*.jpg")):
    stem = os.path.splitext(os.path.basename(src))[0]
    shutil.copy(src, os.path.join(V3, "train", "images", src.split(os.sep)[-1].replace("paste_", "paste_")))
    shutil.copy(os.path.join(STAGE, "pasted", "labels", stem + ".txt"), os.path.join(V3, "train", "labels", stem + ".txt"))
    n_paste += 1

with open(os.path.join(V3, "data.yaml"), "w", encoding="utf-8") as f:
    f.write("train: train/images\nval: valid/images\ntest: test/images\n\nnc: 12\n")
    f.write("names: ['Motorcycle', 'Private Car', 'Taxi', 'Light Van', 'LGV', 'MGV', 'HGV', 'Container', 'Public Light Bus/GMB', 'Light Bus', 'Franchised Bus', 'Coach']\n")

rows = []
for split in ["train", "valid", "test"]:
    for lf in glob.glob(os.path.join(V3, split, "labels", "*.txt")):
        stem = os.path.splitext(os.path.basename(lf))[0]
        if stem.startswith("moto_"):
            src, seq = "motorcycle_vietnam_v3", "moto_pseudo"
        elif stem.startswith("paste_"):
            src, seq = "paste_aug_v3", "paste"
        elif stem.startswith("img_") or stem.startswith("frame_") or stem.startswith("0000") or True:
            src, seq = "merged_v2", split
        with open(lf, "r", encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
        rows.append({"split": split, "file": stem, "source": src, "sequence": seq, "boxes": n})

with open(os.path.join(V3, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["split", "file", "source", "sequence", "boxes"])
    w.writeheader()
    w.writerows(rows)

stats = {}
for split in ["train", "valid", "test"]:
    cls_imgs = {}
    cls_boxes = {}
    imgs = 0
    for lf in glob.glob(os.path.join(V3, split, "labels", "*.txt")):
        imgs += 1
        present = set()
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    c = int(float(parts[0]))
                    cls_boxes[c] = cls_boxes.get(c, 0) + 1
                    present.add(c)
        for c in present:
            cls_imgs[c] = cls_imgs.get(c, 0) + 1
    stats[split] = (imgs, cls_imgs, cls_boxes)

NAMES = ['Motorcycle', 'Private Car', 'Taxi', 'Light Van', 'LGV', 'MGV', 'HGV', 'Container', 'Public Light Bus/GMB', 'Light Bus', 'Franchised Bus', 'Coach']
with open(os.path.join(V2, "report.md"), encoding="utf-8") as f:
    v2_train = {}
    for line in f:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 6 and parts[1].isdigit():
            v2_train[int(parts[1])] = int(parts[3])

lines = ["# Merged Dataset v3 Report", "", "base: dataset_merged_v2 (train augmented; valid/test untouched)", ""]
lines += ["## Additions", "", "| source | images | note |", "|---|---|---|",
          f"| motorcycle_vietnam (pseudo-labeled) | {len(kept)} | GT moto + dual-model pseudo (best.pt conf>=0.5 & YOLO-World agree, or conf>=0.65); moto add-back conf>=0.7; audit bucket dropped |",
          f"| paste_aug (copy-paste) | {n_paste} | b12_van/b12_moto self-crops + HK-rescued crops, feathered, collision-checked, class-median scale |",
          "", "## Rejected sources", "",
          "- VAN (Uganda): auto-filter kept only ~2 complete crops of 2642 - dropped",
          "- vehicles.v2 (Indonesia): deferred (foreign domain, taxi-look livery risk), candidate for v4",
          "- HK web set suspects: toys/posters/trams/product shots dropped after multimodal audit; only verified rescues kept", ""]
lines += ["## Class distribution (boxes)", "", "| id | class | v2 train | v3 train | v3 valid | v3 test |", "|---|---|---|---|---|---|"]
ti, timg, tbox = stats["train"]
for c in range(12):
    lines.append(f"| {c} | {NAMES[c]} | {v2_train.get(c, '-')} | {tbox.get(c, 0)} | {stats['valid'][2].get(c, 0)} | {stats['test'][2].get(c, 0)} |")
lines += ["| | **all** | " + " | ".join([str(sum(v2_train.values())), str(sum(tbox.values())), str(sum(stats['valid'][2].values())), str(sum(stats['test'][2].values()))]) + " |", ""]
lines += [f"train images: {ti} (v2: 6583) | valid: {stats['valid'][0]} | test: {stats['test'][0]}",
          f"imbalance ratio train (max/min): {max(tbox.values()) / min(tbox[c] for c in range(12) if tbox.get(c, 0) > 0):.1f}x", ""]
with open(os.path.join(V3, "report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("v3 train images:", ti)
for c in range(12):
    v2v = v2_train.get(c, 0)
    print(f"  {NAMES[c]:24s} {v2v:6d} -> {tbox.get(c, 0):6d}  ({(tbox.get(c, 0) / max(1, v2v) - 1) * 100:+.0f}%)")
