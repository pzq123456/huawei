import csv
import glob
import os
import shutil

B12 = r"dataset\batch_12.v9i.yolov11"
VN = r"dataset\Vietnam Container truck.v1i.yolov11"
STAGE = r"dataset\_v3_staging"
V3 = r"dataset\dataset_merged_v3"
LITE = r"dataset\dataset_merged_lite"

B12_MAP = {0: 11, 1: 10, 2: 6, 3: 4, 4: 9, 5: 5, 6: 0, 7: 8, 8: 1, 9: 2, 10: 3}
VN_MAP = {0: 7}
NAMES = ['Motorcycle', 'Private Car', 'Taxi', 'Light Van', 'LGV', 'MGV', 'HGV', 'Container', 'Public Light Bus/GMB', 'Light Bus', 'Franchised Bus', 'Coach']

if os.path.exists(LITE):
    shutil.rmtree(LITE)
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(LITE, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(LITE, split, "labels"), exist_ok=True)

manifest = []


def copy_pair(img_src, lbl_src, out_split, out_stem, source):
    shutil.copy(img_src, os.path.join(LITE, out_split, "images", out_stem + os.path.splitext(img_src)[1]))
    shutil.copy(lbl_src, os.path.join(LITE, out_split, "labels", out_stem + ".txt"))
    n = sum(1 for line in open(lbl_src, encoding="utf-8") if line.strip())
    manifest.append({"split": out_split, "file": out_stem, "source": source, "boxes": n})


def find_img(root, split, stem):
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(root, split, "images", stem + ext)
        if os.path.exists(p):
            return p
    return None


for split in ["train"]:
    for lf in glob.glob(os.path.join(B12, split, "labels", "*.txt")):
        stem = os.path.splitext(os.path.basename(lf))[0]
        p = find_img(B12, split, stem)
        if p is None:
            continue
        tmp = stem + "_remap.txt"
        with open(lf, "r", encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as g:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    g.write(f"{B12_MAP[int(float(parts[0]))]} " + " ".join(parts[1:5]) + "\n")
        copy_pair(p, tmp, split, f"b12_{stem}", f"batch12_{split}")
        os.remove(tmp)

for split in ["train"]:
    for lf in glob.glob(os.path.join(VN, split, "labels", "*.txt")):
        stem = os.path.splitext(os.path.basename(lf))[0]
        p = find_img(VN, split, stem)
        if p is None:
            continue
        tmp = stem + "_remap.txt"
        with open(lf, "r", encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as g:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    g.write(f"{VN_MAP[int(float(parts[0]))]} " + " ".join(parts[1:5]) + "\n")
        copy_pair(p, tmp, split, f"vn_{stem}", f"vietnam_{split}")
        os.remove(tmp)

for p in glob.glob(os.path.join(V3, "train", "images", "moto_*.jpg")):
    stem = os.path.splitext(os.path.basename(p))[0]
    copy_pair(p, os.path.join(V3, "train", "labels", stem + ".txt"), "train", stem, "motorcycle_pseudo")

pm_rows = list(csv.DictReader(open(os.path.join(STAGE, "paste_manifest.csv"), encoding="utf-8")))
moto_paste_outs = {r["out"] for r in pm_rows if r["cls"] == "Motorcycle"}
for o in sorted(moto_paste_outs):
    stem = os.path.splitext(o)[0]
    copy_pair(os.path.join(STAGE, "pasted", "images", o), os.path.join(STAGE, "pasted", "labels", stem + ".txt"), "train", stem, "paste_moto")

for split in ["valid", "test"]:
    for p in glob.glob(os.path.join(V3, split, "images", "*")):
        stem = os.path.splitext(os.path.basename(p))[0]
        copy_pair(p, os.path.join(V3, split, "labels", stem + ".txt"), split, stem, f"v2benchmark_{split}")

with open(os.path.join(LITE, "data.yaml"), "w", encoding="utf-8") as f:
    f.write("train: train/images\nval: valid/images\ntest: test/images\n\nnc: 12\n")
    f.write("names: ['Motorcycle', 'Private Car', 'Taxi', 'Light Van', 'LGV', 'MGV', 'HGV', 'Container', 'Public Light Bus/GMB', 'Light Bus', 'Franchised Bus', 'Coach']\n")

with open(os.path.join(LITE, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["split", "file", "source", "boxes"])
    w.writeheader()
    w.writerows(manifest)

stats = {}
for split in ["train", "valid", "test"]:
    imgs = 0
    cls_boxes = {}
    for lf in glob.glob(os.path.join(LITE, split, "labels", "*.txt")):
        imgs += 1
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    c = int(float(parts[0]))
                    cls_boxes[c] = cls_boxes.get(c, 0) + 1
    stats[split] = (imgs, cls_boxes)

src_count = {}
for r in manifest:
    key = (r["split"], r["source"])
    src_count[key] = src_count.get(key, 0) + 1

lines = ["# Merged Dataset lite Report", "", "recipe: batch_12 (HK local, 11cls->12cls remap) + Vietnam Container + motorcycle augmentation (pseudo frames + moto pastes)", "benchmark: valid/test shared with v2/v3 for direct comparison", ""]
lines += ["## Composition", "", "| split | source | images |", "|---|---|---|"]
for (split, source), n in sorted(src_count.items()):
    lines.append(f"| {split} | {source} | {n} |")
lines += ["", "## Class distribution (boxes)", "", "| id | class | train | valid | test |", "|---|---|---|---|---|"]
for c in range(12):
    lines.append(f"| {c} | {NAMES[c]} | {stats['train'][1].get(c, 0)} | {stats['valid'][1].get(c, 0)} | {stats['test'][1].get(c, 0)} |")
lines += ["", f"train images: {stats['train'][0]} | valid: {stats['valid'][0]} | test: {stats['test'][0]}"]
with open(os.path.join(LITE, "report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("lite composition:")
for (split, source), n in sorted(src_count.items()):
    print(f"  {split:6s} {source:22s} {n}")
print("train boxes per class:")
for c in range(12):
    print(f"  {NAMES[c]:24s} {stats['train'][1].get(c, 0)}")

for split in ["train", "valid", "test"]:
    imgs = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(LITE, split, "images", "*"))}
    lbls = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(LITE, split, "labels", "*.txt"))}
    assert not (imgs - lbls) and not (lbls - imgs), f"{split} mismatch"
print("structure OK")
