import glob
import os
import shutil
from collections import Counter

SRC = os.path.join("dataset", "batch_12.v10i.merge8.yolov11")
DST = os.path.join("dataset", "batch_12.v11i.merge8.van2x.yolov11")
SPLITS = ["train", "valid", "test"]
VAN_ID = 7

NAMES = ["Bus", "Franchised Bus", "Truck", "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van"]


def link(src, dst):
    if os.path.exists(dst):
        return
    os.link(src, dst)


def label_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                ids.append(int(float(p[0])))
    return ids


def write_data_yaml():
    names = "[" + ", ".join(f"'{n}'" for n in NAMES) + "]"
    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(
            "train: ../train/images\n"
            "val: ../valid/images\n"
            "test: ../test/images\n"
            "\n"
            f"nc: {len(NAMES)}\n"
            f"names: {names}\n"
        )


def main():
    if os.path.exists(DST):
        raise SystemExit(f"target already exists: {DST}")

    stats = {}
    for split in SPLITS:
        src_img = os.path.join(SRC, split, "images")
        src_lbl = os.path.join(SRC, split, "labels")
        dst_img = os.path.join(DST, split, "images")
        dst_lbl = os.path.join(DST, split, "labels")
        os.makedirs(dst_img)
        os.makedirs(dst_lbl)

        before = Counter()
        after = Counter()
        n_img = 0
        n_dup = 0
        for name in sorted(os.listdir(src_img)):
            stem, ext = os.path.splitext(name)
            lbl_src = os.path.join(src_lbl, stem + ".txt")
            ids = label_ids(lbl_src) if os.path.isfile(lbl_src) else []
            for c in ids:
                after[c] += 1
                if split == "train":
                    before[c] += 1
            n_img += 1
            link(os.path.join(src_img, name), os.path.join(dst_img, name))
            if os.path.isfile(lbl_src):
                shutil.copy2(lbl_src, os.path.join(dst_lbl, stem + ".txt"))
            else:
                open(os.path.join(dst_lbl, stem + ".txt"), "w").close()
            if split == "train" and VAN_ID in ids:
                dup = stem + "_van2x" + ext
                link(os.path.join(src_img, name), os.path.join(dst_img, dup))
                shutil.copy2(lbl_src, os.path.join(dst_lbl, stem + "_van2x.txt"))
                for c in ids:
                    after[c] += 1
                n_dup += 1
        stats[split] = dict(before=before, after=after, n_img=n_img, n_dup=n_dup)
        print(f"[{split}] images={n_img} duplicated={n_dup}")

    print(f"\ntrain per-class instances (before -> after, dup {stats['train']['n_dup']} van-images):")
    print(f"{'class':16s} {'before':>8s} {'after':>8s} {'delta':>7s} {'+%':>6s}")
    for i, n in enumerate(NAMES):
        b = stats["train"]["before"][i]
        a = stats["train"]["after"][i]
        print(f"{n:16s} {b:8d} {a:8d} {a - b:7d} {100 * (a - b) / max(1, b):6.1f}")
    b_ratio = stats["train"]["before"][VAN_ID] / max(1, stats["train"]["before"][5])
    a_ratio = stats["train"]["after"][VAN_ID] / max(1, stats["train"]["after"][5])
    print(f"\nVan/Car instance ratio: {b_ratio:.4f} -> {a_ratio:.4f}  (x{a_ratio / b_ratio:.2f})")
    print(f"Van absolute: x{stats['train']['after'][VAN_ID] / max(1, stats['train']['before'][VAN_ID]):.2f}")
    print(f"train images: {stats['train']['n_img']} -> {stats['train']['n_img'] + stats['train']['n_dup']}")
    print(f"valid/test: unchanged ({stats['valid']['n_img']}/{stats['test']['n_img']} images, hardlinks)")

    write_data_yaml()
    print(f"\ndata.yaml written (nc={len(NAMES)}), done: {DST}")


if __name__ == "__main__":
    main()
