import os
import shutil
from collections import Counter

SRC = os.path.join("dataset", "batch_12.v9i.yolov11")
DST = os.path.join("dataset", "batch_12.v10i.merge8.yolov11")
SPLITS = ["train", "valid", "test"]

OLD_NAMES = [
    "Coach", "Franchised Bus", "HGV", "LGV", "Light Bus", "MGV",
    "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van",
]
NEW_NAMES = [
    "Bus", "Franchised Bus", "Truck", "Motorcycle", "PLB GMB",
    "Private Car", "Taxi", "Van",
]

MERGE = {
    "Bus": ["Coach", "Light Bus"],
    "Franchised Bus": ["Franchised Bus"],
    "Truck": ["HGV", "MGV", "LGV"],
    "Motorcycle": ["Motorcycle"],
    "PLB GMB": ["PLB GMB"],
    "Private Car": ["Private Car"],
    "Taxi": ["Taxi"],
    "Van": ["Van"],
}

OLD_TO_NEW = {
    OLD_NAMES.index(old): NEW_NAMES.index(new)
    for new, olds in MERGE.items()
    for old in olds
}


def remap_label(src_path, dst_path, old_counter):
    out_lines = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            cid = int(float(parts[0]))
            if cid not in OLD_TO_NEW:
                raise ValueError(f"{src_path}: unknown class id {cid}")
            old_counter[cid] += 1
            parts[0] = str(OLD_TO_NEW[cid])
            out_lines.append(" ".join(parts))
    with open(dst_path, "w", encoding="utf-8") as f:
        if out_lines:
            f.write("\n".join(out_lines) + "\n")
    return len(out_lines)


def write_data_yaml():
    names = "[" + ", ".join(f"'{n}'" for n in NEW_NAMES) + "]"
    content = (
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "\n"
        f"nc: {len(NEW_NAMES)}\n"
        f"names: {names}\n"
    )
    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(content)


def main():
    if not os.path.isdir(SRC):
        raise SystemExit(f"source not found: {SRC}")
    if os.path.exists(DST):
        raise SystemExit(f"target already exists: {DST}")

    grand_old = Counter()
    per_split_new = {split: Counter() for split in SPLITS}

    for split in SPLITS:
        src_img = os.path.join(SRC, split, "images")
        src_lbl = os.path.join(SRC, split, "labels")
        dst_img = os.path.join(DST, split, "images")
        dst_lbl = os.path.join(DST, split, "labels")
        os.makedirs(dst_img)
        os.makedirs(dst_lbl)

        split_old = Counter()
        n_missing = 0
        n_boxes = 0
        for name in sorted(os.listdir(src_img)):
            stem, _ = os.path.splitext(name)
            shutil.copy2(os.path.join(src_img, name), os.path.join(dst_img, name))
            lbl_src = os.path.join(src_lbl, stem + ".txt")
            lbl_dst = os.path.join(dst_lbl, stem + ".txt")
            if os.path.isfile(lbl_src):
                n_boxes += remap_label(lbl_src, lbl_dst, split_old)
            else:
                n_missing += 1
                open(lbl_dst, "w").close()
        grand_old.update(split_old)
        for old_cid, cnt in split_old.items():
            per_split_new[split][OLD_TO_NEW[old_cid]] += cnt
        print(f"[{split}] images={len(os.listdir(dst_img))} boxes={n_boxes} missing_labels={n_missing}")

    print("\nold -> new mapping (all splits):")
    print(f"{'old':>3} {'old name':<16} {'->':>2} {'new':>3} {'new name':<16} {'count':>8}")
    for old_cid in range(len(OLD_NAMES)):
        new_cid = OLD_TO_NEW[old_cid]
        print(f"{old_cid:>3} {OLD_NAMES[old_cid]:<16} -> {new_cid:>3} {NEW_NAMES[new_cid]:<16} {grand_old[old_cid]:>8}")

    print("\nmerged dataset distribution:")
    print(f"{'id':>3} {'name':<16} {'train':>8} {'valid':>8} {'test':>8}")
    for new_cid, name in enumerate(NEW_NAMES):
        print(f"{new_cid:>3} {name:<16}"
              f" {per_split_new['train'][new_cid]:>8}"
              f" {per_split_new['valid'][new_cid]:>8}"
              f" {per_split_new['test'][new_cid]:>8}")

    write_data_yaml()
    print(f"\ndata.yaml written, nc={len(NEW_NAMES)}")
    print(f"done: {DST}")


if __name__ == "__main__":
    main()
