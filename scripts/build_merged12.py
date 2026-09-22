"""Build 12-class merged dataset: batch_12.v9i (11 cls) + tmp/9.22 (X-AnyLabeling, +Container).

Decisions (user-confirmed):
- taxonomy: v9 class order unchanged (0-10) + Container appended as id 11
- 9.22 images recovered from rtsp_mine_1200_merged where the .json has no sibling .jpg
- 9.22 split by camera + chronological contiguous 80/10/10 (sequence-aware, no shuffle)
- v9 train/valid/test preserved verbatim (labels unchanged, ids already 0-10)
- empty 9.22 annotations kept as background (empty label file)
"""
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_V9 = ROOT / "dataset" / "batch_12.v9i.yolov11"
SRC_922 = ROOT / "tmp" / "9.22" / "images"
RECOVER = ROOT / "dataset" / "rtsp_mine_1200_merged" / "images"
OUT = ROOT / "dataset" / "batch_12.v14i.merged12.yolov11"

NAMES = ["Coach", "Franchised Bus", "HGV", "LGV", "Light Bus", "MGV",
         "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van", "Container"]
NAME2ID = {n: i for i, n in enumerate(NAMES)}
SPLITS = ["train", "valid", "test"]


def parse_xany(jp: Path):
    d = json.loads(jp.read_text(encoding="utf-8"))
    w, h = int(d["imageWidth"]), int(d["imageHeight"])
    boxes = []
    for s in d.get("shapes", []):
        st = s.get("shape_type", "rectangle")
        if st != "rectangle":
            raise ValueError(f"{jp.name}: unsupported shape_type={st}")
        label = s["label"].strip()
        if label not in NAME2ID:
            raise ValueError(f"{jp.name}: unknown label={label!r}")
        pts = s["points"]
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x1, x2 = sorted((min(xs), max(xs)))
        y1, y2 = sorted((min(ys), max(ys)))
        x1 = max(0.0, min(float(w), x1)); x2 = max(0.0, min(float(w), x2))
        y1 = max(0.0, min(float(h), y1)); y2 = max(0.0, min(float(h), y2))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1e-6 or bh <= 1e-6:
            continue
        boxes.append((NAME2ID[label], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw / w, bh / h))
    return boxes


def split_922():
    stems = sorted(p.stem for p in SRC_922.glob("*.json"))
    by_cam = defaultdict(list)
    for s in stems:
        cam = s.rsplit("_", 1)[1]
        by_cam[cam].append(s)
    assign = {}
    for cam, lst in by_cam.items():
        lst.sort()
        n = len(lst)
        n_val = max(1, round(n * 0.1)) if n >= 10 else 0
        n_test = max(1, round(n * 0.1)) if n >= 10 else 0
        i_test = n - n_test
        i_val = i_test - n_val
        for i, s in enumerate(lst):
            assign[s] = "train" if i < i_val else ("valid" if i < i_test else "test")
    return assign, by_cam


def main():
    if OUT.exists():
        raise SystemExit(f"target already exists: {OUT}")
    for split in SPLITS:
        (OUT / split / "images").mkdir(parents=True)
        (OUT / split / "labels").mkdir(parents=True)

    counts = {s: Counter() for s in SPLITS}
    imgs = Counter()
    missing_v9_labels = 0
    recovered = 0

    # ---- 1. copy v9 verbatim ----
    for split in SPLITS:
        for img in sorted((SRC_V9 / split / "images").iterdir()):
            if not img.is_file():
                continue
            shutil.copy2(img, OUT / split / "images" / img.name)
            imgs[split] += 1
            src_lbl = SRC_V9 / split / "labels" / (img.stem + ".txt")
            dst_lbl = OUT / split / "labels" / (img.stem + ".txt")
            if src_lbl.is_file():
                text = src_lbl.read_text(encoding="utf-8").strip()
                dst_lbl.write_text((text + "\n") if text else "", encoding="utf-8")
                for line in text.splitlines():
                    if line.strip():
                        counts[split][int(line.split()[0])] += 1
            else:
                missing_v9_labels += 1
                dst_lbl.write_text("", encoding="utf-8")

    # ---- 2. convert 9.22 ----
    assign, by_cam = split_922()
    for jp in sorted(SRC_922.glob("*.json")):
        stem = jp.stem
        split = assign[stem]
        boxes = parse_xany(jp)
        img_src = SRC_922 / (stem + ".jpg")
        if not img_src.is_file():
            img_src = RECOVER / (stem + ".jpg")
            if not img_src.is_file():
                raise SystemExit(f"missing image for annotation: {stem}")
            recovered += 1
        shutil.copy2(img_src, OUT / split / "images" / (stem + ".jpg"))
        imgs[split] += 1
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes]
        (OUT / split / "labels" / (stem + ".txt")).write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        for c, *_ in boxes:
            counts[split][c] += 1

    # ---- 3. data.yaml ----
    names = "[" + ", ".join(f"'{n}'" for n in NAMES) + "]"
    (OUT / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
        f"nc: {len(NAMES)}\nnames: {names}\n", encoding="utf-8")

    # ---- 4. report ----
    print(f"output: {OUT}")
    print(f"9.22 recovered images from rtsp_mine_1200_merged: {recovered}")
    print(f"v9 images without label file (kept as background): {missing_v9_labels}")
    print(f"\n9.22 per-camera split (n images):")
    for cam, lst in sorted(by_cam.items()):
        c = Counter(assign[s] for s in lst)
        print(f"  {cam:14s} total={len(lst):4d}  train={c['train']:4d} valid={c['valid']:3d} test={c['test']:3d}")
    print(f"\nimages per split: " + "  ".join(f"{s}={imgs[s]}" for s in SPLITS),
          f"  total={sum(imgs.values())}")
    print(f"\n{'id':>2} {'name':<15} {'train':>7} {'valid':>6} {'test':>6}")
    for i, n in enumerate(NAMES):
        print(f"{i:>2} {n:<15} {counts['train'][i]:>7} {counts['valid'][i]:>6} {counts['test'][i]:>6}")
    print(f"{'':>2} {'TOTAL':<15} "
          f"{sum(counts['train'].values()):>7} {sum(counts['valid'].values()):>6} {sum(counts['test'].values()):>6}")


if __name__ == "__main__":
    main()
