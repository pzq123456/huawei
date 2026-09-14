"""合并多批 RTSP 采集集：图片 + 同名 JSON 同目录（X-AnyLabeling 识别要求）。

用法:
  python scripts/rtsp_mine_300/merge_batches.py \
    --sources dataset/rtsp_mine_300_pending dataset/rtsp_mine_300_b2_pending \
    --out dataset/rtsp_mine_600_merged
输出:
  <out>/images/{*.jpg, *.json}   同名校对
  <out>/classes.txt, manifest.csv, README.md
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
WEAK = {'LGV', 'HGV', 'Motorcycle', 'PLB GMB', 'Coach', 'Light Bus', 'Van'}


def phash(img, hs=8):
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (hs + 1, hs))
    d = (small[:, 1:] > small[:, :-1])
    return sum(1 << i for i, v in enumerate(d.flatten()) if v)


def ham(a, b):
    return bin(a ^ b).count('1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dedup-thresh", type=int, default=8)
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    out_img = out / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    records = []
    hashes = []            # (hash, file)
    dropped_dup = []
    near_dups = 0
    missing_json = []
    for si, src in enumerate(args.sources):
        src = ROOT / src
        batch = f"b{si + 1}"
        imgs = sorted((src / "images").glob("*.jpg"))
        for img in imgs:
            jp = src / "annotations_xany" / f"{img.stem}.json"
            if not jp.is_file():
                missing_json.append(img.name)
                continue
            d = json.load(open(jp, encoding="utf-8"))
            labels = [s["label"] for s in d["shapes"]]
            h = phash(cv2.imread(str(img)))
            match = next(((hh, f) for hh, f in hashes if ham(h, hh) < args.dedup_thresh), None)
            if match is not None:
                near_dups += 1
            if match is not None and not args.no_dedup:
                dropped_dup.append((img.name, match[1]))
                continue
            hashes.append((h, img.name))
            # copy image + co-located same-name json
            shutil.copy2(img, out_img / img.name)
            shutil.copy2(jp, out_img / f"{img.stem}.json")
            ts = img.stem
            cam = ts.split("_")[-1]
            rare = [l for l in labels if l in {'LGV', 'HGV', 'Motorcycle', 'PLB GMB'}]
            records.append(dict(
                file=img.name, batch=batch, cam=cam, timestamp=ts.split("_")[-2] if "_" in ts else "",
                n_box=len(labels), n_cls=len(set(labels)),
                rare_hit=";".join(sorted(set(rare))),
                weak_hit=int(bool(set(labels) & WEAK))))

    # classes.txt from first source
    classes_src = ROOT / args.sources[0] / "classes.txt"
    (out / "classes.txt").write_text(classes_src.read_text(encoding="utf-8"), encoding="utf-8")

    # manifest
    import csv
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "batch", "cam", "timestamp", "n_box", "n_cls", "rare_hit", "weak_hit"])
        w.writeheader()
        w.writerows(records)

    # stats
    lab = Counter()
    for r in records:
        d = json.load(open(out_img / f"{Path(r['file']).stem}.json", encoding="utf-8"))
        for s in d["shapes"]:
            lab[s["label"]] += 1
    per_batch = Counter(r["batch"] for r in records)
    per_cam = Counter(r["cam"] for r in records)
    weak_frames = sum(r["weak_hit"] for r in records)
    print(f"merged {len(records)} images -> {out_img}")
    print("per-batch:", dict(per_batch))
    print("per-cam:", dict(per_cam))
    print(f"weak-class frames: {weak_frames}/{len(records)}")
    print(f"near-duplicate pairs (pHash hamming <{args.dedup_thresh}, info only): {near_dups}")
    print("dropped duplicates:", len(dropped_dup))
    for a, b in dropped_dup[:10]:
        print(f"  dup {a} ~ {b}")
    print("missing json:", len(missing_json))
    print("class dist:", dict(lab.most_common()))

    (out / "README.md").write_text(
        "# rtsp_mine_600_merged（待标注）\n\n"
        f"两批 RTSP 采集合并，共 **{len(records)}** 张；图片与**同名 JSON 同目录**（`images/`），"
        "X-AnyLabeling 打开 `images/` 即自动关联标注。\n\n"
        f"- 弱类帧：{weak_frames}/{len(records)}\n"
        f"- 类别：见 `classes.txt`（11 类 v9 口径）\n"
        f"- 清单：`manifest.csv`（含 batch/cam/n_box/n_cls/rare_hit/weak_hit）\n"
        f"- 去重：pHash 汉明 <{args.dedup_thresh} 视为重复，丢弃 {len(dropped_dup)} 张\n",
        encoding="utf-8")
    print(f"classes.txt / manifest.csv / README.md -> {out}")


if __name__ == "__main__":
    main()
