"""核验 rtsp_mine_300_pending 数据集：尺寸/灰屏/JSON合法性/分布 + 生成预览拼图。

用法: python scripts/rtsp_mine_300/verify_dataset.py
输出: output/rtsp_mine_300/verify_report.json, output/rtsp_mine_300/dataset_preview.jpg
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "dataset/rtsp_mine_300_pending"
OUT = ROOT / "output/rtsp_mine_300"


def main():
    global DS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/rtsp_mine_300_pending")
    ap.add_argument("--preview", default="dataset_preview.jpg")
    args = ap.parse_args()
    DS = ROOT / args.dataset
    print(f"dataset: {DS}")
    imgs = sorted((DS / "images").glob("*.jpg"))
    ann = DS / "annotations_xany"
    ann_dir = ann if ann.is_dir() else (DS / "images")
    jsons = {p.stem: p for p in ann_dir.glob("*.json")}
    classes = [l.strip() for l in (DS / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

    sizes = Counter()
    mins = []
    bad_json, missing_json, gray = [], [], []
    labels = Counter()
    boxes_per_img = []
    stds = []
    for img in imgs:
        im = cv2.imread(str(img))
        h, w = im.shape[:2]
        sizes[f"{w}x{h}"] += 1
        std = float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).std())
        stds.append(std)
        if std < 12:
            gray.append(img.name)
        jp = jsons.get(img.stem)
        if jp is None:
            missing_json.append(img.name)
            continue
        d = json.load(open(jp, encoding="utf-8"))
        for k in ("version", "flags", "shapes", "imagePath", "imageHeight", "imageWidth"):
            if k not in d:
                bad_json.append(f"{jp.name}:missing {k}")
        if d.get("imageWidth") != w or d.get("imageHeight") != h:
            bad_json.append(f"{jp.name}:size mismatch")
        n = 0
        for s in d["shapes"]:
            n += 1
            labels[s["label"]] += 1
            if s["shape_type"] != "rectangle" or len(s["points"]) != 2:
                bad_json.append(f"{jp.name}:bad shape")
            if s["label"] not in classes:
                bad_json.append(f"{jp.name}:label not in classes.txt: {s['label']}")
        boxes_per_img.append(n)

    report = dict(
        images=len(imgs), jsons=len(jsons), classes=classes,
        sizes=dict(sizes), min_gray_std=round(min(stds), 1),
        gray_images=gray, missing_json=missing_json, bad_json=bad_json,
        boxes_per_img_avg=round(sum(boxes_per_img) / max(1, len(boxes_per_img)), 2),
        prelabel_class_dist=dict(labels.most_common()),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    report_name = f"verify_report_{DS.name}.json"
    (OUT / report_name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    random.seed(0)
    sample = random.sample(imgs, min(30, len(imgs)))
    tiles = []
    for img in sample:
        im = cv2.imread(str(img))
        d = json.load(open(jsons[img.stem], encoding="utf-8"))
        for s in d["shapes"]:
            (x1, y1), (x2, y2) = s["points"]
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(im, s["label"], (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(im, img.stem.split("_")[-1], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        tiles.append(cv2.resize(im, (448, 252)))
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    tiles += [tiles[0] * 0] * (rows * cols - len(tiles))
    sheet = cv2.vconcat([cv2.hconcat(tiles[i * cols:(i + 1) * cols]) for i in range(rows)])
    cv2.imwrite(str(OUT / args.preview), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"\npreview -> {OUT / args.preview}")


if __name__ == "__main__":
    main()
