"""用通用 COCO 模型(yolo26)交叉核验空镜：我方模型有框但 COCO 无车辆的帧。

用法:
  python scripts/rtsp_mine_300/filter_empty.py --dataset dataset/rtsp_mine_1200_merged --conf 0.25
  python scripts/rtsp_mine_300/filter_empty.py --dataset ... --apply   # 移出到 _removed_empty/
输出:
  output/rtsp_mine_300/empty_audit_<ds>.csv
  output/rtsp_mine_300/empty_flags_<ds>.jpg   # 被标记帧拼图（人工复核）
"""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]
VEH = {"car", "motorcycle", "bus", "truck"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset/rtsp_mine_1200_merged")
    ap.add_argument("--coco-weights", default=r"C:\Users\admin\Desktop\personal\gprBox\yolo26m.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--apply", action="store_true", help="把标记帧移到 quarantine")
    ap.add_argument("--remove-only-moto", action="store_true",
                    help="仅移除我方标签全为 Motorcycle 的标记帧（避免误伤 COCO 漏检的真车）")
    ap.add_argument("--config", default="scripts/rtsp_mine_300/config.yaml",
                    help="用于检查 conf_low_classes 的采集配置")
    ap.add_argument("--allow-bare-apply", action="store_true",
                    help="显式确认：明知存在低置信度补充框，仍允许裸 --apply")
    ap.add_argument("--quarantine", default="_removed_empty")
    ap.add_argument("--montage-max", type=int, default=48)
    args = ap.parse_args()

    # 硬检查：采集配置启用了低阈补充框时，禁止裸 --apply（避免被 COCO 漏检的弱类框误删）
    if args.apply and not args.remove_only_moto and not args.allow_bare_apply:
        cfgp = ROOT / args.config
        low = []
        if cfgp.is_file():
            _cfg = yaml.safe_load(open(cfgp, encoding="utf-8")) or {}
            low = (_cfg.get("model", {}) or {}).get("conf_low_classes") or []
        if low:
            raise SystemExit(
                f"[拒绝执行] {cfgp} 的 conf_low_classes 非空 {low}：低置信度补充框可能被 COCO "
                f"漏检而误判为空镜删除。请改用 --remove-only-moto，或显式加 --allow-bare-apply 确认。")

    from ultralytics import YOLO
    import torch

    ds = ROOT / args.dataset
    img_dir = ds / "images"
    ann_dir = ds / "annotations_xany"
    ann_dir = ann_dir if ann_dir.is_dir() else img_dir
    out = ROOT / "output/rtsp_mine_300"
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.coco_weights, task="detect")
    names = {int(k): v for k, v in model.names.items()} if isinstance(model.names, dict) else dict(enumerate(model.names))
    veh_ids = [i for i, n in names.items() if n in VEH]
    print(f"COCO vehicle ids: {[(i, names[i]) for i in veh_ids]}", flush=True)

    imgs = sorted(img_dir.glob("*.jpg"))
    rows, flagged = [], []
    with torch.no_grad():
        for i, img in enumerate(imgs):
            r = model.predict(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False,
                              device=args.device, classes=veh_ids)[0]
            coco = [names[int(c)] for c in r.boxes.cls.tolist()] if r.boxes is not None and len(r.boxes) else []
            ours = []
            jp = ann_dir / f"{img.stem}.json"
            if jp.is_file():
                ours = [s["label"] for s in json.load(open(jp, encoding="utf-8"))["shapes"]]
            rec = dict(file=img.name, coco_veh=len(coco), coco_labels=";".join(coco),
                       our_boxes=len(ours), our_labels=";".join(sorted(set(ours))))
            rows.append(rec)
            if len(coco) == 0:
                flagged.append(rec)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(imgs)}  flagged={len(flagged)}", flush=True)

    rows.sort(key=lambda r: (r["coco_veh"], r["our_boxes"]))
    report = out / f"empty_audit_{ds.name}.csv"
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "coco_veh", "coco_labels", "our_boxes", "our_labels"])
        w.writeheader()
        w.writerows(rows)

    print(f"\ntotal={len(rows)}  flagged(coco_veh==0)={len(flagged)}  "
          f"({len(flagged)/max(1,len(rows))*100:.1f}%)")
    print("of flagged, our-tail label mix:", dict(Counter(l for r in flagged for l in r["our_labels"].split(";") if l)))

    # montage
    tiles = []
    for rec in flagged[:args.montage_max]:
        im = cv2.imread(str(img_dir / rec["file"]))
        if im is None:
            continue
        jp = ann_dir / f"{Path(rec['file']).stem}.json"
        if jp.is_file():
            for s in json.load(open(jp, encoding="utf-8"))["shapes"]:
                (x1, y1), (x2, y2) = s["points"]
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.putText(im, s["label"], (int(x1), max(12, int(y1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.rectangle(im, (0, 0), (520, 24), (0, 0, 0), -1)
        cv2.putText(im, rec["file"].split("_")[-1] + " coco=0", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        tiles.append(cv2.resize(im, (520, 292)))
    if tiles:
        cols = 4
        rows_n = (len(tiles) + cols - 1) // cols
        tiles += [tiles[0] * 0] * (rows_n * cols - len(tiles))
        sheet = cv2.vconcat([cv2.hconcat(tiles[i * cols:(i + 1) * cols]) for i in range(rows_n)])
        mp = out / f"empty_flags_{ds.name}.jpg"
        cv2.imwrite(str(mp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"montage -> {mp}")

    if args.apply:
        q = ds / args.quarantine
        q.mkdir(parents=True, exist_ok=True)
        to_remove = flagged
        if args.remove_only_moto:
            to_remove = [r for r in flagged if set(r["our_labels"].split(";")) == {"Motorcycle"}]
            kept = [r for r in flagged if set(r["our_labels"].split(";")) != {"Motorcycle"}]
            (out / f"empty_kept_coco_miss_{ds.name}.csv").write_text(
                "file,our_labels,coco_labels\n" + "\n".join(
                    f"{r['file']},{r['our_labels']},{r['coco_labels']}" for r in kept), encoding="utf-8")
            print(f"kept (COCO miss, review): {len(kept)} -> empty_kept_coco_miss_{ds.name}.csv")
        for rec in to_remove:
            shutil.move(str(img_dir / rec["file"]), str(q / rec["file"]))
            jp = ann_dir / f"{Path(rec['file']).stem}.json"
            if jp.is_file():
                shutil.move(str(jp), str(q / jp.name))
        print(f"moved {len(to_remove)} frames -> {q}")

    print(f"report -> {report}")


if __name__ == "__main__":
    main()
