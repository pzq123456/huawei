"""对已发现的 RTSP 流抓帧并用 11 类模型打分，筛出车辆多/稀有类多的机位。

用法:
  python scripts/rtsp_mine_300/rank_streams.py --frames 4 --spacing 2.5 --workers 6
输出:
  output/rtsp_mine_300/stream_ranking.csv      每路聚合指标(按加权分排序)
  output/rtsp_mine_300/scan_frames/<name>/*.jpg  抓取的原帧
  output/rtsp_mine_300/top_streams_montage.jpg   Top 机位拼接预览
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from collect import is_gray_screen, score_dets, phash  # noqa: E402

RARE_W = {"LGV": 5.0, "HGV": 5.0, "Motorcycle": 3.0, "PLB GMB": 2.0,
          "Coach": 1.5, "Light Bus": 1.5, "Van": 1.5, "MGV": 1.0,
          "Bus": 1.5, "Truck": 5.0, "Franchised Bus": 0.5, "Taxi": 0.2, "Private Car": 0.1}


def crop_auto(img, mode="left"):
    w = img.shape[1]
    if mode in ("left", "auto-left") and w >= 2600:
        return img[:, : w // 2]
    if mode == "right" and w >= 2600:
        return img[:, w // 2:]
    return img


def grab_frames(url: str, k: int, spacing: float, timeout_s: float = 6.0, max_wait: float = 30.0):
    """连一次流，按墙钟间隔抓 k 帧。返回 bgr ndarrays。"""
    out, last, t0 = [], 0.0, time.time()
    try:
        c = av.open(url, options={"rtsp_transport": "tcp", "stimeout": str(int(timeout_s * 1e6))})
    except Exception as e:  # noqa: BLE001
        return out, str(e)
    try:
        for frame in c.decode(video=0):
            now = time.time()
            if now - t0 > max_wait:
                break
            if now - last < spacing and out:
                continue
            img = frame.to_ndarray(format="bgr24")
            out.append(img)
            last = now
            if len(out) >= k:
                break
    except Exception as e:  # noqa: BLE001
        return out, str(e)
    finally:
        try:
            c.close()
        except Exception:
            pass
    return out, None


def capture_one(name, url, outdir, k, spacing):
    frames, err = grab_frames(url, k, spacing)
    saved = []
    for i, img in enumerate(frames):
        p = outdir / name / f"f{i}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), crop_auto(img), [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved.append(str(p))
    return name, saved, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="rtsp://118.140.234.166:8554")
    ap.add_argument("--list", default="output/rtsp_mine_300/discovered_streams_full.csv")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=2.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--weights", default="runs/detect/yolo26m_merge8_20260903_0950/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--top-montage", type=int, default=20)
    args = ap.parse_args()

    with open(ROOT / args.list, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    streams = [r["name"] for r in rows]
    print(f"{len(streams)} streams, grabbing {args.frames} frames each @ {args.spacing}s", flush=True)

    frames_dir = ROOT / "output/rtsp_mine_300/scan_frames"
    captured = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(capture_one, n, f"{args.host}/{n}", frames_dir, args.frames, args.spacing): n
                for n in streams}
        done = 0
        for fut in as_completed(futs):
            name, saved, err = fut.result()
            done += 1
            captured[name] = saved
            print(f"[{done}/{len(streams)}] {name}: {len(saved)} frames" + (f" ERR={err}" if err else ""),
                  flush=True)

    from ultralytics import YOLO
    import torch
    model = YOLO(str(ROOT / args.weights), task="detect")
    names = {int(k): v for k, v in model.names.items()} if isinstance(model.names, dict) else \
        dict(enumerate(model.names))

    records = []
    for name in streams:
        files = captured.get(name, [])
        if not files:
            records.append(dict(name=name, n_frames=0, n_veh_total=0, n_veh_max=0, n_cls=0,
                                weighted_max=0.0, weighted_mean=0.0, gray_frames=0, rare_hit="", err=1))
            continue
        veh_total, veh_max, cls_set, wmax, wsum, gray_cnt, rare_cnt = 0, 0, set(), 0.0, 0.0, 0, {}
        for fp in files:
            img = cv2.imread(fp)
            bad, _ = is_gray_screen(img)
            if bad:
                gray_cnt += 1
                continue
            r = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False, device=args.device)[0]
            labels = [names[int(c)] for c in r.boxes.cls.tolist()] if r.boxes is not None and len(r.boxes) else []
            veh_total += len(labels)
            veh_max = max(veh_max, len(labels))
            cls_set |= set(labels)
            sc = score_dets(labels, RARE_W, 1.0, 0.3)
            wmax = max(wmax, sc)
            wsum += sc
            for l in labels:
                if RARE_W.get(l, 0) >= 2.0:
                    rare_cnt[l] = rare_cnt.get(l, 0) + 1
        nf = max(1, len(files))
        records.append(dict(
            name=name, n_frames=len(files), n_veh_total=veh_total, n_veh_max=veh_max,
            n_cls=len(cls_set), weighted_max=round(wmax, 1), weighted_mean=round(wsum / nf, 2),
            gray_frames=gray_cnt,
            rare_hit=";".join(f"{k}:{v}" for k, v in sorted(rare_cnt.items(), key=lambda x: -x[1])),
            err=0))

    records.sort(key=lambda r: (r["weighted_max"], r["n_veh_max"], r["n_veh_total"]), reverse=True)
    out = ROOT / "output/rtsp_mine_300/stream_ranking.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    print("\n=== ranking (top 25) ===")
    for r in records[:25]:
        print(f"  {r['name']:16s} wmax={r['weighted_max']:6.1f} veh_max={r['n_veh_max']:2d} "
              f"veh_tot={r['n_veh_total']:3d} cls={r['n_cls']} gray={r['gray_frames']} {r['rare_hit']}")

    # montage of top-N (first available frame)
    top = [r["name"] for r in records[:args.top_montage]]
    tiles = []
    for name in top:
        files = captured.get(name, [])
        if not files:
            continue
        img = cv2.imread(files[0])
        if img is None:
            continue
        img = cv2.resize(img, (480, 270))
        rec = next(r for r in records if r["name"] == name)
        cv2.rectangle(img, (0, 0), (480, 26), (0, 0, 0), -1)
        cv2.putText(img, f"{name} wmax={rec['weighted_max']:.0f} vmax={rec['n_veh_max']}",
                    (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(img)
    if tiles:
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        blank = (tiles[0] * 0).copy()
        while len(tiles) < rows * cols:
            tiles.append(blank.copy())
        sheet = cv2.vconcat([cv2.hconcat(tiles[i * cols:(i + 1) * cols]) for i in range(rows)])
        mp = ROOT / "output/rtsp_mine_300/top_streams_montage.jpg"
        cv2.imwrite(str(mp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\nmontage -> {mp}")

    print(f"ranking -> {out}")


if __name__ == "__main__":
    main()
