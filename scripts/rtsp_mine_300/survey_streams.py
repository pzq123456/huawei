"""候选流质量调查：短时抽帧 -> merged12 模型检测 -> 统计每路质量。

目的：为"多车复杂 + 弱类(尤其 Motorcycle)优先"的采集选路定配额。
对每路统计：车流密度(平均框数)、复杂度(>=4框帧占比)、弱类/摩托车产量、
画面健康(灰屏/坏帧/静止)、以及一个综合评分。

用法：
  python scripts/rtsp_mine_300/survey_streams.py --duration 25 --workers 6
  python scripts/rtsp_mine_300/survey_streams.py --streams dahua1001636 dahua1001637 --duration 20
输出：output/rtsp_mine_300/survey_<ts>/survey.csv + survey.json
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from collect import StreamReader, is_gray_screen, phash  # noqa: E402

NAMES = ['Coach', 'Franchised Bus', 'HGV', 'LGV', 'Light Bus', 'MGV',
         'Motorcycle', 'PLB GMB', 'Private Car', 'Taxi', 'Van', 'Container']
WEAK = ['LGV', 'Container', 'HGV', 'Van', 'Motorcycle', 'MGV']
RARE_W = {'Container': 8.0, 'LGV': 5.0, 'HGV': 5.0, 'Motorcycle': 3.0,
          'Van': 3.0, 'MGV': 2.0}
GBL = threading.Lock()  # 模型推理串行化


def load_hosts(cfg_path: Path):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


HWY_MAP = {"hwy0548": "dahua548", "hwy0866": "dahua866", "hwy1003": "dahua1003"}
ICC_DEFS = {"cam2181": "1002605", "cam2200": "1002689",
            "cam2249": "1002712", "cam2250": "1002711"}


def icc_streams(code: str = "HK02"):
    from icc_rtsp_provider import ICCProvider, load_creds
    prov = ICCProvider(load_creds(code))
    out = []
    for name, dev in ICC_DEFS.items():
        try:
            url = prov.live_url(dev, 0)
            out.append((name, url))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] ICC {name} 取流失败: {str(e)[:120]}", flush=True)
    return out


def default_streams(host: str):
    """高速3路 + 全部已发现 dahua 城市路。"""
    out = []
    disc = ROOT / "output/rtsp_mine_300/discovered_streams_full.csv"
    if disc.is_file():
        for r in csv.DictReader(open(disc, encoding="utf-8")):
            out.append((r["name"], f"{host}/{r['name']}"))
    for name, path in HWY_MAP.items():
        out.append((name, f"{host}/{path}"))
    return out


def resolve_url(host: str, name: str) -> str:
    return f"{host}/{HWY_MAP.get(name, name)}"


def crop_frame(img):
    w = img.shape[1]
    if w >= 2600:
        return img[:, : w // 2], True
    return img, False


def survey_one(name, url, model, args, results, lock):
    import queue as _q
    reader = StreamReader(url, args.interval, name=name)
    reader.start()
    stats = {"name": name, "url": url, "connected": False}
    t0 = time.time()
    n_frame = n_det = 0
    boxes_hist = Counter()
    cls_total = Counter()
    cls_frames = Counter()
    complex_frames = multi_weak_frames = gray_frames = corrupt_frames = 0
    hashes = []
    try:
        while time.time() - t0 < args.duration:
            try:
                frame = reader.q.get(timeout=3.0)
            except _q.Empty:
                continue
            stats["connected"] = True
            n_frame += 1
            img, _ = crop_frame(frame)
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            white = float((g > 250).mean())
            black = float((g < 5).mean())
            bad, _ = is_gray_screen(img)
            corrupt = white > 0.35 or black > 0.50
            gray_frames += int(bad)
            corrupt_frames += int(corrupt)
            hashes.append(phash(img))
            with GBL:
                r = model.predict(img, conf=args.conf, imgsz=args.imgsz,
                                  verbose=False, device=args.device)[0]
            labels = []
            if r.boxes is not None and len(r.boxes):
                for c in r.boxes.cls.tolist():
                    labels.append(model.names[int(c)])
            n_det += 1
            boxes_hist[len(labels)] += 1
            for l in labels:
                cls_total[l] += 1
            for l in set(labels):
                cls_frames[l] += 1
            if len(labels) >= 4:
                complex_frames += 1
            if len(set(labels) & set(WEAK)) >= 2:
                multi_weak_frames += 1
    except Exception as e:  # noqa: BLE001
        stats["error"] = str(e)[:200]
    finally:
        reader.stopped = True

    dur = max(time.time() - t0, 1e-6)
    # 静止检测：连续帧 phash 汉明距离过小
    dup = 0
    for a, b in zip(hashes, hashes[1:]):
        if bin(a ^ b).count("1") < 3:
            dup += 1
    stats.update({
        "duration": round(dur, 1),
        "frames_read": n_frame,
        "frames_det": n_det,
        "fps": round(n_frame / dur, 1),
        "avg_boxes": round(sum(boxes_hist[k] * k for k in boxes_hist) / max(n_det, 1), 2),
        "complex_frac": round(complex_frames / max(n_det, 1), 3),
        "multi_weak_frac": round(multi_weak_frames / max(n_det, 1), 3),
        "gray_frac": round(gray_frames / max(n_det, 1), 3),
        "corrupt_frac": round(corrupt_frames / max(n_det, 1), 3),
        "static_frac": round(dup / max(len(hashes) - 1, 1), 3),
        "motorcycle": cls_total.get("Motorcycle", 0),
        "motorcycle_frames": cls_frames.get("Motorcycle", 0),
        "container": cls_total.get("Container", 0),
        "lgv": cls_total.get("LGV", 0),
        "hgv": cls_total.get("HGV", 0),
        "van": cls_total.get("Van", 0),
        "mgv": cls_total.get("MGV", 0),
        "private_car": cls_total.get("Private Car", 0),
        "taxi": cls_total.get("Taxi", 0),
        "class_total": dict(cls_total),
        "boxes_hist": dict(boxes_hist),
    })
    # 综合评分：复杂度 + 弱类加权(摩托额外加权) - 画质惩罚
    weak_score = sum(RARE_W.get(l, 0.0) * v for l, v in cls_total.items())
    health = 1.0 - min(1.0, stats["gray_frac"] + stats["corrupt_frac"] + stats["static_frac"])
    score = (stats["avg_boxes"] * 1.0 + stats["complex_frac"] * 6.0
             + stats["multi_weak_frac"] * 8.0 + weak_score * 0.15
             + cls_frames.get("Motorcycle", 0) * 1.5)
    stats["quality_score"] = round(score * health, 2)
    with lock:
        results.append(stats)
        print(f"[{name}] fps={stats['fps']} avg_box={stats['avg_boxes']} "
              f"cplx={stats['complex_frac']} moto={stats['motorcycle']}/{stats['motorcycle_frames']}f "
              f"weak={stats['multi_weak_frac']} gray={stats['gray_frac']} "
              f"corrupt={stats['corrupt_frac']} static={stats['static_frac']} "
              f"score={stats['quality_score']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="rtsp://118.140.234.166:8554")
    ap.add_argument("--streams", nargs="*", default=None, help="显式流名(不含host前缀)")
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--weights", default="runs/detect/yolo26m_merged12_20260922_0757/weights/best.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-icc", action="store_true", help="附加 ICC 4 路 (cam2181/2200/2249/2250)")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch
    model = YOLO(args.weights, task="detect")
    print(f"模型 {args.weights} 类别={list(model.names.values())}", flush=True)

    if args.streams:
        streams = [(n, resolve_url(args.host, n)) for n in args.streams]
    else:
        streams = default_streams(args.host)
    if args.include_icc:
        streams = streams + icc_streams()
    print(f"调查 {len(streams)} 路，每路 {args.duration}s，采样间隔 {args.interval}s，"
          f"并发 {args.workers}", flush=True)

    out_dir = ROOT / (args.out or f"output/rtsp_mine_300/survey_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results, lock = [], threading.Lock()
    sem = threading.Semaphore(args.workers)

    def worker(item):
        nonlocal results
        with sem:
            with torch.no_grad():
                survey_one(item[0], item[1], model, args, results, lock)

    threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in streams]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\n调查完成，用时 {(time.time()-t0)/60:.1f} min", flush=True)

    results.sort(key=lambda r: r.get("quality_score", 0), reverse=True)
    cols = ["name", "connected", "quality_score", "fps", "avg_boxes", "complex_frac",
            "multi_weak_frac", "motorcycle", "motorcycle_frames", "container", "lgv", "hgv",
            "van", "mgv", "private_car", "taxi", "gray_frac", "corrupt_frac", "static_frac",
            "frames_det"]
    with open(out_dir / "survey.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    (out_dir / "survey.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"\n{'name':<18}{'score':>7}{'avg':>6}{'cplx':>7}{'multiW':>7}{'moto':>6}"
          f"{'motoF':>6}{'cont':>6}{'lgv':>5}{'hgv':>5}{'van':>5}{'gray':>7}{'static':>7}")
    for r in results:
        print(f"{r['name']:<18}{r.get('quality_score',0):>7}{r.get('avg_boxes',0):>6}"
              f"{r.get('complex_frac',0):>7}{r.get('multi_weak_frac',0):>7}"
              f"{r.get('motorcycle',0):>6}{r.get('motorcycle_frames',0):>6}"
              f"{r.get('container',0):>6}{r.get('lgv',0):>5}{r.get('hgv',0):>5}"
              f"{r.get('van',0):>5}{r.get('gray_frac',0):>7}{r.get('static_frac',0):>7}")
    print(f"\n-> {out_dir}/survey.csv")


if __name__ == "__main__":
    main()
