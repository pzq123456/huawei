"""14路混采：老路 + 高速3路，按 quota 分权重，弱类优先 + 复杂画面门槛 + 进度打印。

和 collect.py 的区别：
  1. streams 各自带 quota / detect_interval / min_save_interval，配额满即停该路；
  2. 复杂画面门槛：超稀有类(LGV/Container/Motorcycle/HGV)允许单框，
     其余要求 >=3框 或 (>=2框且>=2类)，纯 Private Car/Taxi 帧一律不要；
  3. 进度打印：每次落盘 + 每30s心跳（总进度/各路quota/弱类直方图/丢弃原因/ETA）。

用法：
  python scripts/rtsp_mine_300/collect_highway15.py --smoke-test
  python scripts/rtsp_mine_300/collect_highway15.py --target 600
"""
import argparse
import csv
import json
import os
import queue
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from collect import (  # noqa: E402
    StreamReader,
    bbox_touches_edge,
    is_gray_screen,
    phash,
    roi_sharpness,
    score_dets,
)
from export_xany import save as save_json  # noqa: E402
from export_xany import to_xany  # noqa: E402

HEARTBEAT_SECS = 30.0


def load_config(path: Path) -> dict:
    import os
    os.chdir(ROOT)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fmt_eta(elapsed: float, done: int, total: int) -> str:
    if done <= 0 or elapsed <= 0:
        return "--"
    rate = done / elapsed
    left = max(0, total - done) / max(rate, 1e-6)
    m, s = int(left // 60), int(left % 60)
    return f"{m:02d}:{s:02d}"


def print_progress(total_saved, target, t_start, per_cam, quotas, weak_hist, reasons):
    el = time.time() - t_start
    pct = 100.0 * total_saved / max(target, 1)
    line = f"[进度 {total_saved}/{target} {pct:5.1f}% | 用时{el/60:.1f}min ETA{fmt_eta(el, total_saved, target)}]"
    print(line, flush=True)
    qparts = " ".join(f"{c}:{per_cam.get(c, 0)}/{quotas.get(c, 0)}"
                      for c in sorted(quotas))
    print(f"  各路: {qparts}", flush=True)
    if weak_hist:
        top = " ".join(f"{k}x{v}" for k, v in weak_hist.most_common(8))
        print(f"  弱类: {top}", flush=True)
    if reasons:
        top = " ".join(f"{k}:{v}" for k, v in Counter(reasons).most_common(6))
        print(f"  丢弃: {top}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scripts/rtsp_mine_300/config_highway15.yaml")
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--json-mode", choices=["empty", "prelabel"], default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="最长运行分钟数，0=不限；到点优雅退出（防配额不齐时空转）")
    args = ap.parse_args()

    # 单实例锁：防止双进程同时写同一份数据集（Windows msvcrt 文件锁）
    lock_path = ROOT / "output/rtsp_mine_300/collect_highway15.pid"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import msvcrt
        lock_f = open(lock_path, "w")
        try:
            msvcrt.locking(lock_f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("已有采集进程在跑（pid锁被占），退出以避免双写污染 manifest")
        lock_f.write(str(os.getpid()))
        lock_f.flush()
    except ImportError:
        lock_f = None  # 非 Windows：无锁运行

    cfg = load_config(Path(args.config))
    target = args.target or cfg["sampling"]["target_total"]
    json_mode = args.json_mode or cfg["output"]["json_mode"]
    names = cfg["names_merged12"]
    quotas = {s["name"]: int(s.get("quota", 0)) for s in cfg["streams"]}
    # quota 全0时退化为均分
    if sum(quotas.values()) <= 0:
        q = max(1, target // len(cfg["streams"]))
        quotas = {s["name"]: q for s in cfg["streams"]}
    target = min(target, sum(quotas.values()))

    from ultralytics import YOLO
    weights = Path(cfg["model"]["weights"])
    if not weights.is_file():
        weights = Path(cfg["model"]["fallback"])
        print(f"[warn] 主权重缺失，回退到 {weights}", flush=True)
    model = YOLO(str(weights), task="detect")
    print(f"模型 {weights} 类别={model.names}", flush=True)

    ds_dir = ROOT / (args.out_dir or cfg["output"]["dataset_dir"])
    img_dir = ds_dir / "images"
    js_dir = ds_dir / "annotations_xany"
    img_dir.mkdir(parents=True, exist_ok=True)
    js_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    run_log = ROOT / cfg["output"]["run_log_dir"] / f"run_highway15_{datetime.now():%Y%m%d_%H%M%S}.csv"
    run_log.parent.mkdir(parents=True, exist_ok=True)

    icc_prov = None
    if any(s.get("icc_device") for s in cfg["streams"]):
        from icc_rtsp_provider import ICCProvider, load_creds
        icc_code = next(s.get("icc_code", "HK02") for s in cfg["streams"] if s.get("icc_device"))
        icc_prov = ICCProvider(load_creds(icc_code))
        print(f"ICC provider 就绪 code={icc_code}", flush=True)

    def make_reader(s):
        interval = float(s.get("detect_interval", cfg["sampling"]["detect_interval"]))
        if not s.get("icc_device"):
            return StreamReader(s["url"], interval, name=s["name"])
        dev, ch = str(s["icc_device"]), int(s.get("icc_channel", 0))

        def resolve():
            d = icc_prov.start_video(dev, ch)
            return f"{d['url'].split('|')[-1]}&token={d['token']}"

        def release():
            icc_prov.stop_video(dev, ch)

        return StreamReader(resolve, interval, name=s["name"], on_stop=release)

    readers = [make_reader(s) for s in cfg["streams"]]
    for r in readers:
        r.start()

    model_names = {int(k): v for k, v in model.names.items()} if isinstance(model.names, dict) else \
        {i: v for i, v in enumerate(model.names)}
    if set(model_names.values()) != set(names):
        print(f"[warn] 模型类别与配置不完全一致，prelabel按模型名输出：{list(model_names.values())}", flush=True)

    def apply_crop(img, mode):
        w = img.shape[1]
        if mode in ("left", "auto-left") and w >= 2600:
            return img[:, : w // 2], "right"
        if mode == "right" and w >= 2600:
            return img[:, w // 2:], "left"
        return img, None

    # resume：按文件名后缀统计各路已有张数
    per_cam = Counter()
    existing = 0
    if args.resume:
        for p in img_dir.glob("*.jpg"):
            stem = p.stem
            cam = stem.rsplit("_", 1)[-1] if "_" in stem else ""
            if cam in quotas:
                per_cam[cam] += 1
                existing += 1
    total_saved = 0
    target_new = target  # 本次要新采的张数（resume 时 total 目标含已有）
    if args.resume:
        print(f"resume: 已有 {existing} 张，各路 {dict(per_cam)}", flush=True)

    last_save = {s["name"]: 0.0 for s in cfg["streams"]}
    last_hash: dict = {}
    weak_hist = Counter()
    reasons: list = []
    cls_cam_cnt = Counter()
    cap_state_path = ds_dir / "soft_cap_state.json"
    if args.resume and cap_state_path.is_file():
        try:
            raw = json.loads(cap_state_path.read_text(encoding="utf-8"))
            cls_cam_cnt.update({tuple(k.split("\t", 1)): int(v) for k, v in raw.items()})
        except Exception as e:  # noqa: BLE001
            print(f"[warn] soft-cap state 读取失败: {e}", flush=True)

    def dump_cap_state():
        try:
            cap_state_path.write_text(
                json.dumps({f"{c}\t{k}": v for (c, k), v in cls_cam_cnt.items()}, ensure_ascii=False),
                encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] soft-cap state 写入失败: {e}", flush=True)

    bal = cfg.get("balance", {})
    weak_classes = set(bal.get("weak_classes", []))
    drop_single = bool(bal.get("drop_single_private_car", False))
    soft_cap = int(bal.get("soft_cam_class_cap", 0) or 0)
    soft_drop = float(bal.get("soft_cap_drop_prob", 0.8))
    cx = cfg.get("complexity", {}) or {}
    ultra = set(cx.get("ultra_keep_singleton", ["LGV", "Container", "Motorcycle", "HGV"]))
    min_boxes = int(cx.get("min_boxes", 3))
    min_mc = int(cx.get("min_boxes_multiclass", 2))
    target_classes = set(cx.get("target_classes", list(ultra)))   # 含目标类可从宽
    allow_with_target = int(cx.get("allow_with_target", min_mc))
    moto_singleton = bool(cx.get("motorcycle_singleton", False))
    weak_min_boxes = int(cx.get("weak_min_boxes", min_boxes))
    target_min_boxes = int(cx.get("target_min_boxes", allow_with_target))
    noweak_min_boxes = int(cx.get("no_weak_min_boxes", min_boxes))
    noweak_min_classes = int(cx.get("no_weak_min_classes", 3))
    require_weak = bool(bal.get("require_weak", False))
    noweak_max_ratio = float(bal.get("no_weak_max_ratio", 0.0))

    oscore = float(cfg["model"].get("conf_scoring", 0.30))
    conf_low = float(cfg["model"].get("conf_low", oscore))
    low_classes = set(cfg["model"].get("conf_low_classes", []))
    conf_ov = {k: float(v) for k, v in (cfg["model"].get("conf_overrides") or {}).items()}
    per_class_conf = {c: (conf_low if c in low_classes else oscore) for c in names}
    per_class_conf.update(conf_ov)
    conf_floor = min(per_class_conf.values())

    clean_cfg = cfg.get("clean_capture", {}) or {}
    clean_on = bool(clean_cfg.get("enabled", False))
    edge_margin = float(clean_cfg.get("edge_margin", 2.0))
    min_area_frac = float(clean_cfg.get("min_box_area_frac", 0.0))
    lap_min = float(clean_cfg.get("laplacian_var_min", 0.0))
    uncond_interval = float(cfg["sampling"].get("unconditional_interval", 0.0))
    uncond_max = int(target * float(cfg["sampling"].get("unconditional_max_ratio", 0.0)))
    last_uncond = {s["name"]: 0.0 for s in cfg["streams"]}
    uncond_saved = 0
    noweak_saved = 0
    noweak_max = int(target * noweak_max_ratio)

    manifest_path = ds_dir / "manifest.csv"
    manifest_exists = manifest_path.exists() and manifest_path.stat().st_size > 0
    manifest_f = open(manifest_path, "a" if (args.resume and manifest_exists) else "w",
                      newline="", encoding="utf-8")
    manifest = csv.writer(manifest_f)
    if not (args.resume and manifest_exists):
        manifest.writerow(["file", "cam", "timestamp", "score", "n_box", "n_cls",
                           "rare_hit", "weak_hit", "json_mode", "uncond", "n_filtered"])

    print(f"开始混采：14路（高速3+老路11），本次新采目标={target_new} 总配额={sum(quotas.values())} "
          f"输出={ds_dir}", flush=True)
    print(f"配额: {quotas}", flush=True)
    print(f"弱类={sorted(weak_classes)} 复杂门槛: 超稀有{sorted(ultra)}允许单框, "
          f"其余>={min_boxes}框或(>={min_mc}框且>=2类)，纯私家车/Taxi不要", flush=True)

    t_start = time.time()
    last_heartbeat = t_start
    smoke_deadline = 240.0
    quota_met_logged = set()
    smoke_seen = Counter()

    with open(run_log, "w", newline="", encoding="utf-8") as log_f:
        log = csv.writer(log_f)
        log.writerow(["file", "cam", "score", "n_box", "n_cls", "rare_hit", "saved", "reason"])
        try:
            import torch
            with torch.no_grad():
                while total_saved < target_new:
                    if args.max_minutes and time.time() - t_start > args.max_minutes * 60:
                        print(f"[到点] 已运行 {args.max_minutes:.0f} 分钟，优雅退出", flush=True)
                        break
                    if all(per_cam.get(s["name"], 0) >= quotas[s["name"]] for s in cfg["streams"]):
                        print("全部配额已满，提前结束", flush=True)
                        break
                    for s, r in zip(cfg["streams"], readers):
                        cam = s["name"]
                        if per_cam.get(cam, 0) >= quotas[cam]:
                            if cam not in quota_met_logged:
                                print(f"[配额满] {cam} {per_cam[cam]}/{quotas[cam]}，该路停止", flush=True)
                                quota_met_logged.add(cam)
                            continue
                        try:
                            frame = r.q.get(timeout=1.0)
                        except queue.Empty:
                            if args.smoke_test and time.time() - t_start > smoke_deadline:
                                raise SystemExit(f"smoke-test 超时，仅收到 {total_saved} 帧，检查RTSP")
                            continue
                        frame, seam = apply_crop(frame, s.get("crop"))
                        H, W = frame.shape[:2]
                        res = model.predict(frame, conf=conf_floor,
                                            imgsz=cfg["model"]["imgsz"], verbose=False,
                                            device=cfg["model"]["device"])[0]
                        raw = []
                        if res.boxes is not None and len(res.boxes):
                            for c, cf, xy in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(),
                                                 res.boxes.xyxy.tolist()):
                                raw.append((model_names[int(c)], float(cf), [float(v) for v in xy]))
                        kept, n_filtered = [], 0
                        for label, cf, xy in raw:
                            if cf < per_class_conf.get(label, oscore):
                                n_filtered += 1
                                continue
                            if clean_on:
                                if bbox_touches_edge(xy, W, H, seam, edge_margin):
                                    n_filtered += 1
                                    continue
                                if min_area_frac > 0 and (xy[2] - xy[0]) * (xy[3] - xy[1]) < min_area_frac * W * H:
                                    n_filtered += 1
                                    continue
                                if lap_min > 0 and roi_sharpness(frame, xy) < lap_min:
                                    n_filtered += 1
                                    continue
                            kept.append((label, cf, xy))
                        labels = [k[0] for k in kept]
                        sc = score_dets(labels, cfg["scoring"]["rare_w"],
                                        cfg["scoring"]["w_cls"], cfg["scoring"]["w_box"])
                        rare_hit = next((l for l in labels if cfg["scoring"]["rare_w"].get(l, 0) >= 2.0), "")
                        weak_hit = next((l for l in labels if l in weak_classes), "")
                        now = time.time()
                        h = phash(frame)
                        reason = "ok"
                        gf = cfg["sampling"].get("gray_filter", {})
                        bad, metric = is_gray_screen(
                            frame, std_thresh=float(gf.get("std_thresh", 12.0)),
                            sat_thresh=float(gf.get("sat_thresh", 12.0)),
                            chan_diff_thresh=float(gf.get("chan_diff_thresh", 8.0)))
                        # 解码坏帧：大面积纯白/纯黑（灰屏过滤器看不见这类半幅白块）
                        _g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        white_ratio = float((_g > 250).mean())
                        black_ratio = float((_g < 5).mean())
                        corrupt = white_ratio > 0.35 or black_ratio > 0.50
                        if args.smoke_test:
                            print(f"[smoke] {cam} {frame.shape} gray=({metric}) "
                                  f"wb={white_ratio:.2f}/{black_ratio:.2f} dets={labels} score={sc:.1f}",
                                  flush=True)
                            total_saved += 1
                            smoke_seen[cam] += 1
                            live_cams = [c for c in quotas if smoke_seen[c] > 0]
                            if live_cams and (all(smoke_seen[c] >= 3 for c in live_cams)
                                              or time.time() - t_start > 90):
                                print(f"smoke ok 有效流={live_cams}", flush=True)
                                return
                            continue
                        min_interval = float(s.get("min_save_interval",
                                                 cfg["sampling"]["min_save_interval"]))
                        uncond_due = (uncond_interval > 0 and uncond_saved < uncond_max
                                      and now - last_uncond[cam] >= uncond_interval)
                        if corrupt:
                            reason = "corrupt-frame"
                        elif bad:
                            reason = "gray-screen"
                        elif uncond_due:
                            reason = "uncond"
                        elif not labels:
                            reason = "no-det" if not raw else "unclean"
                        elif set(labels) <= {"Private Car", "Taxi"}:
                            reason = "strong-only"
                        else:
                            n_lab, n_cls, lab_set = len(labels), len(set(labels)), set(labels)
                            has_weak = bool(lab_set & weak_classes)
                            has_target = bool(lab_set & target_classes)
                            if has_weak or require_weak:
                                # 含弱类(或强制弱类)：目标类>=2框，其余弱类>=weak_min_boxes，
                                # 摩托车单框放行；不够复杂则丢
                                ok_complex = (
                                    (moto_singleton and n_lab == 1 and labels[0] == "Motorcycle")
                                    or (has_target and n_lab >= target_min_boxes)
                                    or n_lab >= weak_min_boxes
                                )
                                if not ok_complex:
                                    reason = "too-simple"
                                elif require_weak and not has_weak:
                                    reason = "no-weak"
                            else:
                                # 无弱类：只收真正多车多类的复杂帧，且总量限流
                                if n_lab < noweak_min_boxes or n_cls < noweak_min_classes:
                                    reason = "no-weak"
                                elif noweak_saved >= noweak_max:
                                    reason = "no-weak-cap"
                        if reason == "ok":
                            if now - last_save[cam] < min_interval:
                                reason = "interval"
                            elif any(bin(h ^ ph).count("1") < cfg["sampling"]["phash_thresh"]
                                     for ph in last_hash.values()):
                                reason = "dup-phash"
                            elif soft_cap > 0 and any(cls_cam_cnt[(cam, c)] >= soft_cap
                                                      for c in set(labels)) and random.random() < soft_drop:
                                reason = "soft-cap"
                        if reason in ("ok", "uncond"):
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            fname = f"{ts}_{cam}.jpg"
                            cv2.imwrite(str(img_dir / fname), frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, cfg["output"]["jpeg_quality"]])
                            dets = [(l, xy[0], xy[1], xy[2], xy[3], cf) for l, cf, xy in kept] \
                                if json_mode == "prelabel" else []
                            save_json(js_dir / (Path(fname).stem + ".json"), to_xany(fname, W, H, dets))
                            total_saved += 1
                            per_cam[cam] += 1
                            last_save[cam] = now
                            last_hash[cam] = h
                            if reason == "uncond":
                                last_uncond[cam] = now
                                uncond_saved += 1
                            for c in set(labels):
                                cls_cam_cnt[(cam, c)] += 1
                            if weak_hit:
                                weak_hist[weak_hit] += 1
                            else:
                                noweak_saved += 1
                            dump_cap_state()
                            el = time.time() - t_start
                            pct = 100.0 * total_saved / max(target_new, 1)
                            print(f"[{total_saved}/{target_new} {pct:5.1f}%] {fname} "
                                  f"{cam}({per_cam[cam]}/{quotas[cam]}) score={sc:.1f} "
                                  f"n={len(labels)} {labels} ETA{fmt_eta(el, total_saved, target_new)}",
                                  flush=True)
                            manifest.writerow([fname, cam, ts, f"{sc:.2f}", len(labels),
                                               len(set(labels)), rare_hit, weak_hit, json_mode,
                                               int(reason == "uncond"), n_filtered])
                            manifest_f.flush()
                        else:
                            reasons.append(reason)
                        log.writerow([f"{cam}_{int(now)}", cam, f"{sc:.2f}", len(labels),
                                      len(set(labels)), rare_hit, int(reason in ("ok", "uncond")), reason])
                        if time.time() - last_heartbeat >= HEARTBEAT_SECS:
                            print_progress(total_saved, target_new, t_start, per_cam, quotas,
                                           weak_hist, reasons)
                            last_heartbeat = time.time()
        except KeyboardInterrupt:
            print("Ctrl+C 中断", flush=True)
        finally:
            for r in readers:
                r.stopped = True
            dump_cap_state()
            manifest_f.close()
    print_progress(total_saved, target_new, t_start, per_cam, quotas, weak_hist, reasons)
    print(f"done: 本次 {total_saved} 张 -> {ds_dir}，过程日志 {run_log}", flush=True)


if __name__ == "__main__":
    main()
