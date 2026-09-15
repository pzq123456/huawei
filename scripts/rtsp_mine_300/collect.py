"""双流 RTSP 采集 ~300 张待标注帧。

策略：低阈送检打分（多类别+稀有加权）-> 大时间间隔 + pHash去重 -> 落盘全幅jpg
+ X-AnyLabeling JSON（empty默认空shapes；prelabel写入模型框供人工改）+ manifest.csv。
不写 YOLO txt，避免未审核标签污染训练。

用法：
  python scripts/rtsp_mine_300/collect.py --smoke-test      # 每路抓3帧验连通性
  python scripts/rtsp_mine_300/collect.py --target 300 --json-mode empty
"""
import argparse
import csv
import json
import queue
import random
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    import os
    os.chdir(ROOT)
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def score_dets(labels: list, rare_w: dict, w_cls: float, w_box: float) -> float:
    n_cls = len(set(labels))
    return w_cls * n_cls + w_box * len(labels) + sum(float(rare_w.get(l, 0.0)) for l in labels)


def is_gray_screen(img, std_thresh: float = 12.0, sat_thresh: float = 12.0,
                  chan_diff_thresh: float = 8.0) -> tuple:
    """灰屏/无信号帧过滤：均匀灰（低方差+低饱和+三通道近似相等）判废。
    返回 (is_bad: bool, metric: str)。正常夜景/浓雾方差通常远高于此阈。"""
    import numpy as np
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gstd = float(gray.std())
    if gstd >= std_thresh:
        return False, f'std={gstd:.1f}'
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].mean())
    b, g, r = (img[:, :, i].astype(np.float32) for i in range(3))
    chdiff = float(max(abs(b.mean() - g.mean()), abs(g.mean() - r.mean()), abs(b.mean() - r.mean())))
    bad = sat < sat_thresh and chdiff < chan_diff_thresh
    return bad, f'std={gstd:.1f},sat={sat:.1f},chdiff={chdiff:.1f}'


def phash(img, hash_size: int = 8) -> int:
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (hash_size + 1, hash_size))
    diff = (small[:, 1:] > small[:, :-1])
    return sum(1 << i for i, v in enumerate(diff.flatten()) if v)


def bbox_touches_edge(xyxy, w: int, h: int, seam, margin: float = 2.0) -> bool:
    """检测框是否触碰真实图像边界。seam 为 apply_crop 产生的拼接缝所在侧
    （'right'/'left'），拼缝不是真实边界，对应侧不计入贴边判定。"""
    x1, y1, x2, y2 = xyxy
    left, top, right, bottom = x1 <= margin, y1 <= margin, x2 >= w - margin, y2 >= h - margin
    if seam == 'right':
        right = False
    elif seam == 'left':
        left = False
    return left or top or right or bottom


def roi_sharpness(img, xyxy) -> float:
    """框内区域 Laplacian 方差，作为清晰度/运动模糊的廉价代理。"""
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    roi = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


class StreamReader(threading.Thread):
    """单流 PyAV 读取线程（复用 scripts/collect_data.py 的 EventAVStreamer 思想）。"""

    def __init__(self, url: str, interval: float):
        super().__init__(daemon=True)
        self.url = url
        self.interval = interval
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.stopped = False

    def run(self):
        import av
        last_push = 0.0
        container = None
        while not self.stopped:
            try:
                if container is None:
                    container = av.open(self.url, options={"rtsp_transport": "tcp", "stimeout": "8000000"})
                for frame in container.decode(video=0):
                    if self.stopped:
                        return
                    now = time.time()
                    if now - last_push < self.interval:
                        continue
                    img = frame.to_ndarray(format="bgr24")
                    last_push = now
                    if self.q.full():
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                    self.q.put(img)
            except Exception as e:  # noqa: BLE001 - 拉流需容错重连
                print(f"[解码中断 {self.url}] {e}，重连...")
                container = None
                time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='scripts/rtsp_mine_300/config.yaml')
    ap.add_argument('--target', type=int, default=None)
    ap.add_argument('--json-mode', choices=['empty', 'prelabel'], default=None)
    ap.add_argument('--out-dir', default=None, help='覆盖 config 的 dataset_dir（分批采集用）')
    ap.add_argument('--resume', action='store_true', help='断点续采：不清空已有图片/manifest，只补采 --target 张')
    ap.add_argument('--smoke-test', action='store_true')
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    target = args.target or cfg['sampling']['target_total']
    json_mode = args.json_mode or cfg['output']['json_mode']
    names = cfg['names_v9'] if cfg.get('classes', 'v9') == 'v9' else cfg['names_merge8']

    from ultralytics import YOLO
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from export_xany import to_xany, save as save_json

    weights = Path(cfg['model']['weights'])
    if not weights.is_file():
        weights = Path(cfg['model']['fallback'])
        print(f'[warn] 主权重缺失，回退到 {weights}')
    model = YOLO(str(weights), task='detect')
    print(f'模型 {weights} 类别={model.names}')

    ds_dir = ROOT / (args.out_dir or cfg['output']['dataset_dir'])
    img_dir = ds_dir / 'images'
    js_dir = ds_dir / 'annotations_xany'
    img_dir.mkdir(parents=True, exist_ok=True)
    js_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / 'classes.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')

    run_log = ROOT / cfg['output']['run_log_dir'] / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv"
    run_log.parent.mkdir(parents=True, exist_ok=True)

    readers = [StreamReader(s['url'], cfg['sampling']['detect_interval']) for s in cfg['streams']]
    for r in readers:
        r.start()

    model_names = {int(k): v for k, v in model.names.items()} if isinstance(model.names, dict) else \
        {i: v for i, v in enumerate(model.names)}
    if len(model_names) != len(names):
        print(f'[warn] 模型类别数 {len(model_names)} != classes.txt {len(names)}，'
              f'prelabel 标签将按模型名输出：{list(model_names.values())}')

    def apply_crop(img, mode):
        w = img.shape[1]
        if mode in ('left', 'auto-left') and w >= 2600:
            return img[:, : w // 2], 'right'
        if mode == 'right' and w >= 2600:
            return img[:, w // 2:], 'left'
        return img, None

    saved, last_save, last_hash = 0, {s['name']: 0.0 for s in cfg['streams']}, {}
    car_only = 0
    bal = cfg.get('balance', {})
    weak_classes = set(bal.get('weak_classes', []))
    strong_caps = {k: int(v) for k, v in bal.get('strong_frame_caps', {}).items()}
    drop_single = bool(bal.get('drop_single_private_car', False))
    cls_frame_cnt = Counter()

    # 分类阈值化：conf_low_classes 低阈出框交给人工，其余维持基线，overrides 最高优先
    oscore = float(cfg['model'].get('conf_scoring', 0.30))
    conf_low = float(cfg['model'].get('conf_low', oscore))
    low_classes = set(cfg['model'].get('conf_low_classes', []))
    conf_ov = {k: float(v) for k, v in (cfg['model'].get('conf_overrides') or {}).items()}
    per_class_conf = {c: (conf_low if c in low_classes else oscore) for c in names}
    per_class_conf.update(conf_ov)
    conf_floor = min(per_class_conf.values())   # 预测用最低阈，之后按类别二次筛选
    # 清洗（清晰截取）与无条件采样
    clean_cfg = cfg.get('clean_capture', {}) or {}
    clean_on = bool(clean_cfg.get('enabled', False))
    edge_margin = float(clean_cfg.get('edge_margin', 2.0))
    min_area_frac = float(clean_cfg.get('min_box_area_frac', 0.0))
    lap_min = float(clean_cfg.get('laplacian_var_min', 0.0))
    uncond_interval = float(cfg['sampling'].get('unconditional_interval', 0.0))
    uncond_max = int(target * float(cfg['sampling'].get('unconditional_max_ratio', 0.0)))
    last_uncond = {s['name']: 0.0 for s in cfg['streams']}
    uncond_saved = 0
    soft_cap = int(bal.get('soft_cam_class_cap', 0) or 0)
    soft_drop = float(bal.get('soft_cap_drop_prob', 0.8))
    cls_cam_cnt = Counter()
    # 软上限计数持久化：跨 --resume/重启保留，避免长期挂机下软上限形同虚设
    cap_state_path = ds_dir / 'soft_cap_state.json'
    if args.resume and cap_state_path.is_file():
        try:
            raw_state = json.loads(cap_state_path.read_text(encoding='utf-8'))
            cls_cam_cnt.update({tuple(k.split('\t', 1)): int(v) for k, v in raw_state.items()})
            print(f"soft-cap state 载入 {len(raw_state)} 项 <- {cap_state_path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f'[warn] soft-cap state 读取失败，忽略: {e}')

    def dump_cap_state():
        try:
            cap_state_path.write_text(
                json.dumps({f"{c}\t{k}": v for (c, k), v in cls_cam_cnt.items()}, ensure_ascii=False),
                encoding='utf-8')
        except Exception as e:  # noqa: BLE001
            print(f'[warn] soft-cap state 写入失败: {e}')
    t_start = time.time()
    smoke_deadline = 150.0  # smoke-test 最多等待150s，避免流不通时无限挂起
    existing = len(list(img_dir.glob('*.jpg'))) if args.resume else 0
    manifest_path = ds_dir / 'manifest.csv'
    manifest_exists = manifest_path.exists() and manifest_path.stat().st_size > 0
    manifest_f = open(manifest_path, 'a' if (args.resume and manifest_exists) else 'w',
                      newline='', encoding='utf-8')
    manifest = csv.writer(manifest_f)
    if not (args.resume and manifest_exists):
        manifest.writerow(['file', 'cam', 'timestamp', 'score', 'n_box', 'n_cls', 'rare_hit', 'weak_hit', 'json_mode', 'uncond', 'n_filtered'])
    print(f"resume={args.resume} existing={existing} 本次补采目标={target}", flush=True)
    with open(run_log, 'w', newline='', encoding='utf-8') as log_f:
        log = csv.writer(log_f)
        log.writerow(['file', 'cam', 'score', 'n_box', 'n_cls', 'rare_hit', 'saved', 'reason'])
        try:
            import torch
            with torch.no_grad():
                while saved < target:
                    for s, r in zip(cfg['streams'], readers):
                        try:
                            frame = r.q.get(timeout=1.0)
                        except queue.Empty:
                            if args.smoke_test and time.time() - t_start > smoke_deadline:
                                raise SystemExit(
                                    f"smoke-test 超时({smoke_deadline}s)仅收到 {saved} 帧，检查RTSP连通性")
                            continue
                        frame, seam = apply_crop(frame, s.get('crop'))
                        H, W = frame.shape[:2]
                        res = model.predict(frame, conf=conf_floor,
                                            imgsz=cfg['model']['imgsz'], verbose=False,
                                            device=cfg['model']['device'])[0]
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
                        sc = score_dets(labels, cfg['scoring']['rare_w'], cfg['scoring']['w_cls'], cfg['scoring']['w_box'])
                        rare_hit = next((l for l in labels if cfg['scoring']['rare_w'].get(l, 0) >= 2.0), '')
                        now = time.time()
                        h = phash(frame)
                        reason = 'ok'
                        gf = cfg['sampling'].get('gray_filter', {})
                        bad, metric = is_gray_screen(
                            frame, std_thresh=float(gf.get('std_thresh', 12.0)),
                            sat_thresh=float(gf.get('sat_thresh', 12.0)),
                            chan_diff_thresh=float(gf.get('chan_diff_thresh', 8.0)))
                        if args.smoke_test:
                            print(f"[smoke] {s['name']} {frame.shape} gray=({metric}) dets={labels} score={sc:.1f}")
                            saved += 1
                            if saved >= 3 * len(readers):
                                print('smoke ok'); return
                            continue
                        uncond_due = (uncond_interval > 0 and uncond_saved < uncond_max
                                      and now - last_uncond[s['name']] >= uncond_interval)
                        if bad:
                            reason = 'gray-screen'
                        elif uncond_due:
                            reason = 'uncond'
                        elif not labels:
                            reason = 'no-det' if not raw else 'unclean'
                        elif now - last_save[s['name']] < cfg['sampling']['min_save_interval']:
                            reason = 'interval'
                        elif any(bin(h ^ ph).count('1') < cfg['sampling']['phash_thresh'] for ph in last_hash.values()):
                            reason = 'dup-phash'
                        elif drop_single and len(labels) == 1 and labels[0] == 'Private Car':
                            reason = 'single-private-car'
                        elif bal.get('require_weak') and not (set(labels) & weak_classes):
                            reason = 'no-weak'
                        elif not (set(labels) & weak_classes) and any(
                                cls_frame_cnt[c] >= strong_caps.get(c, 10 ** 9) for c in set(labels)
                                if c in strong_caps):
                            reason = 'strong-cap'
                        elif soft_cap > 0 and any(cls_cam_cnt[(s['name'], c)] >= soft_cap
                                                  for c in set(labels)) and random.random() < soft_drop:
                            reason = 'soft-cap'
                        if reason in ('ok', 'uncond'):
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                            fname = f"{ts}_{s['name']}.jpg"
                            cv2.imwrite(str(img_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, cfg['output']['jpeg_quality']])
                            dets = [(l, xy[0], xy[1], xy[2], xy[3], cf) for l, cf, xy in kept] \
                                if json_mode == 'prelabel' else []
                            save_json(js_dir / (Path(fname).stem + '.json'), to_xany(fname, W, H, dets))
                            saved += 1
                            last_save[s['name']] = now
                            last_hash[s['name']] = h
                            if reason == 'uncond':
                                last_uncond[s['name']] = now
                                uncond_saved += 1
                            for c in set(labels):
                                cls_frame_cnt[c] += 1
                                cls_cam_cnt[(s['name'], c)] += 1
                            dump_cap_state()
                            if set(labels) == {'Private Car'}:
                                car_only += 1
                            print(f"[{existing + saved}/{existing + target}] {fname} score={sc:.1f} "
                                  f"reason={reason} n={len(labels)} filt={n_filtered} {labels}", flush=True)
                            manifest.writerow([fname, s['name'], ts, f"{sc:.2f}",
                                               len(labels), len(set(labels)), rare_hit,
                                               int(bool(set(labels) & weak_classes)), json_mode,
                                               int(reason == 'uncond'), n_filtered])
                            manifest_f.flush()
                        log.writerow([f"{s['name']}_{int(now)}", s['name'], f"{sc:.2f}",
                                      len(labels), len(set(labels)), rare_hit, int(reason in ('ok', 'uncond')), reason])
        except KeyboardInterrupt:
            pass
        finally:
            for r in readers:
                r.stopped = True
            dump_cap_state()
            manifest_f.close()
    print(f'done: 本次 {saved} 张，累计 {existing + saved} 张 -> {ds_dir}，过程日志 {run_log}')


if __name__ == '__main__':
    main()
