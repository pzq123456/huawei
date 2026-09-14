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
import queue
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
        if mode in ('left', 'auto-left'):
            return img[:, : w // 2] if w >= 2600 else img
        if mode == 'right':
            return img[:, w // 2:] if w >= 2600 else img
        return img

    saved, last_save, last_hash = 0, {s['name']: 0.0 for s in cfg['streams']}, {}
    car_only = 0
    bal = cfg.get('balance', {})
    weak_classes = set(bal.get('weak_classes', []))
    strong_caps = {k: int(v) for k, v in bal.get('strong_frame_caps', {}).items()}
    drop_single = bool(bal.get('drop_single_private_car', False))
    cls_frame_cnt = Counter()
    t_start = time.time()
    smoke_deadline = 150.0  # smoke-test 最多等待150s，避免流不通时无限挂起
    manifest_f = open(ds_dir / 'manifest.csv', 'w', newline='', encoding='utf-8')
    manifest = csv.writer(manifest_f)
    manifest.writerow(['file', 'cam', 'timestamp', 'score', 'n_box', 'n_cls', 'rare_hit', 'weak_hit', 'json_mode'])
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
                        frame = apply_crop(frame, s.get('crop'))
                        res = model.predict(frame, conf=cfg['model']['conf_scoring'],
                                            imgsz=cfg['model']['imgsz'], verbose=False,
                                            device=cfg['model']['device'])[0]
                        labels = [model_names[int(c)] for c in res.boxes.cls.tolist()] \
                            if res.boxes is not None and len(res.boxes) else []
                        boxes = res.boxes.xyxy.tolist() if labels else []
                        confs = res.boxes.conf.tolist() if labels else []
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
                        if bad:
                            reason = 'gray-screen'
                        elif not labels:
                            reason = 'no-det'
                        elif now - last_save[s['name']] < cfg['sampling']['min_save_interval']:
                            reason = 'interval'
                        elif any(bin(h ^ ph).count('1') < cfg['sampling']['phash_thresh'] for ph in last_hash.values()):
                            reason = 'dup-phash'
                        elif drop_single and len(labels) == 1 and labels[0] == 'Private Car':
                            reason = 'single-private-car'
                        elif not (set(labels) & weak_classes) and any(
                                cls_frame_cnt[c] >= strong_caps.get(c, 10 ** 9) for c in set(labels)
                                if c in strong_caps):
                            reason = 'strong-cap'
                        if reason == 'ok':
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                            fname = f"{ts}_{s['name']}.jpg"
                            cv2.imwrite(str(img_dir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, cfg['output']['jpeg_quality']])
                            dets = list(zip(labels, [b[0] for b in boxes], [b[1] for b in boxes],
                                            [b[2] for b in boxes], [b[3] for b in boxes], confs)) if json_mode == 'prelabel' else []
                            H, W = frame.shape[:2]
                            save_json(js_dir / (Path(fname).stem + '.json'), to_xany(fname, W, H, dets))
                            saved += 1
                            last_save[s['name']] = now
                            last_hash[s['name']] = h
                            for c in set(labels):
                                cls_frame_cnt[c] += 1
                            if set(labels) == {'Private Car'}:
                                car_only += 1
                            print(f"[{saved}/{target}] {fname} score={sc:.1f} {labels}", flush=True)
                            manifest.writerow([fname, s['name'], ts, f"{sc:.2f}",
                                               len(labels), len(set(labels)), rare_hit,
                                               int(bool(set(labels) & weak_classes)), json_mode])
                            manifest_f.flush()
                        log.writerow([f"{s['name']}_{int(now)}", s['name'], f"{sc:.2f}",
                                      len(labels), len(set(labels)), rare_hit, int(reason == 'ok'), reason])
        except KeyboardInterrupt:
            pass
        finally:
            for r in readers:
                r.stopped = True
            manifest_f.close()
    print(f'done: {saved} 张 -> {ds_dir}，过程日志 {run_log}')


if __name__ == '__main__':
    main()
