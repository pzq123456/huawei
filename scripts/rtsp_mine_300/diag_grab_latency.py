"""诊断：RTSP 采集“检测 -> 落盘”时间差 / 帧陈旧度 / 潜在半截车与空抓。

结论先行（基于 collect.py 现有实现）：
  - StreamReader 每 interval 秒把“最新帧”放入 maxsize=1 队列（中间帧被替换丢弃）。
  - 主循环 get 到该帧后，在该帧上推理，并把**同一帧对象**落盘。
  -> 因此“模型看到的帧”与“落盘的帧”是同一帧，框与图天然对齐，不存在“检测后另抓新帧导致错位”。
  真正存在的问题是**陈旧度(staleness)**与**吞吐/调度延迟**：落盘帧可能是 interval(~2s) 前解码的，
  且多路串行 + 阻塞 get 会放大延迟。若某天改成“先检测、落盘时现取一帧”，则会在延迟内产生
  半截车/空抓——本脚本用合成场景量化这个错位，用真实流量化现有管线的时间差。

用法：
  # 1) 合成扫参：不同车速/延迟下的错位与半截概率（纯离线，秒出）
  python scripts/rtsp_mine_300/diag_grab_latency.py --mode synthetic

  # 2) 真实流诊断：量化 落盘帧陈旧度/推理耗时/队列丢帧/与“当前最新帧”的差异
  python scripts/rtsp_mine_300/diag_grab_latency.py --mode live \
      --url rtsp://118.140.234.166:8554/dahua1001637 --duration 90
输出：output/rtsp_mine_300/diag/ 下的 report_*.csv / compare_*.jpg / summary_*.txt
"""
import argparse
import csv
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/rtsp_mine_300/diag"


# ---------------------------------------------------------------------------
# 合成扫参：量化“检测后另取新帧落盘”的错位
# ---------------------------------------------------------------------------
def synthetic_sweep(speeds=(5, 10, 20, 40, 80), latencies_ms=(0, 50, 100, 200, 500, 1000, 2000),
                    width=1920, car_len=400, trials=2000, seed=0):
    """车以 v px/s 移动；检测时车尾刚进入/完全在画面内。落盘若延后 L 秒取新帧，
    车位置平移 v*L，可能从“完整”变“半截/驶出”。统计半截/驶出概率。"""
    rng = np.random.default_rng(seed)
    rows = []
    for v in speeds:
        for L in latencies_ms:
            Ls = L / 1000.0
            shift = v * Ls
            partial = out = 0
            for _ in range(trials):
                # 检测时刻车头位置 x_head ∈ [car_len, width]（车身完整在画面内）
                x_head = rng.uniform(car_len, width)
                x_head2 = x_head - shift          # 落盘新帧里的位置（向左驶/或方向任意，取最坏）
                x_head2 = min(width, max(-car_len, x_head2))
                if x_head2 < 0:                    # 车头已驶出左边界
                    out += 1
                elif x_head2 < car_len:            # 车身被截断
                    partial += 1
            rows.append(dict(speed_px_s=v, latency_ms=L, shift_px=round(shift, 1),
                             p_partial=partial / trials, p_out=out / trials,
                             p_bad=(partial + out) / trials))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "synthetic_sweep.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("== 合成扫参：落盘延后 L 秒取新帧时的错位概率（1920宽, 车长400） ==")
    print(f"{'v(px/s)':>8} {'L(ms)':>6} {'shift(px)':>9} {'半截':>7} {'驶出':>7} {'异常':>7}")
    for r in rows:
        if r["latency_ms"] in (0, 100, 500, 1000, 2000):
            print(f"{r['speed_px_s']:>8} {r['latency_ms']:>6} {r['shift_px']:>9} "
                  f"{r['p_partial']:>7.3f} {r['p_out']:>7.3f} {r['p_bad']:>7.3f}")
    print(f"\nsweep -> {p}")
    return rows


# ---------------------------------------------------------------------------
# 真实流：带时间戳的诊断读取器（不改动 collect.py）
# ---------------------------------------------------------------------------
class DiagReader(threading.Thread):
    def __init__(self, url, interval):
        super().__init__(daemon=True)
        self.url, self.interval = url, interval
        self.q = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest = None
        self.latest_ts = 0.0
        self.decoded = 0
        self.pushed = 0
        self.dropped = 0
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
                    img = frame.to_ndarray(format="bgr24")
                    with self.lock:
                        self.latest = img          # 每个解码帧都更新（未节流），用于对比“当前最新”
                        self.latest_ts = now
                        self.decoded += 1
                    if now - last_push < self.interval:
                        continue
                    last_push = now
                    if self.q.full():
                        try:
                            self.q.get_nowait()
                            self.dropped += 1
                        except queue.Empty:
                            pass
                    self.q.put((img, now))         # 带上解码时间戳
                    self.pushed += 1
            except Exception as e:  # noqa: BLE001
                print(f"[解码中断 {self.url}] {e}，重连...", flush=True)
                container = None
                time.sleep(1.0)


def run_live(args):
    from ultralytics import YOLO
    import torch

    urls = args.urls if args.urls else [args.url]
    model = YOLO(str(ROOT / args.weights), task="detect")
    names = model.names
    readers = [DiagReader(u, args.interval) for u in urls]
    for r in readers:
        r.start()

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    rep = OUT / f"report_{stamp}.csv"
    cmp_dir = OUT / f"compare_{stamp}"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    rows, ages, infers, saves_ts = [], [], [], []
    last_get = {u: None for u in urls}
    loop_periods = []
    t0 = time.time()
    n_saved = 0
    print(f"诊断 {args.duration}s：{len(urls)} 路 interval={args.interval}s imgsz={args.imgsz}", flush=True)
    with torch.no_grad(), open(rep, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "t", "url", "age_decode_to_save_s", "infer_ms",
                    "fresh_gap_s", "mean_abs_diff", "n_box", "boxes"])
        while time.time() - t0 < args.duration:
            for u, r in zip(urls, readers):
                try:
                    img, t_dec = r.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                t_get = time.time()
                res = model.predict(img, conf=0.30, imgsz=args.imgsz, verbose=False, device=args.device)[0]
                t_inf = time.time()
                labels = [names[int(c)] for c in res.boxes.cls.tolist()] if res.boxes is not None and len(res.boxes) else []
                with r.lock:
                    fresh = None if r.latest is None else r.latest.copy()
                    fresh_ts = r.latest_ts
                fresh_gap = t_inf - fresh_ts if fresh is not None else -1
                diff = float(np.abs(img.astype(np.int16) - fresh.astype(np.int16)).mean()) if fresh is not None else -1
                age = t_inf - t_dec
                ages.append(age)
                infers.append((t_inf - t_get) * 1000)
                if last_get[u] is not None:
                    loop_periods.append(t_get - last_get[u])
                last_get[u] = t_get
                w.writerow([n_saved, round(t_get - t0, 2), u.split("/")[-1], round(age, 4),
                            round((t_inf - t_get) * 1000, 1), round(fresh_gap, 4), round(diff, 2),
                            len(labels), ";".join(labels)])
                f.flush()
                rows.append([round(age, 4), round((t_inf - t_get) * 1000, 1), round(diff, 2)])
                if n_saved < args.save_compare and fresh is not None and u == urls[0]:
                    a = img.copy()
                    for b in (res.boxes.xyxy.tolist() if labels else []):
                        cv2.rectangle(a, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                    stack = cv2.vconcat([cv2.resize(a, (960, 540)), cv2.resize(fresh, (960, 540))])
                    cv2.putText(stack, f"TOP=detected frame  age={age*1000:.0f}ms",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(stack, f"BOTTOM=current frame at save (gap={fresh_gap*1000:.0f}ms, diff={diff:.1f})",
                                (10, 540 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imwrite(str(cmp_dir / f"cmp_{n_saved:03d}.jpg"), stack, [cv2.IMWRITE_JPEG_QUALITY, 88])
                n_saved += 1
            if n_saved and n_saved % 40 < len(urls):
                print(f"  {n_saved} samples: age_med={np.median(ages)*1000:.0f}ms "
                      f"infer_med={np.median(infers):.0f}ms diff_med={np.median([r[2] for r in rows]):.2f}", flush=True)

    for r in readers:
        r.stopped = True

    def stat(x):
        x = np.asarray(x, dtype=float)
        return (f"min={x.min():.3f} med={np.median(x):.3f} p90={np.percentile(x,90):.3f} max={x.max():.3f}"
                if len(x) else "n/a")

    summary = [
        f"urls={len(urls)} duration={args.duration}s interval={args.interval}s",
        f"samples={len(rows)} total_decoded={sum(r.decoded for r in readers)} "
        f"total_pushed={sum(r.pushed for r in readers)} dropped_in_reader={sum(r.dropped for r in readers)}",
        f"age_decode_to_save_s: {stat([r[0] for r in rows])}",
        f"infer_ms:              {stat([r[1] for r in rows])}",
        f"loop_period_s:         {stat(loop_periods)}",
        f"mean_abs_diff(detected vs current): {stat([r[2] for r in rows])}",
    ]
    sp = OUT / f"summary_{stamp}.txt"
    sp.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nreport -> {rep}\ncompare frames -> {cmp_dir}\nsummary -> {sp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "live"], default="synthetic")
    ap.add_argument("--url", default="rtsp://118.140.234.166:8554/dahua1001637")
    ap.add_argument("--urls", nargs="*", default=None,
                    help="多路诊断（给出则覆盖 --url）")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=90)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--weights", default="runs/detect/yolo26m_merge8_20260903_0950/weights/best.pt")
    ap.add_argument("--save-compare", type=int, default=12)
    args = ap.parse_args()

    synthetic_sweep()
    if args.mode == "live":
        run_live(args)


if __name__ == "__main__":
    main()
