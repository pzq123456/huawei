# scripts/analyze_distortion.py
# 目标:用可复现指标回答"哪组畸变重、哪组接近部署相机",避免拍脑袋删数据。
# 方法(无棋盘格下的代理指标,非真 k1):
#   curve_v1: 全图长轮廓拟直线后的 最大偏离/长度 (越大越弯,含车辆曲线干扰)
#   curve_v2: 仅边缘+横平竖直轮廓 (车道线/灯杆/护栏,干扰小,更贴近桶形畸变)
#   edge_box: 标注框中心落在外圈20%边带的比例 (畸变伤害集中在边缘)
#   box_area: 标注框面积中位数 (视角/距离代理)
# 对照组: dataset/car_collected (部署相机 dahua1001619,无标注,只算图像指标)
import cv2
import pathlib
import random
import re
import csv
import numpy as np
from collections import defaultdict

TRAIN_IMG = pathlib.Path("dataset/batch_12.v9i.yolov11/train/images")
TRAIN_LBL = pathlib.Path("dataset/batch_12.v9i.yolov11/train/labels")
DEPLOY = pathlib.Path("dataset/car_collected")
OUT_CSV = pathlib.Path("tmp/distortion_by_group.csv")
N_SAMPLE = 30
SEED = 0


def group_of(name: str) -> str:
    if name.startswith("2025-04-28"): return "2025-04-28"
    if name.startswith("ANMR"): return "ANMR"
    if name.startswith("frame_20250604"): return "frame_20250604"
    if name.startswith("frame"): return "frame"
    if name.startswith("23103HK"): return "23103HK"
    if name.startswith("Motorcycle"): return "Motorcycle"
    if re.match(r"^240\d*HK", name): return "240xxHK"
    if re.match(r"^230\d*HK", name): return "230xxHK"
    if re.match(r"^\d", name): return "web_numeric"
    if name.startswith("qingyiI"): return "qingyiI"
    if name.startswith("shang"): return "shanghuan"
    c = name.split("_")[0]
    return c if len(c) < 20 else "other"


def long_contour_curves(gray, border_hv_only: bool):
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    vals = []
    for c in cnts:
        L = cv2.arcLength(c, False)
        if L < (0.30 if border_hv_only else 0.25) * max(h, w):
            continue
        m = c.reshape(-1, 2).astype(np.float32)
        if border_hv_only:
            x0, x1 = m[:, 0].min(), m[:, 0].max()
            y0, y1 = m[:, 1].min(), m[:, 1].max()
            if not (x0 < 0.2 * w or x1 > 0.8 * w or y0 < 0.2 * h or y1 > 0.8 * h):
                continue
        vx, vy, xc, yc = cv2.fitLine(m, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        if border_hv_only:
            ang = min(a := abs(np.degrees(np.arctan2(vy, vx))), 180 - a)
            if not (ang < 12 or abs(ang - 90) < 12):
                continue
        d = np.abs((m[:, 0] - xc) * (-vy) + (m[:, 1] - yc) * vx)
        vals.append(d.max() / (L + 1e-6))
    if not vals:
        return 0.0
    return float(np.mean(sorted(vals, reverse=True)[:5 if not border_hv_only else 3]))


def label_stats(lbl_path):
    """返回 (boxes_per_img, edge_frac, median_area). 无标注文件返回 (0, nan, nan)."""
    if not lbl_path.is_file():
        return 0.0, float("nan"), float("nan")
    areas, edge = [], 0
    n = 0
    for line in lbl_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        _, xc, yc, bw, bh = map(float, p[:5])
        n += 1
        areas.append(bw * bh)
        if xc < 0.2 or xc > 0.8 or yc < 0.2 or yc > 0.8:
            edge += 1
    if n == 0:
        return 0.0, float("nan"), float("nan")
    return float(n), edge / n, float(np.median(areas))


def main():
    random.seed(SEED)
    groups = defaultdict(list)
    for p in TRAIN_IMG.glob("*.jpg"):
        groups[group_of(p.name)].append(p)
    deploy_files = sorted(DEPLOY.glob("*.jpg"))

    rows = []
    for g in sorted(groups, key=lambda k: -len(groups[k])):
        files = sorted(groups[g])
        samp = random.sample(files, min(N_SAMPLE, len(files)))
        c1, c2, nb, ef, ar = [], [], [], [], []
        for f in samp:
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            c1.append(long_contour_curves(gray, False))
            c2.append(long_contour_curves(gray, True))
            stem = f.stem
            n, e, a = label_stats(TRAIN_LBL / (stem + ".txt"))
            nb.append(n)
            if not np.isnan(e):
                ef.append(e)
            if not np.isnan(a):
                ar.append(a)
        rows.append(dict(group=g, n_total=len(files), n_test=len(samp),
                         curve_v1=np.mean(c1), curve_v2=np.mean(c2),
                         boxes_per_img=np.mean(nb),
                         edge_box=np.mean(ef) if ef else float("nan"),
                         box_area=np.median(ar) if ar else float("nan")))

    # 部署对照(无标注)
    samp = random.sample(deploy_files, min(N_SAMPLE, len(deploy_files)))
    d1, d2 = [], []
    for f in samp:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        d1.append(long_contour_curves(gray, False))
        d2.append(long_contour_curves(gray, True))
    rows.append(dict(group="DEPLOY_car_collected", n_total=len(deploy_files), n_test=len(samp),
                     curve_v1=np.mean(d1), curve_v2=np.mean(d2),
                     boxes_per_img=float("nan"), edge_box=float("nan"), box_area=float("nan")))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'group':22s} {'total':>6s} {'cv1':>7s} {'cv2':>7s} {'box/img':>7s} {'edge%':>6s} {'area_med':>8s}")
    for r in sorted(rows, key=lambda r: (r["group"].startswith("DEPLOY"), -r["curve_v1"])):
        print(f"{r['group']:22s} {r['n_total']:6d} {r['curve_v1']:7.4f} {r['curve_v2']:7.4f} "
              f"{r['boxes_per_img']:7.2f} {r['edge_box'] * 100 if not np.isnan(r['edge_box']) else float('nan'):6.1f} "
              f"{r['box_area']:8.4f}")
    print(f"\nCSV -> {OUT_CSV}")


if __name__ == "__main__":
    main()
