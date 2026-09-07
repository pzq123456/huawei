# scripts/analyze_van_size.py
# 专家实验: Van 按 bbox 面积分桶,统计每桶 Van召回 / Van->Car率 / 漏检率。
# 用现有 merge8 best.pt 在同域 valid 上推理,不需新标注。
# 判读: 小桶混淆高、大桶混淆低 => 视觉信息不足(尺度主因);
#       大桶也大量->Car => 类边界+长尾主因。
import pathlib
import numpy as np
from ultralytics import YOLO

WEIGHTS = "runs/detect/yolo26m_merge8_20260902_0916/weights/best.pt"
DATA = pathlib.Path("dataset/batch_12.v10i.merge8.yolov11/valid")
IMGSZ = 640
CONF = 0.05  # 低阈值全量记录,避免阈值本身掩盖混淆
IOU = 0.5

BUCKETS = [(0.0, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 2.0)]
NAMES = {0: "Bus", 1: "Franchised Bus", 2: "Truck", 3: "Motorcycle",
         4: "PLB GMB", 5: "Private Car", 6: "Taxi", 7: "Van"}


def load_gt(p):
    boxes = []
    if p.is_file():
        for line in p.read_text().splitlines():
            s = line.split()
            if len(s) >= 5:
                c, x, y, w, h = int(float(s[0])), *map(float, s[1:5])
                boxes.append((c, x, y, w, h))
    return boxes


def xywhn_to_xyxy(b):
    _, x, y, w, h = b
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)


def main():
    model = YOLO(WEIGHTS)
    imgs = sorted((DATA / "images").glob("*.jpg"))
    stats = {b: {"n": 0, "van_ok": 0, "to_car": 0, "to_other": 0, "miss": 0,
                 "conf_ok": [], "conf_car": []} for b in BUCKETS}
    for img in imgs:
        gt = load_gt(DATA / "labels" / (img.stem + ".txt"))
        res = model.predict(str(img), imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        preds = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy() / np.array(
                [res.orig_shape[1], res.orig_shape[0]] * 2)  # 归一化
            for bb, cl, cf in zip(xyxy, res.boxes.cls.cpu().numpy(),
                                  res.boxes.conf.cpu().numpy()):
                preds.append((bb, int(cl), float(cf)))
        used = set()
        for g in gt:
            if g[0] != 7:  # 只看 Van GT
                continue
            area = g[3] * g[4]
            bkt = next(b for b in BUCKETS if b[0] <= area < b[1])
            s = stats[bkt]
            s["n"] += 1
            gb = xywhn_to_xyxy(g)
            best, bi = 0.0, -1
            for i, (pb, _, _) in enumerate(preds):
                if i in used:
                    continue
                v = iou(gb, pb)
                if v > best:
                    best, bi = v, i
            if best >= IOU:
                used.add(bi)
                pc, cf = preds[bi][1], preds[bi][2]
                if pc == 7:
                    s["van_ok"] += 1
                    s["conf_ok"].append(cf)
                elif pc == 5:
                    s["to_car"] += 1
                    s["conf_car"].append(cf)
                else:
                    s["to_other"] += 1
            else:
                s["miss"] += 1
    print(f"{'area桶':>14s} {'n':>5s} {'Van召回':>7s} {'Van->Car':>8s} {'->其他':>7s} {'漏检':>6s} {'Van置信中位':>10s} {'判Car置信':>9s}")
    for b in BUCKETS:
        s = stats[b]
        n = s["n"] or 1
        print(f"{b[0]:.2f}~{b[1]:.2f} {s['n']:5d} {s['van_ok']/n:7.2%} {s['to_car']/n:8.2%} "
              f"{s['to_other']/n:7.2%} {s['miss']/n:6.2%} "
              f"{(np.median(s['conf_ok']) if s['conf_ok'] else float('nan')):10.3f} "
              f"{(np.median(s['conf_car']) if s['conf_car'] else float('nan')):9.3f}")


if __name__ == "__main__":
    main()
