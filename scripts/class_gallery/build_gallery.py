"""每类典型样本聚合拼图.

做法(简单可解释):
1. 读 data.yaml 拿类别名, 扫描 train + valid 的 YOLO labels.
2. 对每个类别统计 bbox 面积(w*h)、长宽比(w/h)的中位数, 作为"典型尺寸".
3. 对每个实例打分: 越接近中位数面积越典型; 过小(<0.5%)直接过滤(看不清);
   同图只取该类最大框(保证一张图只贡献一张裁剪, 且聚焦), 兼顾多样性.
4. 每类取 top-N 个实例, 裁剪(带 15% 上下文)后拼成:
   - montage_all_<ds>.jpg : 全类别 x N 裁剪, 一眼看清每类长什么样(主图)
   - overview_<ds>.jpg    : 每类 1 张全图(画出全部框), 看典型场景和相对大小
   - class_<id>_<name>.jpg: 每类单独的 N 张裁剪大图
5. 同时输出 GALLERY.md 简报, 引入上述图片 + 每类数量统计.

用法:
    python scripts/class_gallery/build_gallery.py
    python scripts/class_gallery/build_gallery.py --n-per-class 8 --splits train valid
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
DATASETS = [
    REPO / "dataset" / "batch_12.v9i.yolov11",
    REPO / "dataset" / "for_train_vehicle 3.v1i.yolov11",
]
OUT_DIR = REPO / "output" / "class_gallery"

THUMB_W, THUMB_H = 300, 300
LABEL_W = 260
PAD_RATIO = 0.15
MIN_AREA = 0.005  # 小于 0.5% 的框太小看不清, 直接过滤


def _font(size: int):
    for p in ("C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_TITLE = _font(20)
FONT_BODY = _font(18)
FONT_SMALL = _font(14)
FONT_TINY = _font(13)


def load_names(ds: Path) -> list[str]:
    with open(ds / "data.yaml", encoding="utf-8") as f:
        return list(yaml.safe_load(f)["names"])


def collect(ds: Path, splits: list[str]):
    """返回: instances[class_id] = list(dict(img, cx,cy,w,h,area,aspect,n_box,is_largest))"""
    instances: dict[int, list[dict]] = defaultdict(list)
    img_count: dict[int, set] = defaultdict(set)
    for split in splits:
        img_dir = ds / split / "images"
        lb_dir = ds / split / "labels"
        if not lb_dir.exists():
            continue
        for lb_path in sorted(lb_dir.glob("*.txt")):
            img = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                cand = img_dir / (lb_path.stem + ext)
                if cand.exists():
                    img = cand
                    break
            # roboflow 命名: label stem 与 image stem 可能不完全一致, 兜底按顺序找同名
            if img is None:
                cands = list(img_dir.glob(lb_path.stem + ".*"))
                img = cands[0] if cands else None
            if img is None:
                continue
            boxes = []
            with open(lb_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid, cx, cy, w, h = int(float(parts[0])), *map(float, parts[1:5])
                    if w <= 0 or h <= 0:
                        continue
                    boxes.append((cid, cx, cy, w, h))
            if not boxes:
                continue
            # 同图同类只保留最大框, 避免同一张图刷屏
            best_per_class: dict[int, tuple] = {}
            for b in boxes:
                cid = b[0]
                area = b[3] * b[4]
                if cid not in best_per_class or area > best_per_class[cid][3] * best_per_class[cid][4]:
                    best_per_class[cid] = b
            max_area = max(b[3] * b[4] for b in boxes)
            for cid, (c, cx, cy, w, h) in best_per_class.items():
                area = w * h
                instances[cid].append(
                    dict(
                        img=img,
                        cx=cx, cy=cy, w=w, h=h,
                        area=area, aspect=w / h,
                        n_box=len(boxes),
                        is_largest=abs(area - max_area) < 1e-12,
                    )
                )
                img_count[cid].add(f"{split}/{img.name}")
    return instances, img_count


def pick_typical(items: list[dict], n: int) -> list[dict]:
    """越接近面积中位数越典型; 同分时优先更大(更清晰)、框更少(更聚焦)的图."""
    import statistics

    cands = [it for it in items if it["area"] >= MIN_AREA]
    if not cands:
        cands = list(items)
    med = statistics.median(it["area"] for it in cands)
    scored = []
    for it in cands:
        import math

        dist = abs(math.log(it["area"] / med)) if med > 0 else 0.0
        # 主分: 接近中位数; 次分: 更大更清晰(is_largest 加分), 框少加分
        score = (dist, -it["area"], it["n_box"], str(it["img"]))
        scored.append((score, it))
    scored.sort(key=lambda x: x[0])
    # 去重: 同一图片只出现一次(collect 已保证同类单图单框, 跨 split 仍可能重名, 再保险)
    seen, out = set(), []
    for _, it in scored:
        key = it["img"].name
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= n:
            break
    return out


def crop_with_context(im: Image.Image, cx, cy, w, h) -> Image.Image:
    W, H = im.size
    bw, bh = w * W, h * H
    pad = PAD_RATIO * max(bw, bh)
    x0 = max(0, int(cx * W - bw / 2 - pad))
    y0 = max(0, int(cy * H - bh / 2 - pad))
    x1 = min(W, int(cx * W + bw / 2 + pad))
    y1 = min(H, int(cy * H + bh / 2 + pad))
    crop = im.crop((x0, y0, x1, y1))
    # 等比缩放到 thumb 内(允许放大, 小目标才看得清) + 灰底居中, 保持形状不拉伸
    avail_h = THUMB_H - 26
    if crop.width > 0 and crop.height > 0:
        scale = min(THUMB_W / crop.width, avail_h / crop.height)
        new_w = max(1, int(crop.width * scale))
        new_h = max(1, int(crop.height * scale))
        if (new_w, new_h) != crop.size:
            crop = crop.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (THUMB_W, THUMB_H), (235, 235, 235))
    canvas.paste(crop, ((THUMB_W - crop.width) // 2, (avail_h - crop.height) // 2))
    return canvas


def draw_caption(base: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(base)
    d.rectangle([0, THUMB_H - 26, THUMB_W, THUMB_H], fill=(30, 30, 30))
    d.text((6, THUMB_H - 24), text[:42], fill=(255, 255, 255), font=FONT_TINY)
    return base


def build_montage(ds_name: str, names: list[str], picks: dict[int, list[dict]],
                  img_count, out_path: Path):
    n_cls = len(names)
    n_col = max(len(v) for v in picks.values()) if picks else 1
    header_h = 46
    W = LABEL_W + n_col * THUMB_W
    H = header_h + n_cls * THUMB_H
    board = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(board)
    d.rectangle([0, 0, W, header_h], fill=(24, 33, 72))
    d.text((12, 4), f"{ds_name} : 每类典型裁剪 (每行=1类, 每格=1个典型样本)", fill=(255, 255, 255), font=FONT_BODY)
    d.text((12, 25), "选择标准: 面积接近该类中位数 + 过滤过小框(<0.5%) + 单图单框去重", fill=(200, 210, 255), font=FONT_SMALL)
    for cid, cname in enumerate(names):
        y = header_h + cid * THUMB_H
        # 行标签
        d.rectangle([0, y, LABEL_W, y + THUMB_H], fill=(245, 247, 255))
        d.line([0, y, W, y], fill=(200, 200, 200))
        d.text((10, y + 60), f"[{cid}] {cname}", fill=(0, 0, 0), font=FONT_BODY)
        d.text((10, y + 90), f"samples: {len(picks.get(cid, []))}", fill=(80, 80, 80), font=FONT_SMALL)
        n_img = len(img_count.get(cid, set()))
        n_inst = sum(1 for _ in img_count.get(cid, set()))
        d.text((10, y + 110), f"imgs~{n_img}", fill=(80, 80, 80), font=FONT_SMALL)
        for j, it in enumerate(picks.get(cid, [])):
            x = LABEL_W + j * THUMB_W
            try:
                with Image.open(it["img"]) as im:
                    im = im.convert("RGB")
                    thumb = crop_with_context(im, it["cx"], it["cy"], it["w"], it["h"])
            except Exception:
                thumb = Image.new("RGB", (THUMB_W, THUMB_H), (200, 200, 200))
            thumb = draw_caption(thumb, f"{it['img'].name} a={it['area']:.3f}")
            board.paste(thumb, (x, y))
    board.save(out_path, quality=90)
    return out_path


def build_overview(ds_name: str, names: list[str], picks: dict[int, list[dict]], out_path: Path):
    """每类 1 张全图(画出全部框), 看典型场景."""
    cols = 4
    rows = (len(names) + cols - 1) // cols
    cw, ch = 480, 400
    board = Image.new("RGB", (cols * cw, 46 + rows * ch), (255, 255, 255))
    d = ImageDraw.Draw(board)
    d.rectangle([0, 0, cols * cw, 46], fill=(24, 33, 72))
    d.text((12, 10), f"{ds_name} : 每类 1 张典型全图 (红框=该类焦点, 蓝框=其他目标)", fill=(255, 255, 255), font=FONT_BODY)
    for cid, cname in enumerate(names):
        r, c = divmod(cid, cols)
        x0, y0 = c * cw, 46 + r * ch
        items = picks.get(cid, [])
        cell = Image.new("RGB", (cw, ch), (240, 240, 240))
        if items:
            it = items[0]
            try:
                with Image.open(it["img"]) as im:
                    im = im.convert("RGB")
                    W0, H0 = im.size
                    im.thumbnail((cw, ch - 30), Image.LANCZOS)
                    sx, sy = im.width / W0, im.height / H0
                    cell.paste(im, (0, 0))
                    dd = ImageDraw.Draw(cell)
                    # 读 label 把全图框都画出来
                    lb = Path(str(it["img"]).replace("images", "labels")).with_suffix(".txt")
                    if not lb.exists():  # split 路径差异兜底
                        for cand in Path(it["img"]).parents[2].glob(f"*/labels/{Path(it['img']).stem}.*"):
                            lb = cand.with_suffix(".txt")
                            break
                    if lb.exists():
                        with open(lb, encoding="utf-8") as f:
                            for line in f:
                                p = line.split()
                                if len(p) < 5:
                                    continue
                                cc, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
                                bx0 = (cx - w / 2) * W0 * sx
                                by0 = (cy - h / 2) * H0 * sy
                                bx1 = (cx + w / 2) * W0 * sx
                                by1 = (cy + h / 2) * H0 * sy
                                color = (220, 30, 30) if cc == cid else (60, 120, 220)
                                dd.rectangle([bx0, by0, bx1, by1], outline=color, width=2)
            except Exception as e:
                dd = ImageDraw.Draw(cell)
                dd.text((10, 10), f"read fail: {e}", fill=(0, 0, 0), font=FONT_SMALL)
        dd = ImageDraw.Draw(cell)
        dd.rectangle([0, ch - 30, cw, ch], fill=(30, 30, 30))
        dd.text((6, ch - 26), f"[{cid}] {cname} <- {items[0]['img'].name}"[:58] if items else f"[{cid}] {cname} (no sample)", fill=(255, 255, 255), font=FONT_TINY)
        board.paste(cell, (x0, y0))
    board.save(out_path, quality=90)
    return out_path


def build_per_class(names, picks, out_dir: Path, tag: str):
    paths = {}
    for cid, cname in enumerate(names):
        items = picks.get(cid, [])
        cols = 4
        rows = (len(items) + cols - 1) // cols or 1
        board = Image.new("RGB", (cols * THUMB_W, 40 + rows * THUMB_H), (255, 255, 255))
        d = ImageDraw.Draw(board)
        d.rectangle([0, 0, cols * THUMB_W, 40], fill=(24, 33, 72))
        d.text((12, 8), f"[{cid}] {cname} : {len(items)} 个典型裁剪 ({tag})", fill=(255, 255, 255), font=FONT_BODY)
        for j, it in enumerate(items):
            r, c = divmod(j, cols)
            try:
                with Image.open(it["img"]) as im:
                    im = im.convert("RGB")
                    thumb = crop_with_context(im, it["cx"], it["cy"], it["w"], it["h"])
            except Exception:
                thumb = Image.new("RGB", (THUMB_W, THUMB_H), (200, 200, 200))
            thumb = draw_caption(thumb, f"{it['img'].name} a={it['area']:.3f} r={it['aspect']:.2f}")
            board.paste(thumb, (c * THUMB_W, 40 + r * THUMB_H))
        safe = "".join(ch if ch.isalnum() else "_" for ch in cname)
        p = out_dir / f"class_{cid:02d}_{safe}_{tag}.jpg"
        board.save(p, quality=90)
        paths[cid] = p
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=8)
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_lines = ["# 车辆类别典型图简报", ""]
    for ds in DATASETS:
        tag = ds.name.replace(" ", "_")
        names = load_names(ds)
        instances, img_count = collect(ds, args.splits)
        picks = {cid: pick_typical(instances.get(cid, []), args.n_per_class) for cid in range(len(names))}

        montage = out_dir / f"montage_all_{tag}.jpg"
        overview = out_dir / f"overview_{tag}.jpg"
        build_montage(ds.name, names, picks, img_count, montage)
        build_overview(ds.name, names, picks, overview)
        per_class = build_per_class(names, picks, out_dir, tag)

        md_lines += [f"## {ds.name}", ""]
        md_lines += [f"- 类别数: {len(names)}; 扫描 split: {', '.join(args.splits)}; 每类展示 {args.n_per_class} 个典型裁剪.",
                     "- 典型 = bbox 面积接近该类中位数, 过滤过小框(<0.5% 面积), 同图只取该类最大框(去重).",
                     f"- 类别: {', '.join(f'[{i}] {n}' for i, n in enumerate(names))}", ""]
        md_lines += [f"### 一眼总览(每类 1 张全图, 红框为该类焦点)", "", f"![]({overview.name})", ""]
        md_lines += [f"### 每类聚合(每行=1 类, 每格=1 个典型裁剪)", "", f"![]({montage.name})", ""]
        md_lines += ["### 各类明细", ""]
        md_lines += ["| ID | 类别 | 含该类图片数* | 典型样本来源 |", "|---|---|---|---|"]
        for cid, cname in enumerate(names):
            files = ", ".join(f"`{it['img'].name}`" for it in picks.get(cid, [])[:3])
            md_lines += [f"| {cid} | {cname} | {len(img_count.get(cid, set()))} | {files} |"]
        md_lines += ["", "_*含该类图片数为 train+valid 中出现过该类的去重图片数(按文件名, train/valid 分开计)._", ""]
        md_lines += ["| ID | 类别 | 大图 |", "|---|---|---|"]
        for cid, cname in enumerate(names):
            md_lines += [f"| {cid} | {cname} | ![]({per_class[cid].name}) |"]
        md_lines += ["", "---", ""]
        print(f"[ok] {ds.name}: {montage.name}, {overview.name}, {len(per_class)} per-class")

    md_path = out_dir / "GALLERY.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[ok] markdown: {md_path}")


if __name__ == "__main__":
    main()
