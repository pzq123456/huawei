"""YOLO框 -> X-AnyLabeling v4.0.5 兼容 JSON。

X-AnyLabeling 实测 = LabelMe 超集，必含 imagePath/imageHeight/imageWidth，
shapes 元素: {label, points:[[x1,y1],[x2,y2]], shape_type:"rectangle",
group_id:null, difficult:false, attributes:{}}。
类别只存字符串；数字ID映射靠 classes.txt（见 dataset/rtsp_mine_300_pending/classes.txt）。
"""
import json
from pathlib import Path

VERSION = "2.4.0"  # 与用户示例一致


def to_xany(image_path: str, width: int, height: int, boxes: list) -> dict:
    """boxes: [(label:str, x1,y1,x2,y2 in pixels, conf:float|None), ...]，传 [] 即纯待标注空JSON。"""
    shapes = []
    for label, x1, y1, x2, y2, *rest in boxes:
        shapes.append({
            "label": str(label),
            "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
            "shape_type": "rectangle",
            "group_id": None,
            "difficult": False,
            "attributes": {},
        })
    return {
        "version": VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": Path(image_path).name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def save(json_path: Path, payload: dict):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='重放导出 X-AnyLabeling JSON（不重新拉流）')
    ap.add_argument('--json-mode', choices=['empty', 'prelabel'], default='prelabel')
    ap.add_argument('--conf', type=float, default=0.35)
    args = ap.parse_args()
    print(f'mode={args.json_mode} conf={args.conf}: 占位，正式实现见 collect.py 落盘逻辑')
