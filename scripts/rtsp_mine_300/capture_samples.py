"""从候选 RTSP 各抓 1 帧，留存图片供确认机位。"""
import argparse
import time
from pathlib import Path

import av
import cv2

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STREAMS = [
    "dahua1001601", "dahua1001617", "dahua1001619", "dahua1001620", "dahua1001625",
    "dahua1001630", "dahua1001633", "dahua1001636", "dahua1001637", "dahua1001638",
    "dahua1001640", "dahua1001641", "dahua1001642", "dahua1001645", "dahua1001646",
    "dahua1001647", "dahua1001648", "dahua1001649",
]


def grab(url: str, crop: str = "left", tries: int = 3):
    for _ in range(tries):
        try:
            c = av.open(url, options={"rtsp_transport": "tcp", "stimeout": "8000000"})
            for frame in c.decode(video=0):
                img = frame.to_ndarray(format="bgr24")
                c.close()
                w = img.shape[1]
                if crop in ("left", "auto-left") and w >= 2600:
                    img = img[:, : w // 2]
                elif crop == "right" and w >= 2600:
                    img = img[:, w // 2:]
                return img
        except Exception as e:  # noqa: BLE001
            print(f"  retry {url}: {e}")
            time.sleep(1.0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", nargs="*", default=DEFAULT_STREAMS)
    ap.add_argument("--host", default="rtsp://118.140.234.166:8554")
    ap.add_argument("--crop", default="left")
    ap.add_argument("--out", default="output/rtsp_mine_300/samples")
    ap.add_argument("--skip-gray", action="store_true", help="跳过灰屏帧")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    idx = []
    for name in args.streams:
        img = grab(f"{args.host}/{name}", args.crop)
        if img is None:
            print(f"[FAIL] {name}")
            continue
        std = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std())
        p = out / f"{name}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        idx.append((name, img.shape[1], img.shape[0], round(std, 1)))
        print(f"[OK] {name} {img.shape[1]}x{img.shape[0]} gray_std={std:.1f} -> {p.name}", flush=True)
    (out / "index.txt").write_text(
        "\n".join(f"{n}\t{w}x{h}\tstd={s}" for n, w, h, s in idx), encoding="utf-8")
    print(f"\n{len(idx)} samples -> {out}")


if __name__ == "__main__":
    main()
