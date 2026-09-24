"""按机位(文件名后缀)统计各弱类产量，用于选路。"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['Coach', 'Franchised Bus', 'HGV', 'LGV', 'Light Bus', 'MGV',
         'Motorcycle', 'PLB GMB', 'Private Car', 'Taxi', 'Van', 'Container']
FOCUS = ['Motorcycle', 'Container', 'LGV', 'HGV', 'Van', 'MGV']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/batch_12.v14i.merged12.yolov11")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    root = ROOT / a.data
    per = defaultdict(Counter)
    tot = Counter()
    for split in ("train", "valid", "test"):
        for t in (root / split / "labels").glob("*.txt"):
            cam = t.stem.rsplit("_", 1)[-1]
            for line in open(t, encoding="utf-8", errors="ignore"):
                p = line.split()
                if len(p) >= 5:
                    try:
                        c = int(float(p[0]))
                    except ValueError:
                        continue
                    per[cam][NAMES[c]] += 1
                    tot[NAMES[c]] += 1
    print("总框:", {k: tot[k] for k in NAMES})
    for focus in FOCUS:
        print(f"\n== {focus} 产量最高的机位 ==")
        ranked = sorted(per.items(), key=lambda kv: kv[1][focus], reverse=True)
        for cam, c in ranked[:a.top]:
            if c[focus] == 0:
                break
            extra = " ".join(f"{k}={c[k]}" for k in FOCUS if k != focus and c[k])
            print(f"  {cam:24s} {focus}={c[focus]:4d} total={sum(c.values()):5d}  {extra}")


if __name__ == "__main__":
    main()
