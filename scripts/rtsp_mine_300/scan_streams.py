"""并发扫描 RTSP 流命名空间下的可用路径。

用法:
  python scripts/rtsp_mine_300/scan_streams.py --start 1001500 --end 1002050 --workers 16
输出:
  output/rtsp_mine_300/discovered_streams_full.csv  (name,width,height)
"""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parents[2]


def probe(url: str, timeout_s: float = 5.0):
    try:
        c = av.open(url, options={"rtsp_transport": "tcp", "stimeout": str(int(timeout_s * 1e6))})
        vstreams = [s for s in c.streams if s.type == "video"]
        if not vstreams:
            c.close()
            return None
        w = vstreams[0].codec_context.width
        h = vstreams[0].codec_context.height
        c.close()
        return (w, h)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="rtsp://118.140.234.166:8554")
    ap.add_argument("--prefix", default="dahua")
    ap.add_argument("--start", type=int, default=1001500)
    ap.add_argument("--end", type=int, default=1002050)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--out", default="output/rtsp_mine_300/discovered_streams_full.csv")
    args = ap.parse_args()

    nums = list(range(args.start, args.end + 1))
    results = []
    print(f"scanning {len(nums)} paths with {args.workers} workers ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(probe, f"{args.host}/{args.prefix}{n}", args.timeout): n for n in nums}
        done = 0
        for fut in as_completed(futs):
            n = futs[fut]
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(nums)} ... found {len(results)}", flush=True)
            r = fut.result()
            if r:
                w, h = r
                results.append((f"{args.prefix}{n}", w, h))
                print(f"[OK] {args.prefix}{n}\t{w}x{h}", flush=True)

    results.sort()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["name", "width", "height"])
        wtr.writerows(results)
    print(f"\nfound {len(results)} streams -> {out}")
    print(json.dumps([r[0] for r in results], ensure_ascii=False))


if __name__ == "__main__":
    main()
