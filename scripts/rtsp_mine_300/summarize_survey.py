"""汇总 survey_streams.py 的调查结果，打印可读排名。"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/rtsp_mine_300/survey_round2")
    ap.add_argument("--min-avg", type=float, default=0.6)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(ROOT / a.dir / "survey.csv", encoding="utf-8")))

    def f(r, k):
        try:
            return float(r.get(k) or 0)
        except ValueError:
            return 0.0

    live = [r for r in rows if r["connected"] == "True" and f(r, "avg_boxes") >= a.min_avg]
    live.sort(key=lambda r: f(r, "avg_boxes"), reverse=True)
    head = (f"{'name':<16}{'avg':>5}{'cplx':>6}{'mW':>6}{'moto':>5}{'cont':>5}"
            f"{'lgv':>4}{'hgv':>4}{'van':>4}{'mgv':>4}{'car':>5}{'taxi':>5}{'static':>7}")
    print(head)
    for r in live:
        print(f"{r['name']:<16}{f(r,'avg_boxes'):>5.2f}{f(r,'complex_frac'):>6.2f}"
              f"{f(r,'multi_weak_frac'):>6.2f}{r['motorcycle']:>5}{r['container']:>5}"
              f"{r['lgv']:>4}{r['hgv']:>4}{r['van']:>4}{r['mgv']:>4}"
              f"{r['private_car']:>5}{r['taxi']:>5}{f(r,'static_frac'):>7.2f}")
    print(f"\n合计 {len(rows)} 路 | 未连接 {sum(1 for r in rows if r['connected']!='True')} | "
          f"零检测 {sum(1 for r in rows if r['connected']=='True' and f(r,'avg_boxes')==0)} | "
          f"可用(avg>={a.min_avg}) {len(live)}")


if __name__ == "__main__":
    main()
