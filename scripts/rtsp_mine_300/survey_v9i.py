"""v9i 分布调查：输出 output/rtsp_mine_300/v9i_stats.json（已生成，可重复跑）。"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES11 = ['Coach', 'Franchised Bus', 'HGV', 'LGV', 'Light Bus', 'MGV',
           'Motorcycle', 'PLB GMB', 'Private Car', 'Taxi', 'Van']

def main():
    stats = {}
    for split in ['train', 'valid', 'test']:
        c = Counter()
        for f in (ROOT / f'dataset/batch_12.v9i.yolov11/{split}/labels').glob('*.txt'):
            for line in open(f, encoding='utf-8', errors='ignore'):
                p = line.split()
                if len(p) >= 5:
                    try:
                        c[NAMES11[int(float(p[0]))]] += 1
                    except (ValueError, IndexError):
                        pass
        stats[split] = dict(c)
    stats['names11'] = NAMES11
    stats['rare_priority'] = ['LGV', 'HGV', 'Motorcycle', 'PLB GMB', 'Coach', 'Light Bus', 'Van', 'MGV']
    out = ROOT / 'output/rtsp_mine_300/v9i_stats.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
