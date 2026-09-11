#!/bin/bash
cd /workspaces/huawei
while pgrep -f "python3 train_plrepair2.py" > /dev/null; do sleep 60; done
sleep 90
RUN_DIR=$(ls -dt runs/detect/yolo26m_plrepair2_* | head -1)
echo "[$(date '+%F %T')] training done ($RUN_DIR), summary:"
python3 - "$RUN_DIR/results.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
last = rows[-1]
print(f"epochs_done={last['epoch']} best_epoch={best['epoch']} best_val_mAP50={float(best['metrics/mAP50(B)']):.4f} best_val_mAP50-95={float(best['metrics/mAP50-95(B)']):.4f} total_hours={float(last['time'])/3600:.2f}")
PYEOF
echo "[$(date '+%F %T')] starting eval (E0 vs plrepair-v1 vs plrepair-v2) on GPU 2..."
python3 scripts/eval_confusion.py --device 2 --imgsz 640 --out /tmp/opencode/plrepair2_eval.json \
    --weights runs/detect/yolo26m_merge8_20260902_0916/weights/best.pt \
              runs/detect/yolo26m_plrepair_20260908_0857/weights/best.pt \
              "$RUN_DIR/weights/best.pt" > logs/plrepair2_eval.log 2>&1
echo "[$(date '+%F %T')] eval finished rc=$?"
python3 - <<'PYEOF' | tee logs/plrepair2_eval_summary.txt
import json
d = json.load(open("/tmp/opencode/plrepair2_eval.json"))
tags = list(d)
for split in ["valid","test"]:
    vals = [d[t][split] for t in tags]
    confs = [v["confusion"] for v in vals]
    def ap50(sp,c):
        v=sp["per_class"].get(c,{}).get("AP50"); return f"{v:.4f}" if v is not None else "-"
    print(f"\n===== {split} =====")
    print(f"{'metric':20s} " + " ".join(f"{t[:19]:>19s}" for t in tags))
    for n,cells in [("mAP50",[f"{v['mAP50']:.4f}" for v in vals]),
    ("mAP50-95",[f"{v['mAP50_95']:.4f}" for v in vals]),
    ("Van AP50",[ap50(v,"Van") for v in vals]),
    ("Van->Car",[f"{c['Van_to_Car_pct']}%" for c in confs]),
    ("<0.01 Van->Car",[f"{c['van_buckets'][0]['to_car_pct']}%" for c in confs]),
    ("background->Car",[c["background_to_Car"] for c in confs]),
    ("Van R",[c['per_class']['Van']['R'] for c in confs]),
    ("Car AP50",[ap50(v,"Private Car") for v in vals]),
    ("Car P",[c['per_class']['Private Car']['P'] for c in confs]),
    ("Taxi AP50",[ap50(v,"Taxi") for v in vals]),
    ("Bus AP50",[ap50(v,"Bus") for v in vals]),
    ("Truck AP50",[ap50(v,"Truck") for v in vals]),
    ("Motorcycle AP50",[ap50(v,"Motorcycle") for v in vals]),
    ("PLB GMB AP50",[ap50(v,"PLB GMB") for v in vals])]:
        print(f"{n:20s} " + " ".join(f"{str(x):>19s}" for x in cells))
PYEOF
echo "[$(date '+%F %T')] watcher done."
