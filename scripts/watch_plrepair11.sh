#!/bin/bash
cd /workspaces/huawei
while pgrep -f "python3 train_plrepair11.py" > /dev/null; do sleep 60; done
sleep 90
RUN_DIR=$(ls -dt runs/detect/yolo26m_plrepair11_* | head -1)
echo "[$(date '+%F %T')] training done ($RUN_DIR), summary:"
python3 - "$RUN_DIR/results.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
last = rows[-1]
print(f"epochs_done={last['epoch']} best_epoch={best['epoch']} best_val_mAP50={float(best['metrics/mAP50(B)']):.4f} best_val_mAP50-95={float(best['metrics/mAP50-95(B)']):.4f} total_hours={float(last['time'])/3600:.2f}")
PYEOF
echo "[$(date '+%F %T')] starting eval (E0-11c vs plrepair11) on GPU 2..."
python3 scripts/eval_confusion.py --device 2 --imgsz 640 --data dataset/batch_12.v9i.yolov11/data.yaml --out /tmp/opencode/plrepair11_eval.json \
    --weights runs/detect/yolo26m_merge8_20260903_0950/weights/best.pt \
              "$RUN_DIR/weights/best.pt" > logs/plrepair11_eval.log 2>&1
echo "[$(date '+%F %T')] eval finished rc=$?"
python3 - <<'PYEOF' | tee logs/plrepair11_eval_summary.txt
import json
d = json.load(open("/tmp/opencode/plrepair11_eval.json"))
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
    ("Van R",[c['per_class']['Van']['R'] for c in confs]),
    ("Van->Car(PriCar)",[f"{c['Van_to_Car_pct']}%" for c in confs]),
    ("background->PriCar",[c["background_to_Car"] for c in confs]),
    ("Private Car P/R",[f"{c['per_class']['Private Car']['P']:.3f}/{c['per_class']['Private Car']['R']:.3f}" for c in confs]),
    ("Coach AP50",[ap50(v,"Coach") for v in vals]),
    ("Light Bus AP50",[ap50(v,"Light Bus") for v in vals]),
    ("HGV AP50",[ap50(v,"HGV") for v in vals]),
    ("LGV AP50",[ap50(v,"LGV") for v in vals]),
    ("MGV AP50",[ap50(v,"MGV") for v in vals]),
    ("F.Bus AP50",[ap50(v,"Franchised Bus") for v in vals]),
    ("PLB GMB AP50",[ap50(v,"PLB GMB") for v in vals]),
    ("Motorcycle AP50",[ap50(v,"Motorcycle") for v in vals]),
    ("Taxi AP50",[ap50(v,"Taxi") for v in vals])]:
        print(f"{n:20s} " + " ".join(f"{str(x):>19s}" for x in cells))
PYEOF
echo "[$(date '+%F %T')] watcher done."
