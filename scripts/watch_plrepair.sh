#!/bin/bash
cd /workspaces/huawei
E0=runs/detect/yolo26m_merge8_20260902_0916/weights/best.pt
RUN_DIR=$(ls -dt runs/detect/yolo26m_plrepair_* | head -1)
echo "[$(date '+%F %T')] watcher started, waiting for training to finish ($RUN_DIR)..."
while pgrep -f "python3 train_plrepair.py" > /dev/null; do sleep 60; done
sleep 90
echo "[$(date '+%F %T')] training summary:"
python3 - "$RUN_DIR/results.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
last = rows[-1]
print(f"epochs_done={last['epoch']} best_epoch={best['epoch']} best_val_mAP50={float(best['metrics/mAP50(B)']):.4f} best_val_mAP50-95={float(best['metrics/mAP50-95(B)']):.4f} total_hours={float(last['time'])/3600:.2f}")
PYEOF
echo "[$(date '+%F %T')] starting eval (E0 vs plrepair) on GPU 2..."
python3 scripts/eval_confusion.py --device 2 --imgsz 640 --out /tmp/opencode/plrepair_eval.json \
    --weights "$E0" "$RUN_DIR/weights/best.pt" > logs/plrepair_eval.log 2>&1
echo "[$(date '+%F %T')] eval finished rc=$?"
python3 - <<'PYEOF' | tee logs/plrepair_eval_summary.txt
import json
d = json.load(open("/tmp/opencode/plrepair_eval.json"))
tags = list(d)
for split in ["valid","test"]:
    v0, v1 = d[tags[0]][split], d[tags[1]][split]
    c0, c1 = v0["confusion"], v1["confusion"]
    def ap50(sp,c):
        v=sp["per_class"].get(c,{}).get("AP50"); return f"{v:.4f}" if v is not None else "-"
    print(f"\n===== {split} =====")
    print(f"{'metric':20s} {tags[0][:26]:>26s} {tags[1][:26]:>26s}")
    for n,a,b in [("mAP50",f"{v0['mAP50']:.4f}",f"{v1['mAP50']:.4f}"),
    ("mAP50-95",f"{v0['mAP50_95']:.4f}",f"{v1['mAP50_95']:.4f}"),
    ("Van AP50",ap50(v0,"Van"),ap50(v1,"Van")),
    ("Van->Car",f"{c0['Van_to_Car_pct']}%",f"{c1['Van_to_Car_pct']}%"),
    ("<0.01 Van->Car",f"{c0['van_buckets'][0]['to_car_pct']}%",f"{c1['van_buckets'][0]['to_car_pct']}%"),
    ("background->Car",c0["background_to_Car"],c1["background_to_Car"]),
    ("Van R",c0['per_class']['Van']['R'],c1['per_class']['Van']['R']),
    ("Car AP50",ap50(v0,"Private Car"),ap50(v1,"Private Car")),
    ("Car P",c0['per_class']['Private Car']['P'],c1['per_class']['Private Car']['P']),
    ("Taxi AP50",ap50(v0,"Taxi"),ap50(v1,"Taxi")),
    ("Bus AP50",ap50(v0,"Bus"),ap50(v1,"Bus")),
    ("Truck AP50",ap50(v0,"Truck"),ap50(v1,"Truck")),
    ("Motorcycle AP50",ap50(v0,"Motorcycle"),ap50(v1,"Motorcycle"))]:
        print(f"{n:20s} {str(a):>26s} {str(b):>26s}")
PYEOF
echo "[$(date '+%F %T')] watcher done."
