#!/bin/bash
# Watches van2x training; on completion runs same-caliber eval (E0 vs van2x) and writes ordered summary.
cd /workspaces/huawei
E0=runs/detect/yolo26m_merge8_20260902_0916/weights/best.pt
RUN_DIR=runs/detect/yolo26m_van2x_20260908_0030

echo "[$(date '+%F %T')] watcher started, waiting for training to finish..."
while pgrep -f "python3 train_van2x.py" > /dev/null; do
    sleep 60
done
echo "[$(date '+%F %T')] training process gone, waiting 90s for final flush..."
sleep 90

echo "[$(date '+%F %T')] training summary:"
python3 - "$RUN_DIR/results.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
last = rows[-1]
print(f"epochs_done={last['epoch']} best_epoch={best['epoch']} "
      f"best_val_mAP50={float(best['metrics/mAP50(B)']):.4f} best_val_mAP50-95={float(best['metrics/mAP50-95(B)']):.4f} "
      f"total_hours={float(last['time'])/3600:.2f}")
EOF

echo "[$(date '+%F %T')] starting eval (E0 vs van2x) on GPU 2..."
python3 scripts/eval_confusion.py --device 2 \
    --weights "$E0" "$RUN_DIR/weights/best.pt" \
    --out /tmp/opencode/van2x_eval.json \
    > logs/van2x_eval.log 2>&1
rc=$?
echo "[$(date '+%F %T')] eval finished rc=$rc"

python3 - <<'EOF' | tee logs/van2x_eval_summary.txt
import json
d = json.load(open("/tmp/opencode/van2x_eval.json"))
tags = list(d)
v0, v1 = d[tags[0]]["valid"], d[tags[1]]["valid"]
t0, t1 = d[tags[0]]["test"], d[tags[1]]["test"]
c0, c1 = v0["confusion"], v1["confusion"]
k0, k1 = t0["confusion"], t1["confusion"]
def row(n, a, b): print(f"{n:24s} {a:>12s} {b:>12s}")
def ap50(sp, c):
    v = sp["per_class"].get(c, {}).get("AP50")
    return f"{v:.4f}" if v is not None else "-"
print("\n=== VAL (309 imgs) — priority order ===")
row("metric", tags[0][:24], tags[1][:24])
row("Van AP50", ap50(v0, "Van"), ap50(v1, "Van"))
row("Van->Car %", f"{c0['Van_to_Car_pct']:.1f} ({c0['Van_to_Car'][0]}/{c0['Van_to_Car'][1]})", f"{c1['Van_to_Car_pct']:.1f} ({c1['Van_to_Car'][0]}/{c1['Van_to_Car'][1]})")
row("<0.01 Van->Car %", f"{c0['van_buckets'][0]['to_car_pct']:.1f}", f"{c1['van_buckets'][0]['to_car_pct']:.1f}")
row("background->Car", c0["background_to_Car"], c1["background_to_Car"])
row("Van R", f"{c0['per_class']['Van']['R']:.3f}", f"{c1['per_class']['Van']['R']:.3f}")
row("mAP50", f"{v0['mAP50']:.4f}", f"{v1['mAP50']:.4f}")
row("mAP50-95", f"{v0['mAP50_95']:.4f}", f"{v1['mAP50_95']:.4f}")
row("Van P", f"{c0['per_class']['Van']['P']:.3f}", f"{c1['per_class']['Van']['P']:.3f}")
row("Car AP50", ap50(v0, "Private Car"), ap50(v1, "Private Car"))
row("PLB GMB AP50", ap50(v0, "PLB GMB"), ap50(v1, "PLB GMB"))
row("Motorcycle AP50", ap50(v0, "Motorcycle"), ap50(v1, "Motorcycle"))
print("\n=== TEST (319 imgs) ===")
row("metric", tags[0][:24], tags[1][:24])
row("Van AP50", ap50(t0, "Van"), ap50(t1, "Van"))
row("Van->Car %", f"{k0['Van_to_Car_pct']:.1f} ({k0['Van_to_Car'][0]}/{k0['Van_to_Car'][1]})", f"{k1['Van_to_Car_pct']:.1f} ({k1['Van_to_Car'][0]}/{k1['Van_to_Car'][1]})")
row("<0.01 Van->Car %", f"{k0['van_buckets'][0]['to_car_pct']:.1f}", f"{k1['van_buckets'][0]['to_car_pct']:.1f}")
row("background->Car", k0["background_to_Car"], k1["background_to_Car"])
row("Van R", f"{k0['per_class']['Van']['R']:.3f}", f"{k1['per_class']['Van']['R']:.3f}")
row("mAP50", f"{t0['mAP50']:.4f}", f"{t1['mAP50']:.4f}")
row("mAP50-95", f"{t0['mAP50_95']:.4f}", f"{t1['mAP50_95']:.4f}")
row("Car AP50", ap50(t0, "Private Car"), ap50(t1, "Private Car"))
EOF
echo "[$(date '+%F %T')] watcher done."
