# 生产/候选模型登记

> 有生产价值的模型都记在这里，避免找不到。最后更新：2026-09-09

## 当前生产模型

| 模型 | 路径 | 数据 | val mAP50 / mAP50-95 | 状态 |
|---|---|---|---|---|
| **E0 (merge8 8类)** | `runs/detect/yolo26m_merge8_20260902_0916/weights/best.pt` | v10i merge8 (8类) | 0.8876 / 0.7046 | **生产在用**（main.py 已指向，2026-09-08 修正） |

## 候选（待决策是否替换生产）

| 模型 | 路径 | 数据 | val mAP50 / mAP50-95 | test bg→Car | 状态 |
|---|---|---|---|---|---|
| **plrepair-v2 (8类)** | `runs/detect/yolo26m_plrepair2_20260909_0023/weights/best.pt` | v13 merge8 + 295 伪框 | **0.8911 / 0.7084**（val 全面最强，Van AP +2.9pt） | 142 (E0: 149) | 候选：val 全面新高，但 test 未同步（±0.5pt 内），检出倾向更激进 |
| plrepair-v1 (8类) | `runs/detect/yolo26m_plrepair_20260908_0857/weights/best.pt` | v12 merge8 + 202 伪框 | 0.8814 / 0.7053 | **137**（三方最好） | 候选：保守修复版，bg→Car 改善最稳 |

## 历史/参考（不建议生产）

| 模型 | 路径 | 数据 | val | 备注 |
|---|---|---|---|---|
| merge8_0903_0950 (11类) | `runs/detect/yolo26m_merge8_20260903_0950/weights/best.pt` | v9i (11类) | 0.854 / 0.686 | 历史最好 11 类；**main.py 曾错误指向此模型（类数不匹配），已修正** |
| yolo26m_960_20260907_0815 (11类) | `runs/detect/yolo26m_960_20260907_0815/weights/best.pt` | v9i, imgsz 960 | 0.843 / 0.686 | 960 无收益，已封存 |
| van2x (8类) | `runs/detect/yolo26m_van2x_20260908_0030/weights/best.pt` | v11i Van 2× | 0.8812 / 0.7018 | Van 采样无效，仅作为伪标签共识 donor |

## 重要口径备注

- `background→Car`（val 125 / test 149）被**训练集漏标污染**：审计证明其中 ~2/3 是未标注真车（集中在 12004821、ANMR0000/0012 等来源）。该指标下降不完全等于误报减少。
- 伪标签 provenance：`dataset/batch_12.v12i.../pseudo_provenance.json`、`.../v13i.../pseudo_provenance.json`，可整批撤销。
- 推理 imgsz：**不要超过 640**（640 vs 1280 对照：1280 全面更差且慢 45%，数据集原生 640×640）。
- 同口径评估脚本：`scripts/eval_confusion.py`（conf≥0.25 / IoU 0.5 匹配）。
