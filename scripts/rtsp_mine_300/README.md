# RTSP 补充采集 300 张（待标注）

目标：从 RTSP 补充采集 ~300 张高价值帧，输出到 `dataset/rtsp_mine_300_pending/`
（图 + X-AnyLabeling v4.0.5 兼容 prelabel JSON），便于后续在标注工具里改框改类。

## 状态（2026-09-14）

- **已完成**：300 张，7 路机位，1920×1080，无灰屏，JSON 全部合法。
- 结果见 `output/rtsp_mine_300/`：`stream_ranking.csv`、`run_*.csv`、`manifest`、
  `dataset_preview.jpg`、`verify_report.json`。

## 1. 流发现与选点

- 用户给的 `dahua866`/`dahua1003` 服务端 **404（未注册）**；`1000226/1000227/1000172` 等编号同样 404。
- 服务器只开放 RTSP 8554，无管理 API（9997 等关闭），故用**命名枚举**：`scan_streams.py`
  并发探测 `dahua1001500..dahua1002100`，命中 76 路（另有 1001619/1620/1640 为 3840×1080 双拼）。
- `rank_streams.py` 对每路抓 4 帧、用 11 类模型打分，`stream_ranking.csv` 排序，
  生成 `top_streams_montage.jpg` 供人眼筛选。
- 用户最终选定 8 路：`1001637 1001636 1001527 1001638 1001780 1001798 1001871 1001956`
  （其中 `1001798` 采集期无检出，实际 7 路贡献）。

## 2. 模型与口径

- **11 类（v9）模型**：`runs/detect/yolo26m_merge8_20260903_0950/weights/best.pt`
  （名含 merge8 但实为 v9 11 类，能直接分辨 LGV/HGV/Coach/Light Bus，匹配 `classes.txt`）。
- 推理 imgsz 640、conf_scoring 0.30（低阈召回稀有类）；prelabel 标签即来自该模型，人工复核。

## 3. 采集策略

- 8 路并行；送检 2s、同相机落盘最小间隔 30s；pHash 汉明 <8 去重；PrivateCar-only 占比 ≤15%。
- 灰屏过滤：灰度 std<12 且饱和<12 且三通道均值差<8 判废（`is_gray_screen`）。
- 3840×1080 双拼流按宽度自动裁左半（`auto-left`），本次 7 路均为 1920×1080 单幅，不裁。

## 4. 结果分布

| 机位 | 张数 |
|---|---:|
| dahua1001636 | 57 |
| dahua1001527 | 55 |
| dahua1001637 | 52 |
| dahua1001638 | 48 |
| dahua1001956 | 33 |
| dahua1001780 | 30 |
| dahua1001871 | 25 |
| dahua1001798 | 0 |
| **合计** | **300** |

- 稀有类命中帧：HGV 38、Motorcycle 41、LGV 27、PLB GMB 4。
- prelabel 框类分布：Private Car 152、MGV 126、Motorcycle 50、HGV 41、Taxi 36、Coach 35、LGV 28、Van 12、PLB GMB 5。
- 平均 1.62 框/图；尺寸全 1920×1080；最小灰度 std 50.4（无灰屏）。

## 5. 输出结构

```
dataset/rtsp_mine_300_pending/
  images/                  # YYYYMMDD_HHMMSS_ffffff_<cam>.jpg
  annotations_xany/        # 同名 .json（prelabel，11类字符串标签）
  classes.txt              # 11类 v9 口径
  manifest.csv             # file,cam,timestamp,score,n_box,n_cls,rare_hit,json_mode
```

## 6. 脚本

| 脚本 | 作用 |
|---|---|
| `scan_streams.py` | 并发枚举 RTSP 路径，产出 `discovered_streams_full.csv` |
| `rank_streams.py` | 抓帧+模型打分排序，产出 `stream_ranking.csv` / montage |
| `capture_samples.py` | 各路抓 1 帧存样便于人工确认机位 |
| `collect.py` | 正式采集（`--smoke-test` / `--target 300` / `--out-dir` 分批） |
| `merge_batches.py` | 合并多批，图片+同名 JSON 同目录（X-AnyLabeling 识别） |
| `survey_v9i.py` | 原始 v9i 类别分布统计 |
| `verify_dataset.py` | 核验数据集（尺寸/灰屏/JSON/分布）+ 预览拼图 |
| `export_xany.py` | YOLO框→X-AnyLabeling JSON 转换工具 |
