# train_merged12.py
# dataset : dataset/batch_12.v14i.merged12.yolov11  (12类: v9 11类 + Container)
# results : runs/detect/yolo26m_merged12_<YYYYMMDD_HHMM>
from datetime import datetime

from ultralytics import YOLO
from ultralytics.utils import SETTINGS


def main():
    SETTINGS["tensorboard"] = True

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26m_merged12_{current_time}"

    model = YOLO("yolo26m.pt")
    model.train(
        data="dataset/batch_12.v14i.merged12.yolov11/data.yaml",
        epochs=500,
        patience=100,
        imgsz=640,
        batch=64,
        cos_lr=True,
        device=-1,
        name=run_name,
        workers=8,
    )


if __name__ == "__main__":
    main()
