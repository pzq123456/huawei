# van2x single-variable experiment: E0 = merge8_0916, only change = Van-containing train images duplicated once
from datetime import datetime

from ultralytics import YOLO
from ultralytics.utils import SETTINGS


def main():
    SETTINGS["tensorboard"] = True

    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26m_plrepair11_{current_time}"

    model = YOLO("yolo26m.pt")
    model.train(
        data="dataset/batch_12.v14i.plrepair11.yolov11/data.yaml",
        epochs=500,
        patience=100,
        imgsz=640,
        batch=64,
        cos_lr=True,
        device=1,
        name=run_name,
        workers=8,
    )


if __name__ == "__main__":
    main()
