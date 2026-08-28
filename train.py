# yoloCons/train.py
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def main():
    SETTINGS["tensorboard"] = True
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26m_traffic_{current_time}"
    
    model = YOLO("yolo26m.pt")

    model.train(
        data="dataset/dataset_merged_v2/data.yaml",
        epochs=500,
        patience=100,
        imgsz=640,
        batch=64,
        device=-1,
        name=run_name,
        workers=8,
    )

if __name__ == "__main__":
    main()