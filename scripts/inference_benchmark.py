import time

import torch
from ultralytics import YOLO

import glob

W = "runs/detect/yolo26m_merge8_20260902_0916/weights/best.pt"
IMGS = sorted(glob.glob("dataset/batch_12.v10i.merge8.yolov11/valid/images/*"))[:100]
DEVICE = 2

model = YOLO(W)
torch.cuda.init()
for imgsz in (640, 1280):
    torch.cuda.reset_peak_memory_stats(DEVICE)
    for _ in range(10):  # warmup
        model.predict(IMGS[0], conf=0.4, imgsz=imgsz, verbose=False, device=DEVICE)
    torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()
    for img in IMGS:
        model.predict(img, conf=0.4, imgsz=imgsz, verbose=False, device=DEVICE)
    torch.cuda.synchronize(DEVICE)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    reserved = torch.cuda.max_memory_reserved(DEVICE) / 1024**2
    n = len(IMGS)
    print(f"imgsz={imgsz}: {dt*1000/n:.1f} ms/frame  {n/dt:.1f} FPS  peak_alloc={peak:.0f}MB  peak_reserved={reserved:.0f}MB")
