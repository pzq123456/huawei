import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import av
import cv2
import torch
from ultralytics import YOLO

# ================= 全局配置变量 =================
RTSP_URL = "rtsp://118.140.234.166:8554/dahua1001619"
MODEL_PATH = r"C:\Users\admin\Desktop\personal\gprBox\yolo26m.pt"
OUT_DIR = Path("dataset/car_collected")
DETECT_INTERVAL = 1.0  # 检测抽帧间隔（秒）：每 1.0 秒送入 1 帧给模型检测
SAVE_INTERVAL = 15.0   # 截图落盘间隔（秒）：同一场景最少间隔 15 秒存一张
CONF_THRESH = 0.7
IMGSZ = 1280
DEVICE = 0
KEEP_CLASSES = [2, 3, 5, 7]  # car / motorcycle / bus / truck
# ===============================================


class EventAVStreamer:
    """基于事件驱动与时间戳降频抽帧的 RTSP 读取器"""

    def __init__(self, url, interval=1.0, maxsize=1):
        self.url = url
        self.interval = interval
        self.frame_queue = queue.Queue(maxsize=maxsize)
        self.container = None
        self.stopped = False
        threading.Thread(target=self._update, daemon=True).start()

    def _open(self):
        if self.container:
            try:
                self.container.close()
            except Exception:
                pass
        try:
            self.container = av.open(self.url, options={"rtsp_transport": "tcp", "stimeout": "8000000"})
            return True
        except Exception as e:
            print(f"[连接失败] {e}")
            self.container = None
            return False

    def _update(self):
        last_push_time = 0.0
        while not self.stopped:
            if not self.container and not self._open():
                time.sleep(2)
                continue
            try:
                for frame in self.container.decode(video=0):
                    if self.stopped:
                        return
                    
                    # 【降频抽帧过滤】：距离上一帧入队未达到设定间隔，直接跳过解码转换与入队
                    now = time.time()
                    if now - last_push_time < self.interval:
                        continue
                    
                    img = frame.to_ndarray(format="bgr24")
                    last_push_time = now

                    # 队列已满则替换为最新帧，保证系统不堆积
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(img)

            except Exception as e:
                print(f"[解码中断] {e}，尝试重连...")
                self.container = None
                time.sleep(1)

    def get_frame(self, timeout=1.0):
        try:
            return True, self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def stop(self):
        self.stopped = True
        if self.container:
            try:
                self.container.close()
            except Exception:
                pass


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("加载模型与 RTSP 流...")
    model = YOLO(MODEL_PATH, task="detect")
    streamer = EventAVStreamer(RTSP_URL, interval=DETECT_INTERVAL, maxsize=1)

    last_save, saved, t0 = 0.0, 0, time.time()
    print(f"开始降频抽帧采集，检测频率 {DETECT_INTERVAL}s/帧，保存间隔 {SAVE_INTERVAL}s")

    try:
        with torch.no_grad():
            while True:
                success, frame = streamer.get_frame(timeout=1.0)
                if not success:
                    continue

                left = frame[:, : frame.shape[1] // 2]
                results = model.predict(
                    left, conf=CONF_THRESH, imgsz=IMGSZ, verbose=False, device=DEVICE, classes=KEEP_CLASSES
                )

                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    now = time.time()
                    if now - last_save >= SAVE_INTERVAL:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        path = OUT_DIR / f"veh_{ts}.jpg"
                        cv2.imwrite(str(path), left, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        saved += 1
                        last_save = now

                        tags = [results[0].names[int(b.cls[0])] for b in boxes]
                        print(f"[{time.time()-t0:6.1f}s] 保存 {path.name} | 目标 {len(boxes)}个 ({','.join(tags)}) | 累计 {saved} 张", flush=True)

                del results, boxes

    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        print(f"\n采集结束，共保存 {saved} 张到 {OUT_DIR}")


if __name__ == "__main__":
    main()