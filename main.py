import av
import cv2
import threading
import time
import torch
from ultralytics import YOLO

# ================= 配置参数 =================
RTSP_URL = "rtsp://118.140.234.166:8554/dahua1001619"
# RTSP_URL = "rtsp://118.140.234.166:8554/dahua1001620"
MODEL_PATH = r"best.pt"
CONF_THRES = 0.4
DEVICE = 0

# 推理图像尺寸：建议设为 1280 或 1920，避免直接传入 (h, w) 导致内部 resize 扭曲
IMGSZ = 1280

# 设为 None 表示不过滤，检测模型训练好的全部 12 个类别
KEEP_CLASSES = None 


class AVStreamer:
    """基于 PyAV 的多线程 RTSP 读取类"""
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.container = None
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.stopped = False

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _open(self):
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
        try:
            self.container = av.open(
                self.rtsp_url,
                options={"rtsp_transport": "tcp", "stimeout": "8000000"},
            )
            return True
        except Exception as e:
            print(f"[连接失败] {e}")
            self.container = None
            return False

    def update(self):
        while not self.stopped:
            if self.container is None:
                if not self._open():
                    time.sleep(2.0)
                    continue

            try:
                for frame in self.container.decode(video=0):
                    if self.stopped:
                        break
                    img = frame.to_ndarray(format="bgr24")
                    with self.lock:
                        self.frame = img
                        self.ret = True
            except Exception as e:
                print(f"[解码中断] {e}，尝试重连...")
                with self.lock:
                    self.ret = False
                self._open()
                time.sleep(1.0)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass


def main():
    print("加载自定义交通模型中...")
    model = YOLO(MODEL_PATH, task="detect")

    print(f"连接 RTSP... {RTSP_URL}")
    streamer = AVStreamer(RTSP_URL)

    # 使用 WINDOW_AUTOSIZE 保持原生 1:1 像素映射，防止窗口缩放导致画面双线性插值变糊
    cv2.namedWindow("Traffic Preview", cv2.WINDOW_AUTOSIZE)

    print("开始实时预览（按 'q' 或 'ESC' 退出）...")

    try:
        with torch.no_grad():
            while True:
                success, frame = streamer.read()
                if not success or frame is None:
                    time.sleep(0.01)
                    continue

                # 截取左半区域
                left = frame[:, : frame.shape[1] // 2]

                # YOLO 追踪推理
                results = model.track(
                    left,
                    conf=CONF_THRES,
                    imgsz=IMGSZ,
                    verbose=False,
                    device=DEVICE,
                    classes=KEEP_CLASSES,
                    persist=True,
                    tracker="bytetrack.yaml",
                )

                # 修正参数名：line_thickness -> line_width，移除不兼容的 font_size
                annotated_frame = results[0].plot(
                    line_width=1,  # 强制 1 像素细线条，解决粗框变糊问题
                    pil=False,     # 使用 OpenCV 原生绘制，边缘更清晰
                    boxes=True,
                    labels=True,
                    probs=False,
                )

                cv2.imshow("Traffic Preview", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()
        cv2.destroyAllWindows()
        print("预览结束。")


if __name__ == "__main__":
    main()