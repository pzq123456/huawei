"""大华 ICC 直播 URL 提供器（HK 机房）。

背景：部分机位不在 RTSP 8554 网关，只在大华 ICC 可用。流程：
  1) 密码授权（RSA）取 access_token；
  2) POST MTS/Video/StartVideo -> 媒体服务器 RTSP 地址 + 会话 token；
  3) 公网段 URL 追加 &token=<会话token> 即得可拉流的 RTSP。

用法（自测）：
  python scripts/rtsp_mine_300/icc_rtsp_provider.py --device 1002605
  python scripts/rtsp_mine_300/icc_rtsp_provider.py --device 1002605 --open

凭据：scripts/rtsp_mine_300/icc_credentials.json（已 gitignore），或 ICC_* 环境变量。
详见 docs/icc_live_rtsp.md。
"""
import argparse
import base64
import json
import os
import ssl
import urllib.request
from pathlib import Path

CRED_PATH = Path(__file__).with_name("icc_credentials.json")
PUBKEY_PATH = "/evo-apigw/evo-oauth/1.0.0/oauth/public-key"
TOKEN_PATH = "/evo-apigw/evo-oauth/1.0.0/oauth/extend/token"
START_PATH = "/evo-apigw/admin/API/MTS/Video/StartVideo"
STOP_PATH = "/evo-apigw/admin/API/MTS/Video/StopVideo"
TIMEOUT = 20
VISIBLE = 0
MAIN_STREAM = 1
DATA_TYPE_URL = 1


def _ctx():
    return ssl._create_unverified_context()


def load_creds(code: str = "HK02", path: Path = CRED_PATH) -> dict:
    if path.is_file():
        d = json.loads(path.read_text(encoding="utf-8"))
        return d[code] if code in d else d
    keys = ("icc_ip", "client_id", "client_secret", "username", "password")
    env = {k: os.environ.get("ICC_" + k.upper()) for k in keys}
    if all(env.values()):
        return env
    raise SystemExit(f"ICC 凭据缺失：{path} 或 ICC_* 环境变量")


def _der_ints(der, want=2):
    out, i = [], 0

    def len_at(j):
        n = der[j]; j += 1
        if n & 0x80:
            k = n & 0x7F
            n = int.from_bytes(der[j:j + k], "big"); j += k
        return n, j

    while i < len(der) and len(out) < want:
        if der[i] == 0x02:
            n, i = len_at(i + 1)
            out.append(int.from_bytes(der[i:i + n], "big")); i += n
        else:
            i += 1
    return out


def _rsa_encrypt(pwd: str, key_b64: str) -> str:
    ints = _der_ints(base64.b64decode(key_b64))
    if len(ints) < 2:
        raise SystemExit("bad public key")
    n = max(ints)
    e = 65537
    for v in ints:
        if v < n and v < 2 ** 32:
            e = v
            break
    data = pwd.encode()
    k = (n.bit_length() + 7) // 8
    enc = b""
    for s in range(0, max(len(data), 1), k - 11):
        blk = data[s:s + k - 11]
        pad_len = k - 3 - len(blk)
        pad = b""
        while len(pad) < pad_len:
            b = os.urandom(1)
            if b != b"\x00":
                pad += b
        m = int.from_bytes(b"\x00\x02" + pad + b"\x00" + blk, "big")
        enc += pow(m, e, n).to_bytes(k, "big")
    return base64.b64encode(enc).decode()


def _post(url, payload, token=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"bearer {token}")
    with urllib.request.urlopen(req, context=_ctx(), timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


def _get(url):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, context=_ctx(), timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


class ICCProvider:
    """按需取 ICC 直播 URL；缓存 bearer，StartVideo 会话 token 每次开流重取。"""

    def __init__(self, creds: dict):
        self.icc = creds["icc_ip"].rstrip("/")
        self.cid = creds["client_id"]
        self.secret = creds["client_secret"]
        self.user = creds["username"]
        self.pwd = creds["password"]
        self._bearer = None

    def _login(self):
        pub = _get(self.icc + PUBKEY_PATH)["data"]["publicKey"]
        body = {
            "grant_type": "password",
            "username": self.user,
            "password": _rsa_encrypt(self.pwd, pub),
            "client_id": self.cid,
            "client_secret": self.secret,
            "public_key": pub,
        }
        res = _post(self.icc + TOKEN_PATH, body)
        if str(res.get("code")) != "0":
            raise SystemExit(f"token failed: {res}")
        self._bearer = res["data"]["access_token"]

    def bearer(self, refresh=False):
        if self._bearer is None or refresh:
            self._login()
        return self._bearer

    def start_video(self, device: str, channel: int = VISIBLE, stream_type: int = MAIN_STREAM) -> dict:
        payload = {"data": {"channelId": f"{device}$1$0${channel}",
                            "streamType": int(stream_type), "dataType": DATA_TYPE_URL}}
        for attempt in (0, 1):
            try:
                res = _post(self.icc + START_PATH, payload, self.bearer(refresh=attempt == 1))
                break
            except urllib.error.HTTPError as e:
                if e.code == 401 and attempt == 0:
                    continue
                raise SystemExit(f"StartVideo HTTP {e.code}")
        if res.get("code") != 1000 or not res.get("data", {}).get("url"):
            raise SystemExit(f"StartVideo failed: {res}")
        return res["data"]

    def live_url(self, device: str, channel: int = VISIBLE, stream_type: int = MAIN_STREAM) -> str:
        d = self.start_video(device, channel, stream_type)
        public = d["url"].split("|")[-1]
        return f"{public}&token={d['token']}"

    def stop_video(self, device: str, channel: int = VISIBLE):
        try:
            _post(self.icc + STOP_PATH, {"data": {"channelId": f"{device}$1$0${channel}"}}, self.bearer())
        except Exception:  # noqa: BLE001 - 释放失败不阻塞采集
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="HK02")
    ap.add_argument("--device", required=True)
    ap.add_argument("--channel", type=int, default=VISIBLE)
    ap.add_argument("--stream-type", type=int, default=MAIN_STREAM)
    ap.add_argument("--open", action="store_true", help="用 PyAV 试拉一帧")
    a = ap.parse_args()

    prov = ICCProvider(load_creds(a.code))
    url = prov.live_url(a.device, a.channel, a.stream_type)
    print(url)
    if a.open:
        import av
        c = av.open(url, options={"rtsp_transport": "tcp", "stimeout": "8000000"})
        for f in c.decode(video=0):
            print("frame", f.to_ndarray(format="bgr24").shape)
            c.close()
            break


if __name__ == "__main__":
    main()
