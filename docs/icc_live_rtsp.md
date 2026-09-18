# 大华 ICC 直播拉流（HK02）接入发现

日期：2026-09-17 ｜ 环境：HK02 `118.140.234.166` ｜ 关联脚本：`scripts/rtsp_mine_300/icc_rtsp_provider.py`

## 背景

需要把 4 路新机位（OSD 2181/2200/2249/2250）加入车辆数据集采集。它们**不在** RTSP 8554 网关上，
但在大华 ICC（云平台）里可用。本文记录可复现的拉流方式。

## 关键结论

- RTSP `8554` 网关（原采集管线用的 `rtsp://118.140.234.166:8554/dahua<编号>`）**没有这 4 路**：
  `DESCRIBE` 返回 `404 Not Found`。用 `OPTIONS` 探测会**假阳性**（网关对任意路径都回 `200 OK`），必须用
  `DESCRIBE`/`av.open` 判存在。
- 这 4 路的录像在 ICC 有（`QueryRecords` 各 74 段，`recordSource=2`=设备 SD 卡）。
- ICC 提供**直播 URL 接口** `MTS/Video/StartVideo`，返回媒体服务器 RTSP 地址；追加会话 `token` 后可拉流。

## 可复现配方

1. 取 ICC access_token（密码授权，RSA 加密口令）：
   - `GET {icc}/evo-apigw/evo-oauth/1.0.0/oauth/public-key`
   - `POST {icc}/evo-apigw/evo-oauth/1.0.0/oauth/extend/token`
     body: `{grant_type:password, username, password:<RSA>, client_id, client_secret, public_key}`
   - 注意：`/evo-apigw/oauth/oauth2/token`（client_credentials）在本服务器 **404**，不可用。
2. 开流：
   - `POST {icc}/evo-apigw/admin/API/MTS/Video/StartVideo`
   - body: `{"data":{"channelId":"<设备码>$1$0$0","streamType":1,"dataType":1}}`
     - `streamType`、`dataType` 必须是**无符号整数**（传字符串报 2054/2056）。
     - `dataType=0` 报 2055；用 `1`。
   - 返回 `data.url` = `内网|公网` 两段，另含 `data.token`、`data.session`、`data.wsUrl`。
3. 取公网段，追加 `&token=<data.token>`，用 RTSP 拉流。

## 实测（2026-09-17，1920×1080）

| OSD | 设备码 | 公网 URL | 端口 |
|---|---|---|---|
| 2181 | 1002605 | `:9100/mediaServer/monitor/param/cameraid=1002605%240%26substream=1&token=…` | 9100 |
| 2200 | 1002689 | `:9100/mediaServer/monitor/param/cameraid=1002689%240%26substream=1&token=…` | 9100 |
| 2249 | 1002712 | `:9108/dss/monitor/param/cameraid=1002712%240%26substream=1&token=…` | 9108 |
| 2250 | 1002711 | `:9104/dss/monitor/param/cameraid=1002711%240%26substream=1&token=…` | 9104 |

URL 编码：`%24`=`$`、`%26`=`&`；`cameraid=<设备码>$0`（0=可见光），`substream=1`。

易错点：
- 不追加 `&token=` → `401 Unauthorized`。
- token 放 `user:pass@`、`?token=`、`%26token=` 均无效；只有 **`&token=`** 有效。
- `2249` 首连偶发 "Invalid data found"（transient），重试即可。

## 端口/网关

- HK02 从公网可达直播媒体端口：`9100 / 9104 / 9108`。`8554`（RTSP 网关）无此 4 路。
- HK02 `7086/9090`（文档里的 `streamr_ip`）从当前网络**不可达**；HK01 `118.140.130.26:7086` 可达但这 4 路不属 HK01。
- ICC 网页：`https://118.140.234.166:8092`。管理口 `9997` 在 HK01 开放但需授权。

## 与采集管线的关系

- `collect.py` 的 `StreamReader` 原本吃静态 URL；ICC 流需在**开流/重连时**调用 `StartVideo` 取新 URL+token。
  已改为接受 callable URL（`icc_device` 配置项），并提供 `icc_rtsp_provider.py`。
- 凭据放 `scripts/rtsp_mine_300/icc_credentials.json`（已 gitignore），勿提交。
