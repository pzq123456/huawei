rtsp://118.140.234.166:8554/dahua1001619

## 目标类
Motorcycle
Private Car
Taxi
Light Van
LGV
MGV
HGV
Container
Public Light Bus/ GMB
Light Bus
Franchised Bus
Coach

## COCO 80 类

car
motorcycle
bus
truck

| COCO大类 | 细分子类 |
| :---: | :--- |
| **motorcycle** | Motorcycle |
| **car** | Private Car（私家车） |
| | Taxi（出租车） |
| | Light Van（轻型客货车） |
| **bus** | Public Light Bus / GMB（公共小巴） |
| | Light Bus（轻型巴士） |
| | Franchised Bus（专营巴士） |
| | Coach（长途/旅游巴士） |
| **truck** | LGV（轻型货车） |
| | MGV（中型货车） |
| | HGV（重型货车） |
| | Container（集装箱车） |


tar -czvf NoSuit_cls_clean_aug_split.tar.gz ./dataset/NoSuit_cls_clean_aug_split
tar -zxvf data.tar.gz