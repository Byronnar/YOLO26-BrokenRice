# YOLO26 碎米检测 - 执行指令

## 环境准备

```bash
cd E:\others\code_demo\rice_detection\rice_yolo26
.\venv\Scripts\python.exe -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -e .
```

## 步骤

### 1. 重组数据集（已完成）

```bash
..\venv\Scripts\python.exe prepare_broken_rice.py
```

生成目录结构：

```
datasets/broken-rice-yolo/
├── images/train/  (2174 张)
├── images/val/    (261 张)
├── labels/train/  (2174 个)
└── labels/val/    (261 个)
```

### 2. 训练

```bash
# 默认：YOLO26s + 960 尺寸
..\venv\Scripts\python.exe train_broken_rice.py

# 使用 P2 检测头（更适合小目标）
..\venv\Scripts\python.exe train_broken_rice.py --p2

# 小尺寸快速实验
..\venv\Scripts\python.exe train_broken_rice.py --imgsz 640 --batch 64

# 更大模型
..\venv\Scripts\python.exe train_broken_rice.py --model m --batch 2
```

### 3. 推理

```..\venv\Scripts\python.exe
from ultralytics import YOLO
model = YOLO("outputs/broken_rice_yolo26/yolo26s_img960/weights/best.pt")
results = model.predict(source="path/to/image.jpg", imgsz=960, conf=0.25)

# 查看全部 261 张验证集
..\venv\Scripts\python.exe predict_broken_rice.py --num 0

# 指定单张图片
..\venv\Scripts\python.exe predict_broken_rice.py --source path/to/image.jpg --num 1

# 调低置信度看更多检测
..\venv\Scripts\python.exe predict_broken_rice.py --conf 0.15

# 更大尺寸推理
..\venv\Scripts\python.exe predict_broken_rice.py --imgsz 960


```

## 关键调优参数

| 参数           | 值   | 说明                                   |
| -------------- | ---- | -------------------------------------- |
| `cls_pw`       | 1.0  | 逆频率类别加权（应对 others 极度稀少） |
| `max_det`      | 300  | 密集目标场景增大检测上限               |
| `iou`          | 0.5  | 降低 NMS IoU 阈值，减少漏检            |
| `imgsz`        | 960  | 接近原始尺寸，保留小目标信息           |
| `cos_lr`       | True | 余弦退火学习率                         |
| `close_mosaic` | 15   | 最后 15 轮关闭 Mosaic 增强             |

## 损失函数改进

### IoU 损失类型参数化切换

支持通过训练参数 `--box_loss_type` 一键切换不同的 IoU 损失函数，**默认 CIoU，不破坏原有逻辑**。

**支持的类型**：
| box_loss_type | 说明 |
|---------------|------|
| `ciou` | **默认**，Complete IoU |
| `shape_iou` | Shape-IoU，形状+尺度感知（论文: arxiv.org/abs/2312.17663） |
| `diou` | Distance IoU |
| `giou` | Generalized IoU |

**使用方法**：

```bash
# 默认 CIoU（与原来完全一致）
..\venv\Scripts\python.exe train_broken_rice.py

# 使用 Shape-IoU
..\venv\Scripts\python.exe train_broken_rice.py --imgsz 640 --batch 64 --box_loss_type shape_iou --shape_iou_scale 1.0

# 使用 DIoU
..\venv\Scripts\python.exe train_broken_rice.py --box_loss_type diou
```

**Shape-IoU 的 scale 参数**：
| scale | 适用场景 |
|-------|---------|
| 0 | 通用（均匀权重，接近 DIoU+形状代价） |
| 1 | **碎米数据集推荐**（适度形状感知） |
| 2-3 | 目标宽高比差异大时使用 |

**Shape-IoU 核心优势**：

- **形状感知加权**：基于 GT 框宽高计算形状权重 ww/hh，短边方向偏差获得更高惩罚
- **尺度感知**：`scale` 参数控制形状权重敏感度，scale=0 退化为均匀权重
- **公式**：`L = 1 - IoU + distance^shape + 0.5 * Ω^shape`

**修改文件**：

- `ultralytics/utils/metrics.py` — 新增 `shape_iou()` 函数（原 `bbox_iou` 保留）
- `ultralytics/utils/loss.py` — `BboxLoss` 新增 `box_loss_type`/`shape_iou_scale` 参数
- `ultralytics/utils/tal.py` — `TaskAlignedAssigner` 新增 `box_loss_type`/`shape_iou_scale` 参数
- `ultralytics/cfg/default.yaml` — 新增 `box_loss_type`/`shape_iou_scale` 配置项

## 精度提升

1. 640分辨率，batch-size=64
   Class Images Instances Box(P R mAP50 mAP50-95): 100% ━━━━━━━━━━━━ 5/5 2.1it/s 2.4s
   all 261 3938 0.977 0.971 0.992 0.827
   rice 261 2833 0.972 0.985 0.992 0.924
   broken 259 1099 0.99 0.929 0.989 0.851
   others 6 6 0.969 1 0.995 0.706
   Speed: 1.8ms preprocess, 1.4ms inference, 0.0ms loss, 0.2ms postprocess per image
   Results saved to E:\projects\InfiniteTalk-20260201\Infinitetalk\runs\detect\val

   mAP50: 0.9922
   mAP50-95: 0.8267
   rice: mAP50=0.9238
   broken: mAP50=0.8507
   others: mAP50=0.7057

2. 640分辨率，batch-size=64，shape_iou_scale=1.0
   100% ━━━━━━━━━━━━ 261/261 0.0s
   Class Images Instances Box(P R mAP50 mAP50-95): 100% ━━━━━━━━━━━━ 5/5 1.7it/s 2.9s
   all 261 3938 0.944 0.983 0.986 0.829
   rice 261 2833 0.993 0.985 0.994 0.938
   broken 259 1099 0.994 0.964 0.993 0.863
   others 6 6 0.845 1 0.972 0.687
   Speed: 2.5ms preprocess, 1.3ms inference, 0.0ms loss, 0.3ms postprocess per image
   Results saved to E:\projects\InfiniteTalk-20260201\Infinitetalk\runs\detect\val-2

mAP50: 0.9863
mAP50-95: 0.8293
rice: mAP50=0.9382
broken: mAP50=0.8628
