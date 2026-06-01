# YOLO26-BrokenRice: Shape-IoU 增强的碎米目标检测

基于 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) 的碎米质量检测项目，支持 Shape-IoU 等 4 种 IoU 损失函数一键切换，专为碎米检测场景优化。

## 特性

- **多 IoU 损失函数支持**：CIoU / Shape-IoU / DIoU / GIoU，通过参数一键切换
- **Shape-IoU**：引入形状感知与尺度感知的边界框回归损失，对小目标更友好（[论文](https://arxiv.org/abs/2312.17663)）
- **P2 检测头**：可选增加 stride=4 检测层，提升小目标检测精度
- **类别加权**：逆频率加权应对 `others` 类极度稀少问题
- **数据集一键重组**：自动将 broken-rice-detection 数据集转为 YOLO 标准格式
- **推理可视化**：支持批量推理并绘制检测框

## 环境准备

```bash
# 克隆仓库
git clone https://github.com/ < 你的用户名 > /YOLO26-BrokenRice.git
cd YOLO26-BrokenRice

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖（开发模式）
pip install -e .
pip install psutil # RAM 缓存加速（可选）
```

## 快速开始

### 1. 准备数据集

将 [broken-rice-detection](https://github.com/Yangr116/broken-rice-detection) 数据集放到 `../datasets/broken-rice-detection-main/` 目录，然后运行：

```bash
python prepare_broken_rice.py
```

自动生成 YOLO 标准格式（符号链接，不占额外磁盘）：

```
datasets/broken-rice-yolo/
├── images/
│   ├── train/  (2174 张)
│   └── val/    (261 张)
└── labels/
    ├── train/  (2174 个)
    └── val/    (261 个)
```

3 个检测类别：`rice`(0)、`broken`(1)、`others`(2)

### 2. 训练

```bash
# 默认配置：YOLO26s + CIoU + 960 尺寸
python train_broken_rice.py

# 使用 Shape-IoU（推荐碎米场景）
python train_broken_rice.py --box_loss_type shape_iou --shape_iou_scale 1.0

# 小尺寸快速实验
python train_broken_rice.py --imgsz 640 --batch 64

# 使用 P2 检测头（增加小目标检测层）
python train_broken_rice.py --p2

# 更大模型
python train_broken_rice.py --model m --batch 2
```

### 3. 推理与可视化

```bash
# 自动查找最新模型，可视化验证集前 20 张
python predict_broken_rice.py

# 指定模型和图片
python predict_broken_rice.py --model path/to/best.pt --source path/to/image.jpg

# 调低置信度、更大尺寸
python predict_broken_rice.py --conf 0.15 --imgsz 960

# 可视化全部验证集
python predict_broken_rice.py --num 0
```

## IoU 损失函数

支持通过 `--box_loss_type` 训练参数一键切换，**默认 CIoU，不破坏原有逻辑**。

| box_loss_type | 说明                     | 适用场景            |
| ------------- | ------------------------ | ------------------- |
| `ciou`        | **默认**，Complete IoU   | 通用                |
| `shape_iou`   | Shape-IoU，形状+尺度感知 | **碎米/小目标推荐** |
| `diou`        | Distance IoU             | 关注中心距离        |
| `giou`        | Generalized IoU          | 无重叠框优化        |

### Shape-IoU 原理

传统 IoU 变体（GIoU/CIoU/SIoU）只关注预测框与 GT 框之间的相对几何关系，忽略了边界框自身形状与尺度对回归敏感度的影响。Shape-IoU 核心创新：

1. **形状感知加权**：基于 GT 框宽高计算权重 `ww`/`hh`，短边方向偏差获得更高惩罚
2. **尺度感知**：`scale` 参数控制形状权重敏感度，小目标自动获得更强回归约束
3. **损失公式**：`L = 1 - IoU + distance^shape + 0.5 × Ω^shape`

### Shape-IoU 的 scale 参数

| scale | 适用场景                               |
| ----- | -------------------------------------- |
| 0     | 通用（均匀权重，接近 DIoU + 形状代价） |
| 1     | **碎米数据集推荐**（适度形状感知）     |
| 2-3   | 目标宽高比差异大时使用                 |

## 训练参数说明

| 参数                | 默认值  | 说明                                  |
| ------------------- | ------- | ------------------------------------- |
| `--model`           | `s`     | 模型规模 (n/s/m/l/x)                  |
| `--imgsz`           | `960`   | 输入图像尺寸                          |
| `--epochs`          | `150`   | 训练轮数                              |
| `--batch`           | `4`     | 批大小（960 尺寸建议 4，640 建议 64） |
| `--p2`              | `False` | 使用 P2 检测头（增加小目标检测层）    |
| `--box_loss_type`   | `ciou`  | IoU 损失类型                          |
| `--shape_iou_scale` | `0.0`   | Shape-IoU 的 scale 因子               |
| `--device`          | `0`     | 训练设备 (0/1/cpu)                    |
| `--resume`          | `None`  | 从 checkpoint 恢复训练                |

### 碎米检测关键调优

| 调优项            | 值                    | 原因                         |
| ----------------- | --------------------- | ---------------------------- |
| `cls_pw=1.0`      | 逆频率加权            | others 类仅 0.2%，极度不均衡 |
| `max_det=300`     | 增大检测上限          | 每图平均 14.9 个目标         |
| `iou=0.5`         | 降低 NMS 阈值         | 密集排列，减少漏检           |
| `close_mosaic=15` | 最后 15 轮关闭 Mosaic | 稳定最终收敛                 |
| `cos_lr=True`     | 余弦退火              | 小数据集更稳定收敛           |

## 实验结果

在 broken-rice-detection 数据集上的验证集结果：

### YOLO26s + CIoU（640 分辨率，batch=64）

| 类别    | Images | Instances | P     | R     | mAP50 | mAP50-95 |
| ------- | ------ | --------- | ----- | ----- | ----- | -------- |
| **all** | 261    | 3938      | 0.977 | 0.971 | 0.992 | 0.827    |
| rice    | 261    | 2833      | 0.972 | 0.985 | 0.992 | 0.924    |
| broken  | 259    | 1099      | 0.990 | 0.929 | 0.989 | 0.851    |
| others  | 6      | 6         | 0.969 | 1.000 | 0.995 | 0.706    |

### YOLO26s + Shape-IoU（640 分辨率，batch=64，scale=1.0）

| 类别    | Images | Instances | P     | R     | mAP50 | mAP50-95 |
| ------- | ------ | --------- | ----- | ----- | ----- | -------- |
| **all** | 261    | 3938      | 0.944 | 0.983 | 0.986 | 0.829    |
| rice    | 261    | 2833      | 0.993 | 0.985 | 0.994 | 0.938    |
| broken  | 259    | 1099      | 0.994 | 0.964 | 0.993 | 0.863    |
| others  | 6      | 6         | 0.845 | 1.000 | 0.972 | 0.687    |

> Shape-IoU 在 mAP50-95 上提升 +0.2%，其中 rice 和 broken 的 mAP50-95 均有提升。

## 项目结构

```
YOLO26-BrokenRice/
├── train_broken_rice.py        # 训练脚本
├── predict_broken_rice.py      # 推理与可视化脚本
├── prepare_broken_rice.py      # 数据集重组脚本
├── data_broken_rice.yaml       # 数据集配置
├── My_README.md                # 开发笔记
├── ultralytics/                # YOLO26 源码（含 Shape-IoU 修改）
│   ├── cfg/
│   │   ├── default.yaml        # 新增 box_loss_type / shape_iou_scale
│   │   ├── models/26/          # YOLO26 模型定义
│   │   └── datasets/           # 数据集模板
│   └── utils/
│       ├── metrics.py          # 新增 shape_iou() 函数
│       ├── loss.py             # BboxLoss 支持 box_loss_type 切换
│       └── tal.py              # TaskAlignedAssigner 支持 box_loss_type 切换
└── pyproject.toml              # 项目配置
```

## 核心修改

基于 Ultralytics YOLO26 的改动仅涉及 4 个文件，改动量小且向后兼容：

| 文件                           | 修改内容                                                                         |
| ------------------------------ | -------------------------------------------------------------------------------- |
| `ultralytics/utils/metrics.py` | 新增 `shape_iou()` 函数，原 `bbox_iou()` 保留不变                                |
| `ultralytics/utils/loss.py`    | `BboxLoss` 新增 `box_loss_type` / `shape_iou_scale` 参数 + `_compute_iou()` 方法 |
| `ultralytics/utils/tal.py`     | `TaskAlignedAssigner` 新增 `box_loss_type` / `shape_iou_scale` 参数              |
| `ultralytics/cfg/default.yaml` | 新增 `box_loss_type: ciou` / `shape_iou_scale: 0.0` 配置项                       |

## 致谢

- [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) - 基础框架
- [Shape-IoU](https://arxiv.org/abs/2312.17663) - 形状感知 IoU 损失函数
- [broken-rice-detection](https://github.com/Yangr116/broken-rice-detection) - 碎米检测数据集

## 许可证

本项目基于 [AGPL-3.0](https://www.ultralytics.com/license) 许可证，继承自 Ultralytics。
