"""
训练脚本 - YOLO26 碎米检测
===========================
基于 Ultralytics YOLO26，适配 broken-rice-detection 数据集。

用法:
    python train_broken_rice.py
    python train_broken_rice.py --model s --imgsz 640
    python train_broken_rice.py --model m --epochs 200
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# ────────────────────────────────────────────
# 路径
# ────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_YAML = SCRIPT_DIR / "data_broken_rice.yaml"
PROJECT_DIR = SCRIPT_DIR.parent / "outputs" / "broken_rice_yolo26"

# ────────────────────────────────────────────
# 模型规模映射
# ────────────────────────────────────────────
MODEL_SCALES = {
    "n": "yolo26n.pt",
    "s": "yolo26s.pt",
    "m": "yolo26m.pt",
    "l": "yolo26l.pt",
    "x": "yolo26x.pt",
}

# P2 检测头模型（增加小目标检测层，适合碎米场景）
MODEL_SCALES_P2 = {
    "n": "yolo26n-p2.yaml",
    "s": "yolo26s-p2.yaml",
    "m": "yolo26m-p2.yaml",
    "l": "yolo26l-p2.yaml",
    "x": "yolo26x-p2.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 碎米检测训练")
    parser.add_argument("--model", type=str, default="s", choices=MODEL_SCALES.keys(),
                        help="模型规模 (n/s/m/l/x)")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="输入图像尺寸 (推荐 960 或 640)")
    parser.add_argument("--epochs", type=int, default=150,
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=4,
                        help="批大小 (960 尺寸下建议 4)")
    parser.add_argument("--p2", action="store_true",
                        help="使用 P2 检测头（增加小目标检测层）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 恢复训练")
    parser.add_argument("--device", type=str, default="0",
                        help="训练设备 (0/1/cpu)")
    parser.add_argument("--box_loss_type", type=str, default="ciou",
                        choices=["ciou", "shape_iou", "diou", "giou"],
                        help="IoU 损失类型: ciou(默认), shape_iou, diou, giou")
    parser.add_argument("--shape_iou_scale", type=float, default=0.0,
                        help="Shape-IoU 的 scale 因子 (仅 box_loss_type=shape_iou 时生效, 推荐 0~3)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 确保数据集已准备
    if not DATA_YAML.exists():
        print(f"错误：数据配置文件不存在 {DATA_YAML}")
        print("请先运行: python prepare_broken_rice.py")
        return

    # 选择模型
    if args.p2:
        model_name = MODEL_SCALES_P2[args.model]
        print(f"使用 P2 检测头模型: {model_name}")
        # P2 模型从 yaml 构建，加载 COCO 预训练需要特殊处理
        model = YOLO(model_name)
    else:
        model_name = MODEL_SCALES[args.model]
        print(f"使用模型: {model_name}")
        model = YOLO(model_name)

    # 恢复训练
    if args.resume:
        model = YOLO(args.resume)

    # 训练参数
    train_kwargs = dict(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(PROJECT_DIR),
        name=f"yolo26{args.model}{'_p2' if args.p2 else ''}_img{args.imgsz}_{args.box_loss_type}",
        exist_ok=True,

        # ── 碎米检测调优参数 ──────────────────────
        # 类别加权：others 类极度稀少 (0.2%)，启用逆频率加权
        cls_pw=1.0,

        # IoU 损失类型
        box_loss_type=args.box_loss_type,
        shape_iou_scale=args.shape_iou_scale,

        # 密集目标：增大最大检测数、降低 NMS IoU
        max_det=300,
        iou=0.5,

        # 优化器
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,

        # 数据增强（保持较强增强以应对小数据集）
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,

        # 其他
        patience=30,
        seed=42,
        deterministic=True,
        amp=True,
        val=True,
        plots=True,
        save=True,
        save_period=-1,
        workers=4,
        cache="ram",
    )

    print(f"\n{'='*60}")
    print(f"YOLO26 碎米检测训练")
    print(f"{'='*60}")
    print(f"  模型:     {model_name}")
    print(f"  图像尺寸: {args.imgsz}")
    print(f"  批大小:   {args.batch}")
    print(f"  轮数:     {args.epochs}")
    print(f"  设备:     {args.device}")
    print(f"  P2 检测头: {'是' if args.p2 else '否'}")
    print(f"  IoU Loss: {args.box_loss_type}" + (f" (scale={args.shape_iou_scale})" if args.box_loss_type == "shape_iou" else ""))
    print(f"  输出目录: {PROJECT_DIR}")
    print(f"{'='*60}\n")

    # 开始训练
    results = model.train(**train_kwargs)

    # 训练完成后验证
    print("\n训练完成，运行最终验证...")
    metrics = model.val(data=str(DATA_YAML), imgsz=args.imgsz, batch=args.batch)
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # 各类别 mAP
    names = ["rice", "broken", "others"]
    for i, name in enumerate(names):
        if i < len(metrics.box.maps):
            print(f"  {name}: mAP50={metrics.box.maps[i]:.4f}")


if __name__ == "__main__":
    main()
