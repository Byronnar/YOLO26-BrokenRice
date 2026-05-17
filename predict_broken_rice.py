"""
推理与可视化脚本 - YOLO26 碎米检测
====================================
在图像上绘制检测框并保存结果。

用法:
    python predict_broken_rice.py
    python predict_broken_rice.py --source path/to/image.jpg
    python predict_broken_rice.py --source path/to/folder --conf 0.3
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# ────────────────────────────────────────────
# 默认路径
# ────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR.parent / "outputs" / "broken_rice_yolo26" / "yolo26s_img640" / "weights" / "best.pt"
DEFAULT_SOURCE = SCRIPT_DIR.parent / "datasets" / "broken-rice-yolo" / "images" / "val"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "broken_rice_vis"

# 类别颜色 (BGR)
COLORS = {
    "rice":   (0, 255, 0),     # 绿色
    "broken": (0, 165, 255),   # 橙色
    "others": (0, 0, 255),     # 红色
}


def find_model() -> Path:
    """自动查找最新训练的 best.pt。"""
    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL
    # 在 outputs/broken_rice_yolo26 下搜索
    search_dir = SCRIPT_DIR.parent / "outputs" / "broken_rice_yolo26"
    if search_dir.exists():
        models = sorted(search_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if models:
            return models[0]
    raise FileNotFoundError(f"找不到训练好的模型，请用 --model 指定路径")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 碎米检测推理与可视化")
    parser.add_argument("--model", type=str, default=None, help="模型路径 (默认自动查找)")
    parser.add_argument("--source", type=str, default=None, help="输入图像/目录路径")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    parser.add_argument("--max_det", type=int, default=300, help="最大检测数")
    parser.add_argument("--save_dir", type=str, default=None, help="保存目录")
    parser.add_argument("--num", type=int, default=20, help="最多可视化几张图 (0=全部)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 模型路径
    model_path = Path(args.model) if args.model else find_model()
    print(f"模型: {model_path}")

    # 输入路径
    source = Path(args.source) if args.source else DEFAULT_SOURCE
    print(f"输入: {source}")

    # 保存目录
    save_dir = Path(args.save_dir) if args.save_dir else OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model = YOLO(str(model_path))

    # 推理
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save=False,
        verbose=False,
    )

    # 可视化
    names = model.names  # {0: 'rice', 1: 'broken', 2: 'others'}
    count = 0
    limit = args.num if args.num > 0 else len(results)

    for r in results:
        if count >= limit:
            break

        # 获取带标注的图像 (BGR numpy array)
        annotated = r.plot(
            conf=True,
            line_width=2,
            font_size=12,
            labels=True,
        )

        # 保存
        img_path = Path(r.path)
        save_path = save_dir / f"vis_{img_path.stem}.jpg"
        import cv2
        cv2.imwrite(str(save_path), annotated)

        # 打印统计
        n_rice = sum(1 for c in r.boxes.cls if int(c) == 0)
        n_broken = sum(1 for c in r.boxes.cls if int(c) == 1)
        n_others = sum(1 for c in r.boxes.cls if int(c) == 2)
        print(f"  {img_path.name}: rice={n_rice}, broken={n_broken}, others={n_others}")

        count += 1

    print(f"\n可视化结果已保存到: {save_dir}")
    print(f"共 {count} 张图像")


if __name__ == "__main__":
    main()
