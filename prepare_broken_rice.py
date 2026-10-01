"""
数据集重组脚本 - 将 broken-rice-detection 转为 YOLO 标准格式.
==============================================================
将原始目录结构：
    broken-rice-detection-main/
    ├── Images/          (2435 张 JPG)
    ├── txt_labels/      (2435 个 YOLO TXT)
    └── val              (验证集列表)

转为 YOLO 标准格式（符号链接，不复制文件）：
    broken-rice-yolo/
    ├── images/
    │   ├── train/       (2172 张)
    │   └── val/         (263 张)
    └── labels/
        ├── train/       (2172 个 txt)
        └── val/         (263 个 txt)

用法:
    python prepare_broken_rice.py
    python prepare_broken_rice.py --copy   # 使用复制而非符号链接
"""

import argparse
import os
import shutil
from pathlib import Path

# ────────────────────────────────────────────
# 路径配置
# ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "datasets" / "broken-rice-detection-main"
DST_DIR = BASE_DIR / "datasets" / "broken-rice-yolo"

IMAGES_DIR = SRC_DIR / "Images"
LABELS_DIR = SRC_DIR / "txt_labels"
VAL_LIST = SRC_DIR / "val"


def parse_args():
    parser = argparse.ArgumentParser(description="重组 broken-rice 数据集为 YOLO 格式")
    parser.add_argument("--copy", action="store_true", help="使用复制而非符号链接")
    parser.add_argument("--force", action="store_true", help="强制重建（删除已有目标目录）")
    return parser.parse_args()


def read_val_list(val_file: Path) -> set:
    """读取验证集文件列表，返回文件名（不含扩展名）集合。."""
    names = set()
    with open(val_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            # 去掉 .txt 后缀，得到文件基名
            name = line.replace(".txt", "")
            names.add(name)
    return names


def link_or_copy(src: Path, dst: Path, use_copy: bool):
    """创建符号链接或复制文件。."""
    if use_copy:
        shutil.copy2(src, dst)
    else:
        # Windows 需要管理员权限创建符号链接，尝试使用 junction（目录）或硬链接（文件）
        try:
            os.symlink(src, dst)
        except OSError:
            # fallback 到硬链接
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def main():
    args = parse_args()

    # 检查源目录
    if not IMAGES_DIR.exists():
        print(f"错误：图像目录不存在 {IMAGES_DIR}")
        return
    if not LABELS_DIR.exists():
        print(f"错误：标签目录不存在 {LABELS_DIR}")
        return
    if not VAL_LIST.exists():
        print(f"错误：验证集列表不存在 {VAL_LIST}")
        return

    # 读取验证集列表
    val_names = read_val_list(VAL_LIST)
    print(f"验证集：{len(val_names)} 张")

    # 创建目标目录
    for split in ("train", "val"):
        (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 遍历所有图像
    image_files = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"图像总数：{len(image_files)}")

    train_count = 0
    val_count = 0
    skip_count = 0

    for img_path in image_files:
        stem = img_path.stem
        label_path = LABELS_DIR / f"{stem}.txt"

        # 检查标签文件是否存在
        if not label_path.exists():
            skip_count += 1
            continue

        split = "val" if stem in val_names else "train"
        dst_img = DST_DIR / "images" / split / img_path.name
        dst_lbl = DST_DIR / "labels" / split / label_path.name

        # 跳过已存在的文件
        if dst_img.exists() and dst_lbl.exists():
            if split == "val":
                val_count += 1
            else:
                train_count += 1
            continue

        link_or_copy(img_path, dst_img, args.copy)
        link_or_copy(label_path, dst_lbl, args.copy)

        if split == "val":
            val_count += 1
        else:
            train_count += 1

    print("\n完成！")
    print(f"  训练集：{train_count} 张")
    print(f"  验证集：{val_count} 张")
    print(f"  跳过（缺少标签）：{skip_count} 张")
    print(f"\n输出目录：{DST_DIR}")
    print(f"数据配置文件：{Path(__file__).resolve().parent / 'data_broken_rice.yaml'}")


if __name__ == "__main__":
    main()
