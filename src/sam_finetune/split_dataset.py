"""
将 Apple Disease Dataset 按 7:2:1 分层原地划分为 train/val/test。

操作前:
    data_root/
    ├── images/
    └── labels/

操作后:
    data_root/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

用法:
    python src/sam_finetune/split_dataset.py --data_root data/sam_seg/apple_disease
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def get_class_name(filename):
    """从文件名提取类别。文件格式: ClassName_1234.jpg"""
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def main():
    parser = argparse.ArgumentParser(description="原地划分 Apple Disease Dataset")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据集根目录（含 images/ 和 labels/ 子目录）")
    parser.add_argument("--ratio", type=int, nargs=3, default=[7, 2, 1],
                        help="Train:Val:Test 比例 (默认: 7 2 1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"images/ 目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels/ 目录不存在: {labels_dir}")

    # 收集所有 image-label 对，按类别分组
    class_pairs = defaultdict(list)
    skipped = 0

    for img_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{img_path.stem}.png"
        if not label_path.exists():
            skipped += 1
            continue
        cls = get_class_name(img_path.name)
        class_pairs[cls].append((img_path, label_path))

    if skipped:
        print(f"Warning: {skipped} 张图像无对应 label，已跳过")

    total = sum(len(v) for v in class_pairs.values())
    print(f"有效样本总数: {total}")
    print(f"类别数: {len(class_pairs)}")
    for cls, pairs in sorted(class_pairs.items()):
        print(f"  {cls}: {len(pairs)}")

    # 计算划分比例
    ratio_sum = sum(args.ratio)
    train_r, val_r, test_r = args.ratio[0] / ratio_sum, args.ratio[1] / ratio_sum, args.ratio[2] / ratio_sum

    # 分层划分
    splits = {"train": [], "val": [], "test": []}
    random.seed(args.seed)

    for cls, pairs in class_pairs.items():
        random.shuffle(pairs)
        n = len(pairs)
        n_train = max(1, round(n * train_r))
        n_val = max(1, round(n * val_r))

        splits["train"].extend(pairs[:n_train])
        splits["val"].extend(pairs[n_train:n_train + n_val])
        splits["test"].extend(pairs[n_train + n_val:])

    random.shuffle(splits["train"])
    random.shuffle(splits["val"])
    random.shuffle(splits["test"])

    # 创建目标目录结构
    for split_name in ["train", "val", "test"]:
        (data_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (data_root / split_name / "labels").mkdir(parents=True, exist_ok=True)

    # 移动文件
    for split_name, pairs in splits.items():
        print(f"\n{split_name}: {len(pairs)} 对")
        for img_path, label_path in pairs:
            dst_img = data_root / split_name / "images" / img_path.name
            dst_label = data_root / split_name / "labels" / label_path.name
            shutil.move(str(img_path), str(dst_img))
            shutil.move(str(label_path), str(dst_label))

    # 删除已空的原始目录
    images_dir.rmdir()
    labels_dir.rmdir()

    print(f"\n完成。已原地划分至 {data_root}/")
    print(f"  train: {len(splits['train'])} 张")
    print(f"  val:   {len(splits['val'])} 张")
    print(f"  test:  {len(splits['test'])} 张")


if __name__ == "__main__":
    main()
