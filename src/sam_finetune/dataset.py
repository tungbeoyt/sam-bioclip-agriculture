"""
SAM fine-tuning dataset loader.
Supports COCO-format (PlantDoc), BRACOL mask format, and PlantVillage directory format.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class SAMSegDataset(Dataset):
    """通用分割数据集，支持 COCO JSON / 掩码目录 / 类别文件夹 三种格式."""

    def __init__(
        self,
        data_root: str,
        format: str = "coco",  # "coco" | "mask_dir" | "cls_dir"
        ann_file: str = None,
        image_size: int = 1024,
        mode: str = "train",
        split_ratio: float = 0.85,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.mode = mode
        self.samples = []

        if format == "coco":
            self._load_coco(ann_file)
        elif format == "mask_dir":
            self._load_mask_dir()
        elif format == "cls_dir":
            self._load_cls_dir()
        else:
            raise ValueError(f"Unknown format: {format}")

        random.seed(42)
        random.shuffle(self.samples)
        if split_ratio >= 1.0:
            pass  # use all data
        else:
            split_idx = int(len(self.samples) * split_ratio)
            if mode == "train":
                self.samples = self.samples[:split_idx]
            else:
                self.samples = self.samples[split_idx:]

    def _load_coco(self, ann_file):
        ann_path = Path(ann_file) if ann_file else self.data_root / "annotations.json"
        if not ann_path.exists():
            ann_paths = sorted(self.data_root.glob("*.json"))
            ann_path = ann_paths[0] if ann_paths else None
        if ann_path is None:
            raise FileNotFoundError(f"No COCO annotation found in {self.data_root}")

        self.coco = COCO(str(ann_path))
        self.img_ids = list(self.coco.imgs.keys())
        cat_names = {c["id"]: c["name"] for c in self.coco.loadCats(self.coco.getCatIds())}

        for img_id in self.img_ids[:]:
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
            mask_anns = [a for a in anns if a.get("segmentation") and isinstance(a["segmentation"], list)]
            if not mask_anns:
                continue
            img_info = self.coco.loadImgs([img_id])[0]
            self.samples.append({
                "img_path": str(self.data_root / img_info["file_name"]),
                "img_id": img_id,
                "anns": mask_anns,
                "label": cat_names.get(mask_anns[0]["category_id"], "unknown"),
            })

    def _load_mask_dir(self):
        img_dir = self.data_root / "images"
        mask_dir = self.data_root / "labels"
        if not mask_dir.exists():
            mask_dir = self.data_root / "masks"
        for mask_path in sorted(mask_dir.glob("*.png")):
            stem = mask_path.stem
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                candidate = img_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is not None:
                self.samples.append({"img_path": str(img_path), "mask_path": str(mask_path)})

    def _load_cls_dir(self):
        """PlantVillage 风格: class_name/xxx.jpg, 无真实掩码，用全图作为伪掩码."""
        for cls_dir in sorted(self.data_root.iterdir()):
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.glob("*")):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}:
                    self.samples.append({
                        "img_path": str(img_path),
                        "label": cls_dir.name,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        scale = self.image_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))

        # Pad to image_size x image_size
        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Build mask from annotations
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        if "mask_path" in sample:
            gt = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (new_w, new_h))
            gt = np.pad(gt, ((0, pad_h), (0, pad_w)), mode="constant")
            mask = (gt > 127).astype(np.float32)

        elif "anns" in sample:
            for ann in sample["anns"]:
                seg = ann["segmentation"]
                for poly in seg:
                    pts = np.array(poly).reshape(-1, 2).astype(np.float32)
                    pts[:, 0] = pts[:, 0] * scale
                    pts[:, 1] = pts[:, 1] * scale
                    pts = pts.astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1.0)

        elif "label" in sample:
            mask[:] = 1.0  # pseudo-mask: entire leaf

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        # Generate point prompts from mask (for training)
        if mask.sum() > 0 and self.mode == "train":
            coords = np.argwhere(mask > 0.5)
            if len(coords) > 0:
                pt = coords[random.randint(0, len(coords) - 1)]
                point_coords = torch.tensor([[pt[1], pt[0]]], dtype=torch.float32)
                point_labels = torch.tensor([1], dtype=torch.long)
            else:
                point_coords = torch.tensor([[self.image_size // 2, self.image_size // 2]], dtype=torch.float32)
                point_labels = torch.tensor([0], dtype=torch.long)
        else:
            point_coords = torch.tensor([[self.image_size // 2, self.image_size // 2]], dtype=torch.float32)
            point_labels = torch.tensor([0], dtype=torch.long)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "original_size": (h, w),
            "label": sample.get("label", ""),
        }
