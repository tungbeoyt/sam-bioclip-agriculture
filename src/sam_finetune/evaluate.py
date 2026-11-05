"""
Evaluate SAM LoRA fine-tuning: compare zero-shot SAM vs LoRA-finetuned SAM.
Outputs IoU/Dice metrics and visualization figures.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor
from peft import PeftModel

from dataset import SAMSegDataset


@torch.no_grad()
def evaluate_model(model, dataset, device, model_name="SAM"):
    """Evaluate segmentation IoU/Dice on a dataset."""
    model.eval()
    ious, dices = [], []
    predictor = SamPredictor(model)

    for idx in tqdm(range(len(dataset)), desc=f"Evaluating {model_name}"):
        sample = dataset[idx]
        image = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_mask = sample["mask"].squeeze(0).numpy()

        predictor.set_image(image)

        # Use center point as prompt
        h, w = gt_mask.shape
        if gt_mask.sum() > 0:
            ys, xs = np.where(gt_mask > 0.5)
            cx, cy = int(xs.mean()), int(ys.mean())
            input_point = np.array([[cx, cy]])
            input_label = np.array([1])
        else:
            input_point = np.array([[w // 2, h // 2]])
            input_label = np.array([0])

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        best_idx = scores.argmax()
        pred_mask = masks[best_idx].astype(np.float32)

        intersection = (pred_mask * gt_mask).sum()
        union = (pred_mask + gt_mask).clip(0, 1).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        dice = (2 * intersection + 1e-6) / (pred_mask.sum() + gt_mask.sum() + 1e-6)

        ious.append(iou)
        dices.append(dice)

    return np.mean(ious), np.mean(dices), ious, dices


def visualize_comparison(dataset, base_model, lora_model, device, output_dir, num_samples=8):
    """Generate side-by-side comparison of base SAM vs LoRA-SAM."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    predictor_base = SamPredictor(base_model)
    predictor_lora = SamPredictor(lora_model)

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        image = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_mask = sample["mask"].squeeze(0).numpy()

        h, w = gt_mask.shape
        if gt_mask.sum() > 0:
            ys, xs = np.where(gt_mask > 0.5)
            pt = np.array([[int(xs.mean()), int(ys.mean())]])
            label = np.array([1])
        else:
            pt = np.array([[w // 2, h // 2]])
            label = np.array([0])

        # Ground truth
        axes[row, 0].imshow(image)
        axes[row, 0].imshow(gt_mask, alpha=0.4, cmap="Reds")
        axes[row, 0].set_title("Ground Truth")
        axes[row, 0].axis("off")

        # Base SAM
        predictor_base.set_image(image)
        masks_b, scores_b, _ = predictor_base.predict(
            point_coords=pt, point_labels=label, multimask_output=True
        )
        axes[row, 1].imshow(image)
        axes[row, 1].imshow(masks_b[scores_b.argmax()], alpha=0.4, cmap="Blues")
        axes[row, 1].scatter(*pt[0], c="red", s=30)
        axes[row, 1].set_title("Base SAM")
        axes[row, 1].axis("off")

        # LoRA SAM
        predictor_lora.set_image(image)
        masks_l, scores_l, _ = predictor_lora.predict(
            point_coords=pt, point_labels=label, multimask_output=True
        )
        axes[row, 2].imshow(image)
        axes[row, 2].imshow(masks_l[scores_l.argmax()], alpha=0.4, cmap="Greens")
        axes[row, 2].scatter(*pt[0], c="red", s=30)
        axes[row, 2].set_title("LoRA-SAM")
        axes[row, 2].axis("off")

        # Overlay comparison
        axes[row, 3].imshow(image)
        axes[row, 3].imshow(gt_mask, alpha=0.3, cmap="Reds")
        axes[row, 3].imshow(masks_l[scores_l.argmax()], alpha=0.3, cmap="Greens")
        axes[row, 3].set_title("GT (red) + LoRA (green)")
        axes[row, 3].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "sam_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison figure saved to {output_dir / 'sam_comparison.png'}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base SAM
    print(f"Loading base SAM ({args.model_type})...")
    base_sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    base_sam.to(device)
    base_sam.eval()

    # Load a SEPARATE SAM instance for LoRA to avoid aliasing
    print(f"Loading LoRA adapter from {args.lora_path}...")
    lora_sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    lora_sam = PeftModel.from_pretrained(lora_sam, args.lora_path)
    lora_sam = lora_sam.merge_and_unload()
    lora_sam.to(device)
    lora_sam.eval()

    # Test dataset
    test_dataset = SAMSegDataset(
        data_root=args.data_root,
        format=args.data_format,
        ann_file=args.ann_file,
        image_size=1024,
        mode="val",
        split_ratio=args.split_ratio,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Evaluate base SAM
    base_iou, base_dice, base_ious, base_dices = evaluate_model(
        base_sam, test_dataset, device, "Base SAM"
    )
    print(f"Base SAM  - IoU: {base_iou:.4f}, Dice: {base_dice:.4f}")

    # Evaluate LoRA SAM
    lora_iou, lora_dice, lora_ious, lora_dices = evaluate_model(
        lora_sam, test_dataset, device, "LoRA SAM"
    )
    print(f"LoRA SAM  - IoU: {lora_iou:.4f}, Dice: {lora_dice:.4f}")

    delta_iou = lora_iou - base_iou
    delta_dice = lora_dice - base_dice
    print(f"Improvement: +{delta_iou:.4f} IoU, +{delta_dice:.4f} Dice")

    # Save results
    results = {
        "base_sam": {"iou": float(base_iou), "dice": float(base_dice)},
        "lora_sam": {"iou": float(lora_iou), "dice": float(lora_dice)},
        "delta": {"iou": float(delta_iou), "dice": float(delta_dice)},
    }

    import json
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate visualization
    visualize_comparison(test_dataset, base_sam, lora_sam, device, output_dir)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--data_format", type=str, default="coco")
    parser.add_argument("--ann_file", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--split_ratio", type=float, default=1.0,
                        help="Use 1.0 for pre-split data, <1.0 for internal split")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()
    main(args)
