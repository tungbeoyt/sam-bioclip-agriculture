"""
LoRA fine-tuning for SAM on agricultural disease segmentation.
Uses peft to inject LoRA into SAM's image encoder and mask decoder.
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from segment_anything import sam_model_registry
from peft import LoraConfig, get_peft_model, TaskType

from dataset import SAMSegDataset


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2.0 * intersection + smooth) / (union + smooth)).mean()


def combined_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def train_epoch(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)

        with torch.amp.autocast("cuda"):
            # SAM forward: image_embedding + prompt_embedding -> mask predictions
            image_embeddings = model.image_encoder(images)

            # Process each sample individually for prompt encoding
            losses = []
            for i in range(images.shape[0]):
                # point_coords need to be (1, N, 2), point_labels (1, N)
                pc = point_coords[i].unsqueeze(0)  # (1, N, 2)
                pl = point_labels[i].unsqueeze(0)  # (1, N)
                ie = image_embeddings[i].unsqueeze(0)  # (1, C, H, W)

                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=(pc, pl), boxes=None, masks=None
                )

                mask_preds, _ = model.mask_decoder(
                    image_embeddings=ie,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                # SAM mask_decoder outputs 256x256, resize target to match
                gt = F.interpolate(masks[i].unsqueeze(0), size=(256, 256), mode="nearest")
                loss = combined_loss(mask_preds, gt)
                losses.append(loss)

            loss = torch.stack(losses).mean()
            loss = loss / args.gradient_accumulation

        scaler.scale(loss).backward()

        if (step + 1) % args.gradient_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * args.gradient_accumulation

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    count = 0

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        point_coords = batch["point_coords"].to(device)
        point_labels = batch["point_labels"].to(device)

        image_embeddings = model.image_encoder(images)

        for i in range(images.shape[0]):
            pc = point_coords[i].unsqueeze(0)
            pl = point_labels[i].unsqueeze(0)
            ie = image_embeddings[i].unsqueeze(0)

            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(pc, pl), boxes=None, masks=None
            )

            mask_preds, _ = model.mask_decoder(
                image_embeddings=ie,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred = (torch.sigmoid(mask_preds) > 0.5).float()
            gt = F.interpolate(masks[i].unsqueeze(0), size=(256, 256), mode="nearest")

            intersection = (pred * gt).sum()
            union = (pred + gt).clamp(0, 1).sum()
            iou = (intersection + 1e-6) / (union + 1e-6)
            dice = (2 * intersection + 1e-6) / (pred.sum() + gt.sum() + 1e-6)

            total_iou += iou.item()
            total_dice += dice.item()
            count += 1

    return total_iou / count, total_dice / count


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SAM model
    print(f"Loading SAM {args.model_type}...")
    sam = sam_model_registry[args.model_type](
        checkpoint=args.checkpoint if args.checkpoint else None
    )
    sam.to(device)

    # Configure LoRA
    # Target attention projection layers in image_encoder and mask_decoder
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["qkv", "proj", "q_proj", "v_proj", "lin1", "lin2"],
        modules_to_save=[],
    )

    model = get_peft_model(sam, lora_config)
    model.print_trainable_parameters()

    # Datasets
    train_dataset = SAMSegDataset(
        data_root=args.data_root,
        format=args.data_format,
        ann_file=args.ann_file,
        image_size=1024,
        mode="train",
        split_ratio=1.0 if args.val_data_root else 0.85,
    )
    val_root = args.val_data_root if args.val_data_root else args.data_root
    val_dataset = SAMSegDataset(
        data_root=val_root,
        format=args.data_format,
        ann_file=args.ann_file,
        image_size=1024,
        mode="val",
        split_ratio=1.0 if args.val_data_root else 0.85,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    best_iou = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        iou, dice = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Val IoU: {iou:.4f} | Val Dice: {dice:.4f}")

        if iou > best_iou:
            best_iou = iou
            model.save_pretrained(args.output_dir)
            print(f"  -> Best model saved (IoU: {iou:.4f})")

    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM LoRA Fine-tuning")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root path")
    parser.add_argument("--data_format", type=str, default="coco",
                        choices=["coco", "mask_dir", "cls_dir"])
    parser.add_argument("--ann_file", type=str, default=None, help="COCO annotation file")
    parser.add_argument("--model_type", type=str, default="vit_b",
                        choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--checkpoint", type=str, default=None, help="SAM checkpoint path")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val_data_root", type=str, default=None,
                        help="Validation data root (default: same as data_root)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints/sam_lora")
    args = parser.parse_args()
    main(args)
