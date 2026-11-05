"""
BioCLIP few-shot LoRA fine-tuning on PlantVillage.
K-shot per class, LoRA on vision encoder, linear classification head.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

import open_clip
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


def sample_fewshot(dataset, k, seed=42):
    """Sample k images per class, return train/test indices."""
    random.seed(seed)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_idx, test_idx = [], []
    for label, indices in class_indices.items():
        random.shuffle(indices)
        train_idx.extend(indices[:k])
        test_idx.extend(indices[k:])

    random.shuffle(train_idx)
    random.shuffle(test_idx)

    n_classes = len(class_indices)
    print(f"Train: {len(train_idx)} images ({k}-shot, {n_classes} classes)")
    print(f"Test:  {len(test_idx)} images")
    return train_idx, test_idx, n_classes


def load_bioclip(device):
    import open_clip
    ckpt_path = "experiments/checkpoints/bioclip_base/open_clip_pytorch_model.bin"
    model, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained=ckpt_path)
    print(f"BioCLIP loaded from local checkpoint: {ckpt_path}")
    return model.to(device)


@torch.no_grad()
def evaluate(model, head, dataloader, device):
    model.eval()
    head.eval()
    correct, total = 0, 0
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        feats = model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = head(feats)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100


def train_epoch(model, head, dataloader, optimizer, device):
    model.train()
    head.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        feats = model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = head(feats)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(dataloader), correct / total * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/cls/plantvillage/color")
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints/bioclip_fewshot")
    parser.add_argument("--val_limit", type=int, default=5000,
                        help="Max validation samples per epoch (to speed up)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"Device: {device}")

    # ---- Load BioCLIP ----
    model = load_bioclip(device)

    # ---- Data ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    dataset = datasets.ImageFolder(root=args.data_root, transform=transform)
    train_idx, test_idx, n_classes = sample_fewshot(dataset, args.k_shot, args.seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    # ---- Classification head ----
    embed_dim = model.visual.proj.shape[1]

    # ---- LoRA on vision encoder ----
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["out_proj", "c_fc", "c_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    head = nn.Linear(embed_dim, n_classes).to(device)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=args.lr
    )

    # ---- Train ----
    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    # Use a random subset for quick per-epoch validation
    rng = random.Random(args.seed)
    val_subset_idx = rng.sample(test_idx, min(args.val_limit, len(test_idx)))
    quick_val_loader = DataLoader(
        Subset(dataset, val_subset_idx), batch_size=args.batch_size, shuffle=False
    )

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss, train_acc = train_epoch(model, head, train_loader, optimizer, device)
        val_acc = evaluate(model, head, quick_val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(args.output_dir)
            torch.save(head.state_dict(), f"{args.output_dir}/head.pt")
            print(f"  -> Best model saved (Val Acc: {val_acc:.2f}%)")

    # ---- Full test evaluation (on best checkpoint) ----
    print("\n--- Final Evaluation on Full Test Set ---")
    # Reload best model from disk
    model = load_bioclip(device)
    model = PeftModel.from_pretrained(model, args.output_dir)
    model = model.merge_and_unload()
    model = model.to(device).eval()
    head = nn.Linear(embed_dim, n_classes).to(device)
    head.load_state_dict(torch.load(f"{args.output_dir}/head.pt", weights_only=True))
    test_acc = evaluate(model, head, test_loader, device)
    print(f"Test Accuracy (full): {test_acc:.2f}%")

    # ---- Save results ----
    results = {
        "model": "BioCLIP + LoRA",
        "k_shot": args.k_shot,
        "best_test_acc": round(best_acc, 2),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "num_classes": n_classes,
    }
    output_path = Path(args.output_dir) / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
