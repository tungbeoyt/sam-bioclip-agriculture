"""
CLIP ViT-B/16 zero-shot classification on PlantVillage.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def format_class_name(name):
    """Apple___Black_rot -> apple black rot"""
    name = name.replace("___", " ")
    name = name.replace("_", " ")
    name = " ".join(name.split())
    return name


@torch.no_grad()
def evaluate(model, tokenizer, dataloader, class_names, device):
    model.eval()

    prompts_per_class = []
    for name in class_names:
        readable = format_class_name(name)
        prompts_per_class.append([
            f"a photo of a leaf with {readable}",
            f"a {readable} infected plant leaf",
            f"a plant leaf showing symptoms of {readable}",
            f"一张患有{readable}的叶片照片",
        ])

    # Encode text features (average over templates per class)
    text_features = []
    for prompts in prompts_per_class:
        texts = tokenizer(prompts).to(device)
        feats = model.encode_text(texts)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        text_features.append(feats.mean(dim=0, keepdim=True))
    text_features = torch.cat(text_features, dim=0)

    top1, top5, total = 0, 0, 0
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        img_feats = model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        logits = 100.0 * img_feats @ text_features.T
        _, preds = logits.topk(min(5, logits.shape[-1]), dim=-1)

        top1 += (preds[:, 0] == labels).sum().item()
        top5 += (preds == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    return top1 / total * 100, top5 / total * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="data/cls/plantvillage/color")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    dataset = datasets.ImageFolder(root=args.data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Dataset: {len(dataset)} images, {len(dataset.classes)} classes")

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device)

    top1, top5 = evaluate(model, tokenizer, dataloader, dataset.classes, device)
    print(f"CLIP ViT-B-16  Top-1: {top1:.2f}%  Top-5: {top5:.2f}%")

    output_path = Path(args.output_dir) / "clip_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"model": "CLIP ViT-B-16", "top1": round(top1, 2), "top5": round(top5, 2),
                   "dataset": str(args.data_root), "num_classes": len(dataset.classes),
                   "num_images": len(dataset)}, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
