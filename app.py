"""
Gradio demo app for agricultural disease recognition.
Integrates SAM segmentation and BioCLIP classification.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import gradio as gr

PROJECT_ROOT = Path(__file__).parent.resolve()

# --- Model loading (pre-load at startup for responsiveness) ---

_seg_model = None
_cls_model = None
_cls_head = None

PLANTVILLAGE_CLASSES = [
    "Apple scab", "Apple black rot", "Apple cedar apple rust", "Apple healthy",
    "Blueberry healthy", "Cherry healthy", "Cherry powdery mildew",
    "Corn cercospora leaf spot", "Corn common rust", "Corn healthy", "Corn northern leaf blight",
    "Grape black rot", "Grape esca", "Grape healthy", "Grape leaf blight",
    "Orange huanglongbing", "Peach bacterial spot", "Peach healthy",
    "Pepper bell bacterial spot", "Pepper bell healthy",
    "Potato early blight", "Potato healthy", "Potato late blight",
    "Raspberry healthy", "Soybean healthy", "Squash powdery mildew",
    "Strawberry healthy", "Strawberry leaf scorch",
    "Tomato bacterial spot", "Tomato early blight", "Tomato healthy", "Tomato late blight",
    "Tomato leaf mold", "Tomato septoria leaf spot", "Tomato spider mites",
    "Tomato target spot", "Tomato tomato mosaic virus", "Tomato yellow leaf curl virus",
]


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_seg_model():
    """Load SAM model with LoRA adapter (pre-load on startup)."""
    global _seg_model
    if _seg_model is not None:
        return _seg_model

    device = get_device()
    from segment_anything import sam_model_registry, SamPredictor

    sam_ckpt = PROJECT_ROOT / "experiments/checkpoints/sam_vit_b_01ec64.pth"
    if not sam_ckpt.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt}")

    print(f"Loading SAM ViT-B from {sam_ckpt} ...")
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))
    sam.to(device)
    sam.eval()

    lora_path = PROJECT_ROOT / "experiments/checkpoints/sam_lora"
    if lora_path.exists():
        try:
            from peft import PeftModel
            sam = PeftModel.from_pretrained(sam, str(lora_path))
            sam = sam.merge_and_unload()
            sam.to(device)
            sam.eval()
            print("  -> LoRA adapter loaded and merged")
        except Exception as e:
            print(f"  -> LoRA load failed, using base SAM: {e}")

    _seg_model = SamPredictor(sam)
    print("SAM model ready.")
    return _seg_model


def load_cls_model():
    """Load BioCLIP + LoRA + classification head (pre-load on startup)."""
    global _cls_model, _cls_head
    if _cls_model is not None and _cls_head is not None:
        return _cls_model, _cls_head

    device = get_device()
    import open_clip
    import torch.nn as nn
    from peft import PeftModel

    ckpt_dir = PROJECT_ROOT / "experiments/checkpoints/bioclip_fewshot_20shot"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"BioCLIP checkpoint not found: {ckpt_dir}")

    print("Loading BioCLIP from local checkpoint ...")
    bioclip_ckpt = PROJECT_ROOT / "experiments/checkpoints/bioclip_base/open_clip_pytorch_model.bin"
    _cls_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained=str(bioclip_ckpt),
    )
    print("  -> base model loaded")

    print(f"Loading LoRA adapter from {ckpt_dir} ...")
    _cls_model = PeftModel.from_pretrained(_cls_model, str(ckpt_dir))
    _cls_model = _cls_model.merge_and_unload()
    print("  -> LoRA merged")

    _cls_model = _cls_model.to(device).eval()

    embed_dim = _cls_model.visual.proj.shape[1]
    _cls_head = nn.Linear(embed_dim, 38).to(device)

    head_path = ckpt_dir / "head.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"Classification head not found: {head_path}")
    _cls_head.load_state_dict(torch.load(str(head_path), map_location=device, weights_only=True))
    _cls_head.eval()

    print(f"BioCLIP 20-shot model ready (38 classes).")
    return _cls_model, _cls_head


# --- Segmentation tab ---

def _run_sam(img, input_point, input_label):
    try:
        predictor = load_seg_model()
    except Exception as e:
        return img, f"Error loading SAM model: {e}"

    h, w = img.shape[:2]
    predictor.set_image(img)

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    mask = masks[0].astype(np.uint8) * 255

    overlay = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

    info = f"Mask area: {mask.sum() / 255:.0f} pixels ({mask.sum() / 255 / (h * w) * 100:.1f}% of image)"
    return overlay, info


def segment_image_click(image, evt: gr.SelectData):
    if image is None:
        return None, "Please upload an image first."
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)
    x, y = evt.index[0], evt.index[1]
    input_point = np.array([[x, y]])
    return _run_sam(img, input_point, np.array([1]))


def segment_image_btn(image, prompt_mode):
    if image is None:
        return None, "Please upload an image first."
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)

    h, w = img.shape[:2]
    if prompt_mode == "center":
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])
    else:
        xs = np.linspace(w * 0.2, w * 0.8, 3, dtype=int)
        ys = np.linspace(h * 0.2, h * 0.8, 3, dtype=int)
        grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        input_point = grid
        input_label = np.ones(len(grid), dtype=int)

    return _run_sam(img, input_point, input_label)


# --- Classification tab ---

@torch.no_grad()
def classify_image(image):
    if image is None:
        return {}, "Please upload an image first."

    try:
        model, head = load_cls_model()
    except Exception as e:
        return {}, f"Error loading classification model: {e}"

    device = next(model.parameters()).device

    img = np.array(image) if not isinstance(image, np.ndarray) else image
    if img.shape[-1] == 4:
        img = img[..., :3]

    try:
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        features = model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        logits = head(features)
        probs = logits.softmax(dim=-1).squeeze(0)

        top5_indices = probs.topk(min(5, len(PLANTVILLAGE_CLASSES))).indices.tolist()
        top5_scores = probs[top5_indices].tolist()

        result = {
            PLANTVILLAGE_CLASSES[idx]: round(float(score), 4)
            for idx, score in zip(top5_indices, top5_scores)
        }

        output_lines = ["**Top-5 Predictions:**\n"]
        for i, (idx, score) in enumerate(zip(top5_indices, top5_scores)):
            output_lines.append(f"{i+1}. {PLANTVILLAGE_CLASSES[idx]} — {score:.2%}")

        return result, "\n".join(output_lines)
    except Exception as e:
        return {}, f"Classification failed: {e}"


# --- Build Gradio UI ---

def create_demo():
    theme = gr.themes.Soft()

    with gr.Blocks(theme=theme, title="农业病虫害识别系统") as demo:
        gr.Markdown("""
        # 农业病虫害识别系统
        ### 基于视觉大模型（SAM + BioCLIP）
        """)

        with gr.Tabs():
            with gr.TabItem("病害区域分割 (SAM)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seg_input = gr.Image(label="上传叶片图像（点击图像直接分割）", type="numpy")
                        seg_mode = gr.Radio(
                            choices=["center", "grid"],
                            value="center",
                            label="按钮模式（点击图像无需选择模式）",
                        )
                        seg_btn = gr.Button("开始分割（center/grid）", variant="secondary")

                    with gr.Column(scale=1):
                        seg_output = gr.Image(label="分割结果（红色轮廓=分割区域）")
                        seg_info = gr.Textbox(label="分割信息", lines=3)

            with gr.TabItem("病虫害分类 (BioCLIP)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        cls_input = gr.Image(label="上传叶片图像", type="numpy")
                        cls_btn = gr.Button("开始分类", variant="primary")

                    with gr.Column(scale=1):
                        cls_label = gr.Label(label="Top-5 预测结果", num_top_classes=5)
                        cls_text = gr.Markdown()

        # Event handlers
        seg_input.select(
            segment_image_click,
            inputs=[seg_input],
            outputs=[seg_output, seg_info],
        )
        seg_btn.click(
            segment_image_btn,
            inputs=[seg_input, seg_mode],
            outputs=[seg_output, seg_info],
        )
        cls_btn.click(
            classify_image,
            inputs=[cls_input],
            outputs=[cls_label, cls_text],
        )

    return demo


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    print("Pre-loading models (this may take a few minutes on first run)...")
    print("-" * 50)

    try:
        load_seg_model()
    except Exception as e:
        print(f"SAM model failed to load: {e}")

    try:
        load_cls_model()
    except Exception as e:
        print(f"BioCLIP model failed to load: {e}")

    print("-" * 50)
    print("All models loaded. Starting Gradio server...")
    print("Open http://localhost:7860 in your browser.")

    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
