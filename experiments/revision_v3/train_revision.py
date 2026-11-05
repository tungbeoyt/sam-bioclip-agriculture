"""Controlled experiment launcher around the retained full-data trainer.

The base trainer is shipped alongside this file. Run in the existing Ubuntu venv.
This launcher never starts from an old task checkpoint. Resume reuses saved settings.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path


def class_prompts(classes):
    names = {'Bacterial_spot': 'bacterial spot', 'Early_blight': 'early blight', 'Late_blight': 'late blight',
             'Leaf_Mold': 'leaf mold', 'Septoria_leaf_spot': 'Septoria leaf spot',
             'Spider_mites Two-spotted_spider_mite': 'two spotted spider mite damage', 'Target_Spot': 'target spot',
             'Tomato_Yellow_Leaf_Curl_Virus': 'tomato yellow leaf curl virus', 'Tomato_mosaic_virus': 'tomato mosaic virus',
             'healthy': 'healthy tissue'}
    return ['a photo of a Solanum lycopersicum tomato leaf with ' + names[c.removeprefix('Tomato___')] + '.' for c in classes]

def select_support(manifest, shots, seed):
    result = copy.deepcopy(manifest)
    if shots:
        chosen = []
        for label in range(len(result['classes'])):
            bucket = sorted((r for r in manifest['splits']['train'] if r['label'] == label), key=lambda r: r['path'])
            if len(bucket) < shots:
                raise ValueError(f'Class {label} has fewer than {shots} source training images')
            random.Random(seed * 1009 + label).shuffle(bucket)
            chosen.extend(bucket[:shots])
        result['splits']['train'] = sorted(chosen, key=lambda r: r['path'])
    held = {r['pixel_sha256'] for s in ('val', 'test') for r in result['splits'][s]}
    assert not held.intersection(r['pixel_sha256'] for r in result['splits']['train'])
    result['suite_support'] = dict(shots_per_class=shots or 'all', seed=seed,
                                  validation_labels=len(result['splits']['val']),
                                  policy='Nested support per seed; separate fixed source validation')
    result.pop('fingerprint', None)
    result['fingerprint'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def load_base(path):
    spec = importlib.util.spec_from_file_location('revision_base', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_model(base, method, rank, placement="mlp"):
    """Install one model family while retaining the same selection/evaluation loop."""
    base.LORA = dict(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=['c_fc', 'c_proj'])
    build_original = base.build_model
    forward_original = base.forward
    import torch
    import torch.nn.functional as F

    if method == 'linear':
        return
    if method in ('lora', 'full'):
        def build(mode):
            visual, head, train, evaluation = build_original('lora' if method == 'lora' and placement in ('mlp','both') else 'linear')
            if method == 'full':
                visual.requires_grad_(True)
            elif placement in ('attention','both'):
                from torch.nn.utils import parametrize
                class QKVLoRA(torch.nn.Module):
                    def __init__(self, weight):
                        super().__init__()
                        self.lora_A = torch.nn.Parameter(weight.new_empty(rank, weight.shape[1]))
                        self.lora_B = torch.nn.Parameter(weight.new_zeros(weight.shape[0], rank))
                        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**.5)
                    def forward(self, weight):
                        return weight + 2 * (self.lora_B @ self.lora_A)
                count=0
                for module in list(visual.modules()):
                    if isinstance(module, torch.nn.MultiheadAttention):
                        if module.in_proj_weight is None:
                            raise ValueError('Only fused QKV attention is supported')
                        parametrize.register_parametrization(module, 'in_proj_weight', QKVLoRA(module.in_proj_weight))
                        count+=1
                if count != 12: raise ValueError(f'Expected 12 attention blocks, found {count}')
            return visual, head, train, evaluation
        base.build_model=build
        if method == 'full':
            import inspect
            code=inspect.getsource(base.train_epoch).replace("if step == 0 and args.mode == 'lora':", 'if False:')
            exec(code, base.__dict__)
        return
    if method in ('resnet50', 'mobilenetv2'):
        import torchvision.models as models
        from torchvision import transforms as T
        from torchvision.transforms import InterpolationMode
        base.BASE_MODEL = f'torchvision:{method}:IMAGENET1K_V1'

        def build(_):
            if method == 'resnet50':
                visual = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                dim = visual.fc.in_features
                visual.fc = torch.nn.Identity()
            else:
                visual = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                dim = visual.last_channel
                visual.classifier = torch.nn.Identity()
            head = torch.nn.Linear(dim, 10)
            norm = T.Normalize([.485, .456, .406], [.229, .224, .225])
            train = T.Compose([T.RandomResizedCrop(224, scale=(.85, 1.), ratio=(.9, 1.1), interpolation=InterpolationMode.BICUBIC),
                               T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                               T.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
                               T.ColorJitter(brightness=.15, contrast=.15, saturation=.1, hue=.02), T.ToTensor(), norm])
            evaluation = T.Compose([T.Resize(224, interpolation=InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), norm])
            return visual.cuda(), head.cuda(), train, evaluation
        base.build_model = build
        original_state, original_load = base.weight_state, base.load_weights

        def state(visual, head, classes, epoch, mode):
            result = original_state(visual, head, classes, epoch, mode)
            result['visual_buffers'] = {n: v.detach().cpu().clone() for n, v in visual.named_buffers()}
            return result

        def load(visual, head, checkpoint, classes, mode):
            original_load(visual, head, checkpoint, classes, mode)
            expected = dict(visual.named_buffers())
            if set(checkpoint.get('visual_buffers', {})) != set(expected):
                raise ValueError('CNN checkpoint is missing BatchNorm buffers')
            with torch.no_grad():
                for n, v in checkpoint['visual_buffers'].items():expected[n].copy_(v)
        base.weight_state, base.load_weights = state, load
    else:
        # Both VLM variants share prompts, cosine classifier, loss and logit scale.
        # Only the text adapter is varied between vlm-vision and vlm-both.
        class VisionLanguage(torch.nn.Module):
            def __init__(self, model, tokens):
                super().__init__()
                self.model = model
                self.register_buffer('tokens', tokens, persistent=False)

            def forward(self, x):
                a = F.normalize(self.model.encode_image(x).float(), dim=-1)
                if method == 'vlm-vision':
                    self.model.transformer.eval()
                    with torch.no_grad():
                        b = F.normalize(self.model.encode_text(self.tokens).float(), dim=-1)
                else:
                    b = F.normalize(self.model.encode_text(self.tokens).float(), dim=-1)
                return self.model.logit_scale.exp().float() * a @ b.T

        def build(_):
            # Reuse the source transform definition without keeping a spare GPU model.
            import open_clip
            model, _, evaluation = open_clip.create_model_and_transforms(base.BASE_MODEL)
            model.requires_grad_(False)
            model.visual = base.inject_adapter_in_model(base.LoraConfig(**base.LORA, bias='none'), model.visual)
            if method == 'vlm-both':
                model.transformer = base.inject_adapter_in_model(base.LoraConfig(**base.LORA, bias='none'), model.transformer)
            classes = base.SUITE_CLASSES
            prompts = class_prompts(classes)
            tokens = open_clip.get_tokenizer(base.BASE_MODEL)(prompts)
            from torchvision import transforms as T
            from torchvision.transforms import InterpolationMode
            norm = next(t for t in evaluation.transforms if isinstance(t, T.Normalize))
            train = T.Compose([T.RandomResizedCrop(224, scale=(.85, 1.), ratio=(.9, 1.1), interpolation=InterpolationMode.BICUBIC),
                               T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
                               T.ColorJitter(brightness=.15, contrast=.15, saturation=.1, hue=.02), T.ToTensor(), norm])
            return VisionLanguage(model, tokens).cuda(), torch.nn.Identity().cuda(), train, evaluation
        base.build_model = build

        def forward(visual, head, x, amp):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp):
                logits = visual(x)
            # Matmul is autocast to BF16 even when its inputs were .float().
            # Weighted CE runs outside autocast with FP32 class weights.
            # Casting here keeps the differentiable logits/weights contract.
            return logits.float()
        base.forward = forward

    # CNNs are fully fine-tuned; the original LoRA gradient assertion does not apply.
    # Preserve all other optimizer, clipping, bookkeeping and finite-loss checks.
    import inspect
    code = inspect.getsource(base.train_epoch)
    if method in ('resnet50', 'mobilenetv2'):
        code = code.replace("visual.train(args.mode == 'lora')", 'visual.train(True)')
        code = code.replace("if step == 0 and args.mode == 'lora':", 'if False:')
        exec(code, base.__dict__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['linear', 'lora', 'full', 'resnet50', 'mobilenetv2', 'vlm-vision', 'vlm-both'], default='lora')
    p.add_argument('--rank', type=int, choices=[2, 4, 8, 16], default=8)
    p.add_argument('--placement', choices=['mlp','attention','both'], default='mlp')
    p.add_argument('--shots', type=int, choices=[0, 1, 2, 4, 8, 16], default=0)
    p.add_argument('--base-script', type=Path, default=Path(__file__).with_name('train_tomato_full_base.py'))
    opts, rest = p.parse_known_args()
    if any(x == '--init-checkpoint' or x.startswith('--init-checkpoint=') for x in rest):
        p.error('New comparisons must start from pretrained weights; do not pass --init-checkpoint')
    if any(x == '--mode' or x.startswith('--mode=') for x in rest):
        p.error('Use --method, not --mode')
    resumed = '--resume' in rest
    if resumed:
        last_path = Path(rest[rest.index('--resume') + 1])
        stored = json.loads((last_path.parent / 'suite.json').read_text())
        opts.method, opts.rank, opts.shots = stored['method'], stored['rank'], stored['shots']
        opts.placement = stored.get('placement','mlp')
    base = load_base(opts.base_script.resolve())
    configure_model(base, opts.method, opts.rank, opts.placement)
    original_audit = base.audit_and_split
    original_parse = base.parse_args
    original_export = base.export_results
    metadata = dict(placement=opts.placement, attention_dropout=0.0, method=opts.method, rank=opts.rank, alpha=2 * opts.rank,
                    shots=opts.shots, classifier='cosine-prompts' if opts.method.startswith('vlm') else 'L2-linear',
                    initialization='original pretrained weights', validation_policy='500 extra source labels',
                    launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    base_sha256=hashlib.sha256(opts.base_script.read_bytes()).hexdigest())

    def parse():
        args, state = original_parse()
        if resumed:
            old = json.loads((args.output_dir / 'suite.json').read_text())
            for k in ['launcher_sha256', 'base_sha256']:
                if old[k] != metadata[k]:
                    raise ValueError('Code changed since this run started; use matching files for resume')
        else:
            if (args.output_dir / 'suite.json').exists():
                raise FileExistsError('Use a fresh output directory or --resume')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        base.save_json(args.output_dir / 'suite.json', metadata)
        args.split_manifest = args.output_dir / 'split_manifest.json'
        return args, state

    def audit(args):
        full_args = copy.copy(args)
        full_args.split_manifest = args.output_dir / 'source_full_manifest.json'
        manifest = select_support(original_audit(full_args), opts.shots, args.seed)
        target = args.output_dir / 'split_manifest.json'
        if target.exists() and json.loads(target.read_text()) != manifest:
            raise ValueError('Support manifest changed')
        base.save_json(target, manifest)
        base.SUITE_CLASSES = manifest['classes']
        print('SUITE TRAIN COUNTS', dict(__import__('collections').Counter(r['class'] for r in manifest['splits']['train'])), flush=True)
        return manifest

    def export(out, logits, y, classes, rows, temperature, meta, smoke=False):
        meta = dict(meta, model=opts.method, experiment=metadata)
        return original_export(out, logits, y, classes, rows, temperature, meta, smoke)

    base.parse_args, base.audit_and_split, base.export_results = parse, audit, export
    sys.argv = [str(opts.base_script)] + rest + ['--mode', 'linear' if opts.method == 'linear' else 'lora']
    base.main()


if __name__ == '__main__':
    main()
