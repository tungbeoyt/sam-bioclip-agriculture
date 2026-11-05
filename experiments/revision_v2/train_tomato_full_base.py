"""Stable full-data BioCLIP LoRA training; fixed evaluation sets and epoch resume.

Run --prepare-only to audit all tomato images without loading a model.
Run --smoke-steps 3 --epochs 2 to check the GPU pipeline WITHOUT evaluating test.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, inject_adapter_in_model
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, matthews_corrcoef, cohen_kappa_score)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

BASE_MODEL = 'hf-hub:imageomics/bioclip'
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.1,
            target_modules=['c_fc', 'c_proj'])
PATH_ARGS = ('data_root', 'output_dir', 'reference_manifest', 'split_manifest', 'init_checkpoint')


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def save_checkpoint(path, obj):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def audit_and_split(args):
    """Use every unique color image, preserving reference validation/test exactly."""
    reference = json.loads(args.reference_manifest.read_text())
    classes = sorted(p.name for p in args.data_root.iterdir()
                     if p.is_dir() and p.name.startswith('Tomato___'))
    if len(classes) != 10 or classes != reference['classes']:
        raise ValueError('Dataset classes/order differ from the reference experiment')
    records, by_hash, duplicates = [], {}, []
    for label, name in enumerate(classes):
        files = sorted(p for p in (args.data_root / name).rglob('*')
                       if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        for file in tqdm(files, desc=f'Audit {name}', unit='image', mininterval=1):
            with Image.open(file) as im:
                rgb = im.convert('RGB')
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            row = dict(path=str(file.relative_to(args.data_root)), label=label,
                       **{'class': name}, pixel_sha256=digest)
            records.append(row)
            if digest in by_hash:
                if by_hash[digest]['label'] != label:
                    raise ValueError(f'Identical pixels with conflicting labels: {file}')
                duplicates.append(row['path'])
            else:
                by_hash[digest] = row
    by_path = {r['path']: r for r in records}
    held = set()
    splits = {'train': [], 'val': [], 'test': []}
    for split in ('val', 'test'):
        for row in reference['splits'][split]:
            actual = by_path.get(row['path'])
            if actual != row:
                raise ValueError(f'Reference image changed/missing: {row["path"]}')
            if row['pixel_sha256'] in held:
                raise ValueError('Reference evaluation sets have duplicate pixels')
            held.add(row['pixel_sha256'])
            splits[split].append(row)
    splits['train'] = sorted((r for digest, r in by_hash.items() if digest not in held),
                             key=lambda r: r['path'])
    if set(r['label'] for r in splits['train']) != set(range(10)):
        raise ValueError('Every class must have training images')
    manifest = dict(version=1, data_root=str(args.data_root.resolve()), classes=classes,
                    splits=splits, reference_sha256=fingerprint(reference),
                    raw_images=len(records), unique_images=len(by_hash),
                    duplicate_files=duplicates,
                    policy='All unique images except the fixed reference validation/test sets')
    # Fingerprint includes paths and hashes, not file modification dates.
    manifest['fingerprint'] = fingerprint(manifest)
    if args.split_manifest.exists():
        previous = json.loads(args.split_manifest.read_text())
        if previous != manifest:
            raise ValueError('Fixed manifest changed. Use a new manifest path after reviewing data changes.')
    else:
        save_json(args.split_manifest, manifest)
    print('SPLITS:', {s: len(v) for s, v in splits.items()}, flush=True)
    for c in classes:
        print(c, {s: sum(r['class'] == c for r in v) for s, v in splits.items()}, flush=True)
    print(f'Raw={len(records)} unique={len(by_hash)} duplicate files={len(duplicates)}', flush=True)
    return manifest


class Images(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.root / row['path']) as im:
            x = self.transform(im.convert('RGB'))
        return x, row['label']


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(args, rows, transform, training):
    return DataLoader(Images(args.data_root, rows, transform), batch_size=args.batch_size,
                      shuffle=training, num_workers=args.workers, pin_memory=True,
                      worker_init_fn=seed_worker, persistent_workers=False,
                      generator=torch.Generator().manual_seed(args.seed),
                      **({'prefetch_factor': 2} if args.workers else {}))


def build_model(mode):
    model, _, eval_transform = open_clip.create_model_and_transforms(BASE_MODEL)
    visual = model.visual
    dim = visual.output_dim
    size = visual.image_size
    del model
    visual.requires_grad_(False)
    if mode == 'lora':
        visual = inject_adapter_in_model(LoraConfig(**LORA, bias='none'), visual)
    visual = visual.cuda()
    head = torch.nn.Linear(dim, 10).cuda()
    norm = next(t for t in eval_transform.transforms if isinstance(t, T.Normalize))
    train_transform = T.Compose([
        T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1),
                            interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=.15, contrast=.15, saturation=.1, hue=.02),
        T.ToTensor(), norm,
    ])
    return visual, head, train_transform, eval_transform


def load_weights(visual, head, checkpoint, classes, mode):
    if checkpoint['classes'] != classes or checkpoint['base_model'] != BASE_MODEL:
        raise ValueError('Checkpoint class order/base model does not match')
    expected = {n for n, p in visual.named_parameters() if p.requires_grad}
    if set(checkpoint['adapter']) != expected:
        raise ValueError('Checkpoint adapter keys do not match this mode/model')
    if mode == 'lora' and checkpoint['lora'] != LORA:
        raise ValueError('Checkpoint LoRA configuration differs')
    with torch.no_grad():
        params = dict(visual.named_parameters())
        for n, v in checkpoint['adapter'].items():
            params[n].copy_(v)
    head.load_state_dict(checkpoint['head'], strict=True)


def weight_state(visual, head, classes, epoch, mode):
    return dict(adapter={n: p.detach().cpu().clone() for n, p in visual.named_parameters()
                         if p.requires_grad},
                head={n: p.detach().cpu().clone() for n, p in head.state_dict().items()},
                classes=classes, epoch=epoch, base_model=BASE_MODEL, lora=LORA,
                mode=mode)


def lr_factor(step, total, warmup):
    if step < warmup:
        return max(1, step + 1) / max(1, warmup)
    progress = min(1., (step - warmup) / max(1, total - warmup))
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))


def get_rng():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def set_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def forward(visual, head, x, amp):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp):
        features = visual(x)
    # Keep normalization and the small head in float32.
    return head(F.normalize(features.float(), dim=-1))


def train_epoch(visual, head, loader, optimizer, scheduler, weights, args, epoch):
    visual.train(args.mode == 'lora')
    head.train()
    loader.generator.manual_seed(args.seed + epoch)
    total, correct, loss_sum, grad_sum, grad_max, steps = 0, 0, 0., 0., 0., 0
    parameters = [p for p in visual.parameters() if p.requires_grad] + list(head.parameters())
    bar = tqdm(loader, desc=f'Train {epoch}/{args.epochs}', unit='batch', mininterval=.5)
    for step, (x, y) in enumerate(bar):
        if args.smoke_steps and step >= args.smoke_steps:
            break
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = forward(visual, head, x, args.amp)
        loss = F.cross_entropy(logits, y, weight=weights, label_smoothing=args.label_smoothing)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss; previous epoch checkpoint is preserved')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if step == 0 and args.mode == 'lora':
            if not any(p.grad is not None and torch.count_nonzero(p.grad).item()
                       for n, p in visual.named_parameters() if 'lora_B' in n):
                raise RuntimeError('LoRA is not receiving nonzero gradients')
        optimizer.step()
        scheduler.step()
        n = len(y)
        total += n
        correct += (logits.argmax(1) == y).sum().item()
        loss_sum += loss.item() * n
        grad_sum += norm.item()
        grad_max = max(grad_max, norm.item())
        steps += 1
        if step % 10 == 0:
            bar.set_postfix(loss=f'{loss_sum/total:.4f}', acc=f'{correct/total:.1%}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.2g}')
    return dict(loss=loss_sum / total, accuracy=correct / total,
                grad_norm_mean=grad_sum / steps, grad_norm_max=grad_max)


@torch.no_grad()
def evaluate(visual, head, loader, amp, desc='Validation', limit=0):
    visual.eval()
    head.eval()
    logits, targets = [], []
    for step, (x, y) in enumerate(tqdm(loader, desc=desc, unit='batch', mininterval=1)):
        if limit and step >= limit:
            break
        logits.append(forward(visual, head, x.cuda(non_blocking=True), amp).float().cpu())
        targets.append(y)
    logits, targets = torch.cat(logits), torch.cat(targets)
    if not torch.isfinite(logits).all():
        raise RuntimeError('Non-finite evaluation logits')
    return dict(loss=F.cross_entropy(logits, targets).item(),
                accuracy=(logits.argmax(1) == targets).float().mean().item()), logits, targets


def fit_temperature(logits, y):
    # Validation-only bounded search includes exactly T=1 (never worsens val NLL).
    candidates = sorted(set([1.] + np.geomspace(.25, 8., 121).tolist()))
    return min(candidates, key=lambda t: F.cross_entropy(logits / t, y).item())


def probability_metrics(logits, y, temperature=1.):
    probs = (logits / temperature).softmax(1).numpy()
    target = y.numpy()
    onehot = np.eye(probs.shape[1])[target]
    confidence, pred = probs.max(1), probs.argmax(1)
    ece = 0.
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs((pred[mask] == target[mask]).mean() - confidence[mask].mean())
    return dict(nll=F.cross_entropy(logits / temperature, y).item(),
                brier_multiclass_sum=float(np.square(probs - onehot).sum(1).mean()),
                ece_15_equal_width_bins=float(ece))


def export_results(out, logits, y, classes, rows, temperature, meta, smoke=False):
    probs = logits.softmax(1).numpy()
    calibrated = (logits / temperature).softmax(1).numpy()
    target, pred = y.numpy(), probs.argmax(1)
    report = classification_report(target, pred, labels=list(range(10)), target_names=classes,
                                   zero_division=0, output_dict=True)
    matrix = confusion_matrix(target, pred, labels=list(range(10)))
    metrics = dict(meta, test_loss=F.cross_entropy(logits, y).item(),
                   test_accuracy=float((pred == target).mean()),
                   macro_precision=report['macro avg']['precision'],
                   macro_recall=report['macro avg']['recall'],
                   macro_f1=report['macro avg']['f1-score'],
                   weighted_f1=report['weighted avg']['f1-score'],
                   balanced_accuracy=report['macro avg']['recall'],
                   mcc=float(matthews_corrcoef(target, pred)),
                   kappa=float(cohen_kappa_score(target, pred)),
                   temperature=temperature, calibration_fit='validation only',
                   uncalibrated=probability_metrics(logits,y),
                   calibrated=probability_metrics(logits,y,temperature),
                   classification_report=report, confusion_matrix=matrix.tolist())
    if len(set(target)) == len(classes):
        metrics['macro_roc_auc_ovr'] = float(roc_auc_score(target,probs,multi_class='ovr',average='macro'))
        metrics['macro_average_precision_ovr'] = float(average_precision_score(np.eye(10)[target],probs,average='macro'))
    if smoke:
        metrics['WARNING'] = 'SMOKE ONLY: scores below use a validation subset, NOT held-out test.'
        for key in ('test_accuracy', 'test_loss'):
            metrics['smoke_validation_' + key.removeprefix('test_')] = metrics.pop(key)
    prefix = 'smoke_' if smoke else ''
    save_json(out / f'{prefix}metrics.json',metrics)
    np.savez_compressed(out/f'{prefix}logits.npz',logits=logits.numpy(),targets=target,
                        classes=np.array(classes),paths=np.array([r['path'] for r in rows]))
    with (out/f'{prefix}predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['path','true_class','predicted_class','confidence','calibrated_confidence']+
                        [f'p:{c}' for c in classes]+[f'p_calibrated:{c}' for c in classes])
        for row,t,guess,p,pc in zip(rows,target,pred,probs,calibrated):
            writer.writerow([row['path'],classes[t],classes[guess],float(p.max()),float(pc.max())]+p.tolist()+pc.tolist())
    (out/f'{prefix}classification_report.txt').write_text(classification_report(
        target,pred,labels=list(range(10)),target_names=classes,zero_division=0,digits=4),encoding='utf-8')
    short=[c.replace('Tomato___','').replace('Tomato_','') for c in classes]
    fig,ax=plt.subplots(figsize=(11,9))
    ax.imshow(matrix,cmap='Blues')
    ax.set(xticks=range(10),yticks=range(10),xticklabels=short,yticklabels=short,
           xlabel='Predicted',ylabel='True',title=f'{"SMOKE validation" if smoke else "Test"} (n={len(y)})')
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    fig.tight_layout();fig.savefig(out/f'{prefix}confusion_matrix.png',dpi=150);plt.close(fig)
    return metrics


def write_history(out, history, best_epoch):
    with (out/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]))
        writer.writeheader();writer.writerows(history)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,key in zip(axes[:2],['loss','accuracy']):
        for split in ['train','val']:
            ax.plot([r['epoch'] for r in history],[r[f'{split}_{key}'] for r in history],label=split)
        ax.axvline(best_epoch,color='gray',linestyle='--',label='best')
        ax.set(xlabel='Epoch',ylabel=key);ax.legend()
    axes[0].set_title('Train weighted/smoothed; val plain CE')
    for key in ['lr_adapter','lr_head']:
        axes[2].plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
    axes[2].set(xlabel='Epoch',ylabel='Learning rate');axes[2].legend()
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=140);plt.close(fig)


def compare_reference(args, rows, metrics):
    """Compare only if both prediction files refer to exactly the same test paths."""
    folder=args.reference_manifest.parent
    if not (folder/'predictions.csv').exists() or not (folder/'metrics.json').exists():
        return
    with (folder/'predictions.csv').open() as f:
        old=list(csv.DictReader(f))
    if {r['path'] for r in rows}!={r['path'] for r in old}:
        raise ValueError('Refusing comparison: test membership differs')
    before=json.loads((folder/'metrics.json').read_text())
    comparison={'same_test_paths':True,'test_images':len(rows),
                'note':'Historical benchmark already inspected; not a fresh field test.'}
    for key in ['test_accuracy','macro_precision','macro_recall','macro_f1','test_loss']:
        comparison[key]={'previous':before[key],'current':metrics[key],
                         'delta':metrics[key]-before[key]}
    save_json(args.output_dir/'comparison_to_7000.json',comparison)
    print(f'Same-test accuracy delta: {100*(metrics["test_accuracy"]-before["test_accuracy"]):+.2f} percentage points',flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('data/cls/plantvillage/color'))
    p.add_argument('--reference-manifest',type=Path,default=Path('experiments/results/tomato_7000_20260925_113448/split_manifest.json'))
    p.add_argument('--split-manifest',type=Path,default=Path('experiments/splits/tomato_full_v1.json'))
    p.add_argument('--output-dir',type=Path,default=Path('experiments/results/tomato_full'))
    p.add_argument('--init-checkpoint',type=Path)
    p.add_argument('--resume',type=Path,help='Resume a last.pt with its saved arguments; overrides other CLI settings')
    p.add_argument('--mode',choices=['lora','linear'],default='lora')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--adapter-lr',type=float,default=3e-5)
    p.add_argument('--head-lr',type=float,default=3e-4)
    p.add_argument('--warmup-epochs',type=float,default=1.)
    p.add_argument('--patience',type=int,default=7)
    p.add_argument('--min-delta',type=float,default=1e-4)
    p.add_argument('--class-weighting', choices=['sqrt','none'], default='sqrt')
    p.add_argument('--label-smoothing',type=float,default=.05)
    p.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--skip-test',action='store_true',help='Use for validation-only hyperparameter experiments')
    p.add_argument('--smoke-steps',type=int,default=0,help='Limit train/validation steps; NEVER evaluates test')
    args=p.parse_args()
    resume=None
    if args.resume:
        resume=torch.load(args.resume,weights_only=True,map_location='cpu')
        if resume.get('format')!='tomato-full-v1':
            p.error('--resume requires a last.pt from this script')
        args=argparse.Namespace(**resume['config'],resume=args.resume)
        for key in PATH_ARGS:
            if getattr(args,key) is not None:
                setattr(args,key,Path(getattr(args,key)))
    if min(args.epochs,args.batch_size,args.patience)<=0 or args.workers<0 or args.smoke_steps<0:
        p.error('Invalid counts/workers')
    if not 0<=args.label_smoothing<1 or min(args.adapter_lr,args.head_lr)<=0 or args.min_delta<0:
        p.error('Invalid learning rate, smoothing or min-delta')
    if not 0<=args.warmup_epochs<=args.epochs:
        p.error('warmup-epochs must be between zero and epochs')
    if args.mode=='linear' and args.init_checkpoint:
        p.error('Frozen linear baseline must start from the unadapted BioCLIP encoder')
    return args,resume


def main():
    args,resume=parse_args()
    torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    manifest=audit_and_split(args)
    if args.prepare_only:
        print('PREPARE COMPLETE:',args.split_manifest.resolve());return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    args.amp=args.precision=='bf16' and torch.cuda.is_bf16_supported()
    if args.precision=='bf16' and not args.amp:
        print('BF16 unsupported; using FP32',flush=True)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    out=args.output_dir
    if resume and (out/'metrics.json').exists():
        print('This run already completed test evaluation. Read metrics.json; no training/test repeated.',flush=True)
        return
    if not resume and out.exists() and any(out.glob('*.pt')):
        raise FileExistsError('Output already contains checkpoints; choose a new directory or --resume last.pt')
    out.mkdir(parents=True,exist_ok=True)
    if resume and resume['manifest_fingerprint']!=manifest['fingerprint']:
        raise ValueError('Resume dataset/manifest differs from checkpoint')
    classes,splits=manifest['classes'],manifest['splits']
    config={k:(str(v.resolve()) if isinstance(v,Path) else v) for k,v in vars(args).items() if k!='resume'}
    save_json(out/'config.json',config)
    save_json(out/'split_manifest.json',manifest)
    print('GPU:',torch.cuda.get_device_name(0),'PyTorch:',torch.__version__,
          'precision:', 'bf16' if args.amp else 'fp32','output:',out.resolve(),flush=True)
    visual,head,train_transform,eval_transform=build_model(args.mode)
    save_json(out/'environment.json',dict(torch=str(torch.__version__),open_clip=open_clip.__version__,
              gpu=torch.cuda.get_device_name(0),train_transform=str(train_transform),eval_transform=str(eval_transform),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if resume:
        load_weights(visual,head,resume,classes,args.mode)
    elif args.init_checkpoint:
        init=torch.load(args.init_checkpoint,weights_only=True,map_location='cpu')
        if init.get('smoke_steps',init.get('config',{}).get('smoke_steps',0)):
            raise ValueError('Do not initialize a production run from a smoke checkpoint')
        load_weights(visual,head,init,classes,args.mode)
        print('Initialized from checkpoint epoch',init['epoch'],flush=True)
    loader_rows=dict(splits)
    if args.smoke_steps:
        # Cover all classes in the short validation check instead of only the first class.
        buckets=[[r for r in splits['val'] if r['label']==i] for i in range(10)]
        interleaved=[r for group in zip(*buckets) for r in group]
        loader_rows['val']=interleaved[:args.smoke_steps*args.batch_size]
    loaders={s:make_loader(args,rows,train_transform if s=='train' else eval_transform,s=='train')
             for s,rows in loader_rows.items() if s!='test' or not (args.smoke_steps or args.skip_test)}
    adapters=[p for p in visual.parameters() if p.requires_grad]
    groups=([{'params':adapters,'lr':args.adapter_lr,'name':'adapter'}] if adapters else [])
    groups += [{'params':head.parameters(),'lr':args.head_lr,'name':'head'}]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    steps_per_epoch=min(len(loaders['train']),args.smoke_steps) if args.smoke_steps else len(loaders['train'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_factor(
        step,args.epochs*steps_per_epoch,round(args.warmup_epochs*steps_per_epoch)))
    counts=np.bincount([r['label'] for r in splits['train']],minlength=10)
    weights=torch.tensor(1/np.sqrt(counts),dtype=torch.float32,device='cuda')
    weights=weights/weights.mean()
    if args.class_weighting == 'none': weights=torch.ones_like(weights)
    save_json(out/'class_weights.json',dict(zip(classes,weights.cpu().tolist())))
    trainable=sum(p.numel() for p in adapters)+sum(p.numel() for p in head.parameters())
    print('Trainable parameters:',trainable,'| class-weighted CE + label smoothing',flush=True)
    history=[];start_epoch=1;bad_epochs=0;elapsed_before=0.
    session_start=time.perf_counter()
    if resume:
        optimizer.load_state_dict(resume['optimizer']);scheduler.load_state_dict(resume['scheduler'])
        history=resume['history'];start_epoch=resume['epoch']+1
        best_loss=resume['best_loss'];best_epoch=resume['best_epoch']
        patience_loss=resume['patience_loss'];bad_epochs=resume['bad_epochs']
        elapsed_before=resume['elapsed_seconds'];set_rng(resume['rng'])
        # last.pt embeds best weights, so resuming does not depend on a second file.
        save_checkpoint(out/'best.pt',resume['best_weights'])
        print(f'Resume at epoch {start_epoch}; incomplete epoch is repeated',flush=True)
    else:
        initial,_,_=evaluate(visual,head,loaders['val'],args.amp,'Initial validation',args.smoke_steps)
        best_loss=patience_loss=initial['loss'];best_epoch=0
        save_json(out/'initial_validation.json',initial)
        best=weight_state(visual,head,classes,0,args.mode)
        best.update(smoke_steps=args.smoke_steps,validation_loss=best_loss,manifest_fingerprint=manifest['fingerprint'])
        save_checkpoint(out/'best.pt',best)
        print('Initial validation:',initial,flush=True)
        initial_state=weight_state(visual,head,classes,0,args.mode)
        initial_state.update(format='tomato-full-v1',config=config,
                             manifest_fingerprint=manifest['fingerprint'],
                             optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=get_rng(),
                             history=[],best_loss=best_loss,best_epoch=0,patience_loss=patience_loss,
                             bad_epochs=0,elapsed_seconds=time.perf_counter()-session_start,best_weights=best)
        save_checkpoint(out/'last.pt',initial_state)
    try:
        for epoch in range(start_epoch,args.epochs+1):
            if bad_epochs>=args.patience:
                print('Early stopping condition already reached.',flush=True);break
            epoch_start=time.perf_counter()
            train=train_epoch(visual,head,loaders['train'],optimizer,scheduler,weights,args,epoch)
            val,val_logits,val_y=evaluate(visual,head,loaders['val'],args.amp,limit=args.smoke_steps)
            val_report=classification_report(val_y,val_logits.argmax(1),labels=list(range(10)),
                                             output_dict=True,zero_division=0)
            lr={g['name']:g['lr'] for g in optimizer.param_groups}
            row=dict(epoch=epoch,train_loss=train['loss'],train_accuracy=train['accuracy'],
                     val_loss=val['loss'],val_accuracy=val['accuracy'],val_macro_f1=val_report['macro avg']['f1-score'],
                     lr_adapter=lr.get('adapter',0.),lr_head=lr['head'],
                     grad_norm_mean=train['grad_norm_mean'],grad_norm_max=train['grad_norm_max'],
                     seconds=time.perf_counter()-epoch_start)
            history.append(row)
            if val['loss']<best_loss:
                best_loss,best_epoch=val['loss'],epoch
                best=weight_state(visual,head,classes,epoch,args.mode)
                best.update(smoke_steps=args.smoke_steps,validation_loss=best_loss,manifest_fingerprint=manifest['fingerprint'])
                save_checkpoint(out/'best.pt',best)
            if val['loss']<patience_loss-args.min_delta:
                patience_loss=val['loss'];bad_epochs=0
            else:
                bad_epochs+=1
            state=weight_state(visual,head,classes,epoch,args.mode)
            state.update(format='tomato-full-v1',config=config,manifest_fingerprint=manifest['fingerprint'],
                         optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=get_rng(),
                         history=history,best_loss=best_loss,best_epoch=best_epoch,patience_loss=patience_loss,
                         bad_epochs=bad_epochs,elapsed_seconds=elapsed_before+time.perf_counter()-session_start,
                         best_weights=torch.load(out/'best.pt',weights_only=True,map_location='cpu'))
            save_checkpoint(out/'last.pt',state)
            write_history(out,history,best_epoch)
            print(f'Epoch {epoch:02d}/{args.epochs}: train loss={train["loss"]:.4f} acc={train["accuracy"]:.2%}'
                  f' | val loss={val["loss"]:.4f} acc={val["accuracy"]:.2%} F1={row["val_macro_f1"]:.2%}'
                  f' | best={best_epoch} patience={bad_epochs}/{args.patience} | {row["seconds"]:.1f}s',flush=True)
    except KeyboardInterrupt:
        print('\nStopped. Resume last.pt to repeat the incomplete epoch; best.pt is preserved.',flush=True)
        return
    best=torch.load(out/'best.pt',weights_only=True,map_location='cpu')
    load_weights(visual,head,best,classes,args.mode)
    _,val_logits,val_y=evaluate(visual,head,loaders['val'],args.amp,'Calibration (validation)',args.smoke_steps)
    temperature=fit_temperature(val_logits,val_y)
    save_json(out/'calibration.json',dict(temperature=temperature,fit_split='val',
        before=probability_metrics(val_logits,val_y),after=probability_metrics(val_logits,val_y,temperature)))
    if args.skip_test:
        print('Complete: validation-only run; test NOT evaluated.',flush=True);return
    if args.smoke_steps:
        logits,y=val_logits,val_y
        rows=loader_rows['val'][:len(y)]
    else:
        _,logits,y=evaluate(visual,head,loaders['test'],args.amp,'TEST (selected checkpoint)')
        rows=splits['test']
    meta=dict(model='BioCLIP ViT-B/16 + '+args.mode,counts={s:len(v) for s,v in splits.items()},
              seed=args.seed,best_epoch=best_epoch,best_val_loss=best_loss,trainable_parameters=trainable,
              elapsed_seconds=elapsed_before+time.perf_counter()-session_start,
              peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
              limitations='Fixed historical image-level benchmark, already inspected. No physical-leaf grouping, near-duplicate or field independence claim.')
    metrics=export_results(out,logits,y,classes,rows,temperature,meta,smoke=bool(args.smoke_steps))
    if not args.smoke_steps:
        compare_reference(args,rows,metrics)
    print(f'COMPLETE best epoch={best_epoch} | {"SMOKE validation" if args.smoke_steps else "TEST"}'
          f' accuracy={metrics.get("test_accuracy",metrics.get("smoke_validation_accuracy")):.2%}'
          f' macro F1={metrics["macro_f1"]:.2%} | {out.resolve()}',flush=True)


if __name__=='__main__':
    main()
