"""Evaluate a locked source model or original zero-shot BioCLIP; never fit on target labels."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from train_revision import load_base, configure_model, class_prompts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dataset', choices=['source', 'plantdoc'], default='source')
    p.add_argument('--root', type=Path)
    p.add_argument('--mapping', type=Path)
    p.add_argument('--mapping-reviewed', action='store_true')
    p.add_argument('--zero-shot', action='store_true')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    import torch
    import numpy as np
    from PIL import Image
    from sklearn.metrics import classification_report
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError('Choose an empty output directory')
    config = json.loads((a.run_dir / 'config.json').read_text())
    suite = json.loads((a.run_dir / 'suite.json').read_text())
    suite.update(weighting=config.get('class_weighting','sqrt'), smoothing=config['label_smoothing'])
    manifest = json.loads((a.run_dir / 'split_manifest.json').read_text())
    source_manifest = json.loads((a.run_dir / 'source_full_manifest.json').read_text())
    classes = manifest['classes']
    base = load_base(Path(__file__).with_name('train_tomato_full_base.py'))
    if hashlib.sha256(Path(__file__).with_name('train_tomato_full_base.py').read_bytes()).hexdigest() != suite['base_sha256']:
        raise ValueError('Base trainer differs from training run')
    excluded = []
    if a.dataset == 'source':
        root = a.root or Path(config['data_root'])
        rows = manifest['splits']['test']
        if a.mapping:
            allowed=set(json.loads(a.mapping.read_text()).values())
            if not allowed.issubset(set(classes)):raise ValueError('Unknown scope label')
            rows=[r for r in rows if r['class'] in allowed]
            if not rows:raise ValueError('Empty source label scope')
    else:
        if not (a.root and a.mapping and a.mapping_reviewed):
            p.error('PlantDoc requires --root, --mapping and --mapping-reviewed after manual label/near-duplicate review')
        root = a.root
        mapping = json.loads(a.mapping.read_text())
        if not mapping:raise ValueError('Empty mapping')
        seen = {r['pixel_sha256'] for split in source_manifest['splits'].values() for r in split}
        rows = []
        for folder, target in sorted(mapping.items()):
            if target not in classes:raise ValueError(f'Unknown source label: {target}')
            label = classes.index(target)
            directory = root / folder
            if not directory.is_dir():raise FileNotFoundError(directory)
            for f in sorted(directory.rglob('*')):
                if f.suffix.lower() not in {'.jpg','.jpeg','.png'}:continue
                with Image.open(f) as image:
                    rgb = image.convert('RGB')
                    digest = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                if digest in seen:
                    excluded.append(dict(path=str(f.relative_to(root)),reason='exact pixel duplicate of source or earlier target image'))
                    continue
                seen.add(digest)
                rows.append(dict(path=str(f.relative_to(root)),label=label,**{'class':target},pixel_sha256=digest))
        if not rows:raise ValueError('No eligible target images')
    # Verify source test images too, so changing --root cannot silently change the benchmark.
    for r in rows:
        with Image.open(root/r['path']) as im:
            rgb=im.convert('RGB');digest=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        if digest != r['pixel_sha256']:raise ValueError('Evaluation image changed: '+r['path'])
    amp = config['precision']=='bf16' and torch.cuda.is_bf16_supported()
    if a.zero_shot:
        import open_clip
        import torch.nn.functional as F
        model, _, transform = open_clip.create_model_and_transforms(base.BASE_MODEL)
        model = model.cuda().eval()
        tokens = open_clip.get_tokenizer(base.BASE_MODEL)(class_prompts(classes)).cuda()
        with torch.no_grad():text = F.normalize(model.encode_text(tokens).float(),dim=-1)
        class ZeroShot(torch.nn.Module):
            def forward(self,x):
                return model.logit_scale.exp().float()*F.normalize(model.encode_image(x).float(),dim=-1)@text.T
        visual, head = ZeroShot().cuda(), torch.nn.Identity().cuda()
        def forward(v,h,x,use_amp):
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=use_amp):return v(x)
        base.forward=forward;temperature=1.
    else:
        configure_model(base,suite['method'],suite['rank'],suite.get('placement','mlp'));base.SUITE_CLASSES=classes
        visual,head,_,transform=base.build_model('linear' if suite['method']=='linear' else 'lora')
        checkpoint=torch.load(a.run_dir/'best.pt',weights_only=True,map_location='cpu')
        base.load_weights(visual,head,checkpoint,classes,'linear' if suite['method']=='linear' else 'lora')
        temperature=json.loads((a.run_dir/'calibration.json').read_text())['temperature']
    args=SimpleNamespace(data_root=root,batch_size=a.batch_size,workers=a.workers,seed=config['seed'])
    loader=base.make_loader(args,rows,transform,False)
    _,logits,y=base.evaluate(visual,head,loader,amp,'Locked evaluation')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    scope=sorted(set(int(r['label']) for r in rows))
    meta=dict(model='BioCLIP zero-shot' if a.zero_shot else suite['method'],dataset=a.dataset,
              experiment=dict(suite, method='BioCLIP zero-shot' if a.zero_shot else suite['method']),
              source_run=str(a.run_dir.resolve()),seed=config['seed'],class_scope=[classes[i] for i in scope],
              protocol='Original ten-way output; predictions outside target label intersection count as errors',
              source_temperature_frozen=True,zero_shot_target_adaptation=True,
              zero_shot_source_training=bool(a.zero_shot))
    metrics=base.export_results(a.output_dir,logits,y,classes,rows,temperature,meta)
    metrics['evaluation_fingerprint']=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    if a.dataset=='plantdoc' or a.mapping:
        report=classification_report(y.numpy(),logits.argmax(1).numpy(),labels=scope,
                                    target_names=[classes[i] for i in scope],zero_division=0,output_dict=True)
        for name in ['precision','recall','f1']:
            key='macro_'+name
            metrics[key+'_all10']=metrics[key]
            metrics[key]=report['macro avg'][name if name!='f1' else 'f1-score']
        metrics['balanced_accuracy']=metrics['macro_recall']
        metrics['weighted_f1']=report['weighted avg']['f1-score']
        metrics['classification_report_supported_classes']=report
        metrics['macro_scope']='Only ground truth target classes with positive support; ten-way predictions retained'
        (a.output_dir/'classification_report_supported_classes.txt').write_text(classification_report(
            y.numpy(),logits.argmax(1).numpy(),labels=scope,target_names=[classes[i] for i in scope],zero_division=0,digits=4))
    base.save_json(a.output_dir/'metrics.json',metrics)
    base.save_json(a.output_dir/'evaluation_manifest.json',dict(classes=classes,rows=rows,excluded=excluded,
                                                              mapping=json.loads(a.mapping.read_text()) if a.mapping else None))
    print('COMPLETE',json.dumps({k:metrics[k] for k in ['test_accuracy','macro_f1','macro_recall']}))


if __name__=='__main__':main()
