"""Validation-only tuning, frozen selection, then explicitly requested evaluation."""
import argparse, csv, hashlib, itertools, json, subprocess, sys
from pathlib import Path
SEEDS=(17,42,2026)
HERE=Path(__file__).resolve().parent

def variants(phase):
    core=[dict(method=m,rank=8,placement='mlp',weighting='sqrt',smoothing=.05,shots=0)
          for m in ('linear','lora','mobilenetv2','resnet50','full')]
    if phase=='core': return core
    if phase=='fewshot':
        return [dict(v,shots=k) for v in core[:3] for k in (1,2,4,8,16)]
    anchor=core[1]
    return [dict(anchor,rank=r,placement=p) for r,p in itertools.product((4,8,16),('mlp','attention','both'))]+[
        dict(anchor,weighting=w,smoothing=s) for w,s in itertools.product(('sqrt','none'),(0.,.05))
        if (w,s)!=('sqrt',.05)]

def call(cmd):
    print(' '.join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','select','evaluate','smoke'])
    p.add_argument('--phase',choices=['core','fewshot','ablation'],default='core')
    p.add_argument('--results',type=Path,default=Path('experiments/results/revision_v2'))
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--reference-manifest',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--plantdoc-root',type=Path)
    p.add_argument('--mapping',type=Path)
    p.add_argument('--mapping-reviewed',action='store_true')
    a=p.parse_args()
    root=a.results/a.phase
    if a.action in ('train','smoke'):
        vs=variants(a.phase)
        if a.action=='smoke':
            vs=variants('core')+[dict(variants('core')[1],placement=p) for p in ('attention','both')]
        for i,v in enumerate(vs):
            # Same three candidate LR pairs and seeds for every method/configuration.
            for lr,seed in itertools.product((1e-3,3e-3,1e-2),SEEDS):
                if a.action=='smoke' and (lr!=1e-3 or seed!=17): continue
                out=root/f'v{i:02d}_lr{lr}_s{seed}'
                if a.action=='smoke': out=out.with_name(out.name+'_smoke')
                if (out/'calibration.json').exists(): continue
                cmd=[sys.executable,HERE/'train_revision.py','--method',v['method'],'--rank',v['rank'],
                     '--placement',v['placement'],'--shots',v['shots'],'--class-weighting',v['weighting'],
                     '--label-smoothing',v['smoothing'],'--head-lr',lr,'--adapter-lr',lr/10,
                     '--epochs',a.epochs,'--patience',20,'--skip-test','--output-dir',out,
                     '--data-root',a.data_root,'--reference-manifest',a.reference_manifest,
                     '--seed',seed,'--workers',a.workers,'--batch-size',a.batch_size]
                if a.action=='smoke':cmd+=['--epochs',2,'--smoke-steps',3,'--workers',0,'--batch-size',8]
                if (out/'last.pt').exists():cmd=[sys.executable,HERE/'train_revision.py','--resume',out/'last.pt']
                call(cmd)
                if not (out/'calibration.json').exists():
                    raise SystemExit(f'Run interrupted: {out}. Repeat the same command to resume; no further runs started.')
        return
    lock=root/'selection.json'
    if a.action=='select':
        selected=[]
        for i,v in enumerate(variants(a.phase)):
            candidates=[]
            for lr in (1e-3,3e-3,1e-2):
                runs=[root/f'v{i:02d}_lr{lr}_s{s}' for s in SEEDS]
                losses=[]; boundary=[]
                for run in runs:
                    if not (run/'calibration.json').exists():raise RuntimeError(f'Incomplete run: {run}')
                    history=list(csv.DictReader((run/'history.csv').open()))
                    best=min(history,key=lambda r:float(r['val_loss']))
                    losses.append(float(best['val_loss']))
                    cfg=json.loads((run/'config.json').read_text())
                    boundary.append(int(best['epoch'])==cfg['epochs'])
                candidates.append((sum(losses)/len(losses),lr,runs,boundary))
            loss,lr,runs,boundary=min(candidates,key=lambda x:(x[0],x[1]))
            if any(boundary):raise RuntimeError(f'Variant {i}: selected run hits epoch cap; rerun tuning in NEW root with larger --epochs before looking at test')
            selected.append(dict(variant=v,head_lr=lr,mean_validation_ce=loss,runs=[str(r.resolve()) for r in runs]))
        payload=dict(selection='minimum mean best validation CE across three seeds; no test used',selected=selected)
        text=json.dumps(payload,indent=2)
        if lock.exists() and lock.read_text()!=text:raise RuntimeError('Selection is frozen; choose a new results root')
        lock.write_text(text);print(lock);return
    selection=json.loads(lock.read_text())
    for item in selection['selected']:
        for run in map(Path,item['runs']):
            out=root/'evaluation'/run.name/('plantdoc' if a.plantdoc_root else 'source')
            if (out/'metrics.json').exists():continue
            cmd=[sys.executable,HERE/'evaluate_revision.py','--run-dir',run,'--output-dir',out,
                 '--root',a.plantdoc_root or a.data_root,'--workers',a.workers,'--batch-size',a.batch_size]
            if a.plantdoc_root:
                if not a.mapping or not a.mapping_reviewed:p.error('Review mapping and duplicates before target evaluation')
                cmd+=['--dataset','plantdoc','--mapping',a.mapping,'--mapping-reviewed']
            call(cmd)

if __name__=='__main__':main()
