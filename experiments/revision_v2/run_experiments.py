"""Print a fixed run plan; --execute runs it sequentially and resumes matching interrupted jobs."""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

SEEDS=(17,42,2026)


def plan(phase):
    phases={
        'core':[(m,8,0,s) for m,s in itertools.product(['linear','lora'],SEEDS)],
        'cnn':[(m,8,0,s) for m,s in itertools.product(['resnet50','mobilenetv2'],SEEDS)],
        'rank':[('lora',r,0,s) for r,s in itertools.product([2,4,16],SEEDS)],
        'fewshot':[(m,8,k,s) for m,k,s in itertools.product(['linear','lora'],[1,2,4,8,16],SEEDS)],
        'fewshot-cnn':[(m,8,k,s) for m,k,s in itertools.product(['resnet50','mobilenetv2'],[1,2,4,8,16],SEEDS)],
        'vlm':[(m,8,0,s) for m,s in itertools.product(['vlm-vision','vlm-both'],SEEDS)],
        'smoke':[(m,8,2,17) for m in ['linear','lora','resnet50','mobilenetv2','vlm-vision','vlm-both']],
        'smoke-vlm':[(m,8,2,17) for m in ['vlm-vision','vlm-both']]
    }
    return sum((phases[k] for k in ['core','cnn','rank','fewshot','fewshot-cnn','vlm']),[]) if phase=='all' else phases[phase]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['smoke','smoke-vlm','core','cnn','rank','fewshot','fewshot-cnn','vlm','all'],required=True)
    p.add_argument('--results-root',type=Path,default=Path('experiments/results/revision_v1'))
    p.add_argument('--data-root',type=Path,default=Path('data/cls/plantvillage/color'))
    p.add_argument('--reference-manifest',type=Path,default=Path('experiments/results/tomato_7000_20260925_113448/split_manifest.json'))
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args();jobs=plan(a.phase)
    if a.batch_size<=0 or a.workers<0:p.error('Invalid batch/workers')
    launcher=Path(__file__).with_name('train_revision.py').resolve()
    print(f'{a.phase}: {len(jobs)} runs; execute={a.execute}',flush=True)
    if a.execute:
        if not a.reference_manifest.is_file():raise FileNotFoundError(a.reference_manifest)
        a.results_root.mkdir(parents=True,exist_ok=True)
    for method,rank,shots,seed in jobs:
        smoke=a.phase in ('smoke','smoke-vlm')
        name=f'{method}_r{rank}_k{shots or "all"}_s{seed}'+('_smoke' if smoke else '')
        out=a.results_root/name
        expected=out/('smoke_metrics.json' if smoke else 'metrics.json')
        command=[sys.executable,'-u',str(launcher),'--method',method,'--rank',str(rank),'--shots',str(shots),
                 '--output-dir',str(out),'--data-root',str(a.data_root),'--reference-manifest',str(a.reference_manifest),
                 '--seed',str(seed),'--epochs','2' if smoke else '30',
                 '--batch-size','8' if smoke else str(a.batch_size),'--workers','0' if smoke else str(a.workers),
                 '--adapter-lr','3e-5','--head-lr','3e-4','--warmup-epochs','1','--patience','7','--precision','bf16']
        if smoke:command+=['--smoke-steps','3']
        print('\n'+name,flush=True)
        if not a.execute:
            import shlex
            print(shlex.join(command));continue
        if (out/'config.json').exists():
            previous=json.loads((out/'config.json').read_text())
            expected_settings={'batch_size':8 if smoke else a.batch_size,'workers':0 if smoke else a.workers,
                               'epochs':2 if smoke else 30,'seed':seed}
            if any(previous.get(k)!=v for k,v in expected_settings.items()):
                raise ValueError('Run settings differ from the current plan; use a separate results root')
            if Path(previous['data_root']).resolve()!=a.data_root.resolve() or Path(previous['reference_manifest']).resolve()!=a.reference_manifest.resolve():
                raise ValueError('Dataset/reference path changed; use a separate results root')
        if expected.exists():
            suite=json.loads((out/'suite.json').read_text());cfg=json.loads((out/'config.json').read_text())
            if (suite['method'],suite['rank'],suite['shots'],cfg['seed'])!=(method,rank,shots,seed):
                raise ValueError('Existing result does not match planned job')
            print('Already complete; retained');continue
        if (out/'last.pt').exists():
            command=[sys.executable,'-u',str(launcher),'--resume',str(out/'last.pt')]
        elif (out/'suite.json').exists():
            raise RuntimeError(f'{out} stopped before initial checkpoint. Retain this folder and use a new --results-root.')
        out.mkdir(parents=True,exist_ok=True)
        with (out/'train.log').open('a',encoding='utf-8') as log:
            log.write('\nCOMMAND '+repr(command)+'\n');log.flush()
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in proc.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                result=proc.wait()
            except KeyboardInterrupt:
                # The child receives Ctrl+C from the terminal too and saves only completed epochs.
                proc.wait();raise SystemExit('Stopped. Repeat this phase to resume last.pt.')
        if result or not expected.exists():
            raise SystemExit(f'Run incomplete: {out}; inspect train.log before continuing')
    print('\nDONE' if a.execute else '\nPlan only. Add --execute to start.')


if __name__=='__main__':main()
