"""Fixed-budget pilot for reporting; validation-only, no grid search or test selection."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['smoke','train'])
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--results',type=Path,default=Path('experiments/results/revision_v2_deadline_b32'))
    a=p.parse_args();here=Path(__file__).resolve().parent
    for method in ('lora','mobilenetv2'):
        for seed in ((17,) if a.action=='smoke' else (17,42,2026)):
            out=a.results/f'{method}_s{seed}'
            if a.action=='smoke':out=out.with_name(out.name+'_smoke')
            if (out/'calibration.json').exists():
                cfg=json.loads((out/'config.json').read_text())
                if cfg['batch_size']!=a.batch_size:raise ValueError('Batch changed; use a new results root')
                print('Complete:',out,flush=True);continue
            cmd=[sys.executable,'-u',str(here/'train_revision.py'),'--method',method,
                 '--data-root','data/cls/plantvillage/color','--reference-manifest','reference_conservative_d8_v2.json',
                 '--output-dir',str(out),'--seed',str(seed),'--epochs','40','--patience','7',
                 '--head-lr','0.001','--adapter-lr','0.0001','--batch-size',str(a.batch_size),
                 '--workers','4','--skip-test']
            if a.action=='smoke':cmd+=['--epochs','2','--smoke-steps','3']
            if (out/'last.pt').exists():
                cfg=json.loads((out/'config.json').read_text())
                if cfg['batch_size']!=a.batch_size:raise ValueError('Batch changed; use a new results root')
                cmd=[sys.executable,'-u',str(here/'train_revision.py'),'--resume',str(out/'last.pt')]
            print('RUN',method,seed,'batch',a.batch_size,flush=True)
            subprocess.run(cmd,check=True)
            if not (out/'calibration.json').exists():raise RuntimeError('Run interrupted; rerun the same command')
    print('DONE: validation-only pilot. Check convergence and lock reporting scope before test.',flush=True)

if __name__=='__main__':main()
