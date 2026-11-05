"""33 fresh ablation runs plus 3 reused matched anchors; fixed source-selected LR."""
import argparse,json,sys,subprocess,hashlib
from pathlib import Path
R=Path('/home/thanhtung/projects/sam-bioclip-agriculture')
CODE=R/'experiments/revision_v3';sys.path.insert(0,str(CODE))
from revision_v3 import frozen, command, read, score
from experiment_v2 import variants, SEEDS
p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','train','evaluate']);a=p.parse_args()
RESULTS=R/'experiments/results/revision_v3_20261104';OUT=RESULTS/'ablation_fixed'
selection=read(RESULTS/'lr/selection.json');anchor=next(x for x in selection['selected'] if x['variant']['method']=='lora')
common=read(RESULTS/'lr/protocol.json')['common'];jobs=[];reuse=[]
for i,v in enumerate(variants('ablation')):
 if v==anchor['variant']:
  reuse=anchor['runs'];continue
 for seed in SEEDS:jobs.append(dict(variant=v,head_lr=anchor['head_lr'],seed=seed,run=str(OUT/f'v{i:02d}_s{seed}')))
protocol=dict(common=common,jobs=jobs,reused_anchor=reuse,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 policy='Fixed full-source-selected LoRA LR for all variants; one-factor/factorial loss sensitivity, not independently optimized variants; 33 new + 3 reused runs')
frozen(OUT/'protocol.json',protocol)
if a.action=='plan':print(json.dumps(protocol,indent=2));sys.exit()
if a.action=='train':
 if not (RESULTS/'cross_domain/COMPLETE.txt').exists():raise RuntimeError('Finish priority and cross-domain queue first to avoid GPU contention')
 for j in jobs:
  out=Path(j['run'])
  if (out/'calibration.json').exists():continue
  out.mkdir(parents=True,exist_ok=True)
  cmd=command(j,common)
  if (out/'last.pt').exists():cmd=[sys.executable,str(CODE/'train_revision.py'),'--resume',str(out/'last.pt')]
  print('RUN',out,flush=True)
  with (out/'train.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  if not (out/'calibration.json').exists():raise RuntimeError(f'Interrupted: {out}')
else:
 runs=[j['run'] for j in jobs]+reuse
 # Validate the entire suite before looking at any new test metrics.
 for run in runs:score(run)
 for run in runs:
  dest=OUT/'evaluation'/Path(run).name
  if (dest/'metrics.json').exists():continue
  subprocess.run([sys.executable,str(CODE/'evaluate_revision.py'),'--run-dir',run,'--output-dir',str(dest),
    '--root',str(R/'data/cls/plantvillage/color'),'--batch-size','8','--workers','4'],check=True)
 print('COMPLETE ablation_fixed',flush=True)
