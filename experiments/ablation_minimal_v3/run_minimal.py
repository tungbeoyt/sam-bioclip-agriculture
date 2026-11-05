"""One-factor ablation: 18 fresh runs and 3 existing anchor runs, no factorial claims."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
R=Path('/home/thanhtung/projects/sam-bioclip-agriculture')
CODE=R/'experiments/revision_v3';sys.path.insert(0,str(CODE))
from revision_v3 import frozen,read,command,score
RESULTS=R/'experiments/results/revision_v3_20261104';OUT=RESULTS/'ablation_minimal'
def make_plan():
 anchor=next(x for x in read(RESULTS/'lr/selection.json')['selected'] if x['variant']['method']=='lora')
 variants=[dict(anchor['variant'],**change) for change in [dict(rank=4),dict(rank=16),dict(placement='attention'),dict(placement='both'),dict(weighting='none'),dict(smoothing=0.)]]
 common=read(RESULTS/'lr/protocol.json')['common']
 jobs=[dict(variant=v,head_lr=anchor['head_lr'],seed=seed,run=str(OUT/f'v{i:02d}_s{seed}')) for i,v in enumerate(variants) for seed in (17,42,2026)]
 for f in CODE.glob('*.py'):
  if f.name in common['code_sha256'] and hashlib.sha256(f.read_bytes()).hexdigest()!=common['code_sha256'][f.name]:raise RuntimeError('Original code changed')
 for run in anchor['runs']:score(run)
 return dict(common=common,jobs=jobs,reused_anchor=anchor['runs'],policy='One factor changed from r8 MLP sqrt-weighting smoothing .05 anchor; fixed source-selected LR; no rank-position or loss interaction claims',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),io_wrapper_sha256=hashlib.sha256((R/'experiments/io_optimization_v3/train_io.py').read_bytes()).hexdigest())
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','train','evaluate','all']);a=p.parse_args()
 protocol=make_plan();frozen(OUT/'protocol.json',protocol)
 if a.action=='plan':print('18 new runs; 3 reused anchors; epochs/batch/seeds and checkpoint policy unchanged');return
 if not (RESULTS/'cross_domain/COMPLETE.txt').exists():raise RuntimeError('Finish priority suite first')
 import fcntl
 with (RESULTS/'queue.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('Another queue is running')
  status=OUT/'status.txt'
  try:
   if a.action in ['train','all']:
    for j in protocol['jobs']:
     out=Path(j['run'])
     if (out/'calibration.json').exists():continue
     out.mkdir(parents=True,exist_ok=True)
     cmd=command(j,protocol['common']);cmd[1]=str(R/'experiments/io_optimization_v3/train_io.py')
     if (out/'last.pt').exists():cmd=[sys.executable,cmd[1],'--resume',str(out/'last.pt')]
     status.write_text('TRAIN '+str(out));print('RUN',out,flush=True)
     with (out/'train.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
     if not (out/'calibration.json').exists():raise RuntimeError('Run interrupted; rerun same command to resume')
   if a.action in ['evaluate','all']:
    runs=[j['run'] for j in protocol['jobs']]+protocol['reused_anchor']
    for run in runs:score(run)
    for run in runs:
     dest=OUT/'evaluation'/Path(run).name
     if (dest/'metrics.json').exists():continue
     status.write_text('EVALUATE '+run)
     subprocess.run([sys.executable,str(CODE/'evaluate_revision.py'),'--run-dir',run,'--output-dir',str(dest),'--root',str(R/'data/cls/plantvillage/color'),'--batch-size','8','--workers','4'],check=True)
   status.write_text('COMPLETE '+a.action)
  except BaseException:
   status.write_text('STOPPED OR FAILED; inspect queue.log and current train.log; completed runs preserved');raise
if __name__=='__main__':main()
