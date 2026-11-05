import csv,json,statistics,tempfile,time
from pathlib import Path
import train_io
base=train_io.revision.load_base(train_io.CODE/'train_tomato_full_base.py')
R=train_io.CODE.parents[1]
rows=list(csv.DictReader((R/'experiments/results/revision_v3_20261104/lr/v00_lr0.03_s17/history.csv').open()))
rows=[{k:float(v) for k,v in r.items()} for r in rows]
best=min(rows,key=lambda r:r['val_loss'])['epoch']
original=base.write_history
untouched={k:getattr(base,k) for k in ['train_epoch','evaluate','make_loader','save_checkpoint','forward']}
with tempfile.TemporaryDirectory() as tmp:
 out=Path(tmp);old=[];new=[]
 for _ in range(3):
  start=time.perf_counter();original(out,rows,best);old.append(time.perf_counter()-start)
 expected=(out/'history.csv').read_bytes()
 finish=train_io.install(base)
 for _ in range(3):
  start=time.perf_counter();base.write_history(out,rows,best);new.append(time.perf_counter()-start)
 assert (out/'history.csv').read_bytes()==expected
 assert all(getattr(base,k) is v for k,v in untouched.items())
 finish();assert (out/'learning_curves.png').exists()
 result=dict(original_history_seconds=statistics.median(old),csv_only_seconds=statistics.median(new),
   tested_epochs=len(rows),csv_bytes_identical=True,learning_functions_unchanged=list(untouched),final_plot_created=True)
 (Path(__file__).parent/'io_benchmark.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
