"""Display-only optimization. All learning, validation and checkpoints unchanged."""
import csv,hashlib,json,sys,time
from pathlib import Path
CODE=Path('/home/thanhtung/projects/sam-bioclip-agriculture/experiments/revision_v3')
sys.path.insert(0,str(CODE))
import train_revision as revision

def install(base):
    original_history=base.write_history
    pending={}
    original_tqdm=base.tqdm
    class QuietProgress(original_tqdm):
        def __init__(self,*args,**kwargs):
            kwargs['mininterval']=max(2.,kwargs.get('mininterval',0.))
            super().__init__(*args,**kwargs)
        def set_postfix(self,ordered_dict=None,refresh=True,**kwargs):
            return super().set_postfix(ordered_dict,refresh=False,**kwargs)
    def history(out, rows, best_epoch):
        with (out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        pending.update(out=out,rows=rows,best_epoch=best_epoch)
    base.write_history=history
    base.tqdm=QuietProgress
    original_parse=base.parse_args
    def parse():
        args,state=original_parse()
        p=args.output_dir/'io_optimization_sessions.json'
        records=json.loads(p.read_text()) if p.exists() else []
        records.append(dict(unix=time.time(),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            policy='Only defer curve plotting and throttle progress redraw; CSV, training math, validation, checkpoint cadence unchanged',
            resume_epoch=state.get('epoch') if state else None))
        p.write_text(json.dumps(records,indent=2))
        return args,state
    base.parse_args=parse
    def finish():
        if pending:original_history(pending['out'],pending['rows'],pending['best_epoch'])
    return finish

if __name__=='__main__':
    original=revision.configure_model
    finalizers=[]
    def configure(base,method,rank,placement='mlp'):
        original(base,method,rank,placement)
        finalizers.append(install(base))
    revision.configure_model=configure
    revision.main()
    for finish in finalizers:finish()
