"""Use the frozen v3 plan unchanged; route trainer subprocesses through IO wrapper."""
import sys,subprocess
from pathlib import Path
CODE=Path('/home/thanhtung/projects/sam-bioclip-agriculture/experiments/revision_v3')
sys.path.insert(0,str(CODE))
import revision_v3
original=subprocess.run
def run(cmd,*args,**kwargs):
    cmd=list(cmd)
    if len(cmd)>1 and Path(cmd[1]).resolve()==CODE/'train_revision.py':
        cmd[1]=str(Path(__file__).with_name('train_io.py'))
    return original(cmd,*args,**kwargs)
subprocess.run=run
revision_v3.main()
