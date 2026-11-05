"""Run the unchanged experiment plan with the measured performance wrapper."""
import sys
from pathlib import Path
import experiment_v2

original=experiment_v2.call
def call(cmd):
    converted=[str(Path(__file__).with_name('train_optimized.py'))
               if Path(str(x)).name=='train_revision.py' else x for x in cmd]
    original(converted)

if __name__=='__main__':
    experiment_v2.call=call
    experiment_v2.main()
