"""Let active training finish, then replace only the waiting queue controllers."""
import json,os,signal,subprocess,time
from pathlib import Path
R=Path('/home/thanhtung/projects/sam-bioclip-agriculture')
HERE=Path(__file__).resolve().parent
STATE=HERE/'handover_status.json'
def cmd(pid):
 try:return Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
 except FileNotFoundError:return []
def parent(pid):
 text=Path(f'/proc/{pid}/status').read_text()
 return int(next(s.split()[1] for s in text.splitlines() if s.startswith('PPid:')))
def live(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]!='Z'
 except FileNotFoundError:return False
controllers=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  c=cmd(p.name)
  if len(c)>2 and c[1]=='experiments/revision_v3/revision_v3.py' and c[2]=='train':controllers.append(int(p.name))
if len(controllers)!=1:raise RuntimeError(f'Expected one original controller, found {controllers}')
controller=controllers[0];shell=parent(controller)
assert cmd(shell)==['bash','experiments/revision_v3/run_priority.sh']
assert Path(f'/proc/{controller}/cwd').resolve()==R
paused=[];done=False
try:
 # Pause only orchestration; its training child and DataLoader workers continue.
 for pid in [shell,controller]:os.kill(pid,signal.SIGSTOP);paused.append(pid)
 children=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit() and cmd(p.name):
   try:
    if parent(p.name)==controller:children.append(int(p.name))
   except FileNotFoundError:pass
 STATE.write_text(json.dumps(dict(status='waiting_for_active_run_to_finish',controller=controller,shell=shell,active_children=children),indent=2))
 while any(live(pid) for pid in children):time.sleep(5)
 # No trainer remains. Prevent the old shell ERR trap from marking a false failure.
 os.kill(shell,signal.SIGKILL);os.kill(controller,signal.SIGKILL)
 time.sleep(1)
 with (R/'experiments/v3_priority_io.log').open('a') as log:
  process=subprocess.Popen(['bash',str(HERE/'run_priority_io.sh')],cwd=R,stdout=log,stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,start_new_session=True)
 time.sleep(2)
 if process.poll() is not None:raise RuntimeError('Replacement queue exited; inspect v3_priority_io.log')
 done=True
 STATE.write_text(json.dumps(dict(status='io_optimized_queue_running',pid=process.pid,applied_at='run boundary; active training was not interrupted'),indent=2))
finally:
 if not done:
  for pid in paused:
   try:os.kill(pid,signal.SIGCONT)
   except ProcessLookupError:pass
