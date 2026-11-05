#!/usr/bin/env python3
"""Read-only progress dashboard. No torch import, checkpoint loading or training changes."""
import argparse,csv,io,json,os,re,statistics,subprocess,sys,time
from datetime import datetime,timezone,timedelta
from pathlib import Path

DEFAULT=Path('/home/thanhtung/projects/sam-bioclip-agriculture/experiments/results/revision_v3_20261104')
TRAINERS={'train_revision.py','train_io.py'}

def read_json(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def tail(path,limit=65536):
    try:
        with path.open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-limit));return f.read().decode('utf-8',errors='replace')
    except OSError:return ''

def history(path):
    text=tail(path,2_000_000)
    # Ignore the incomplete final line if the trainer is currently rewriting CSV.
    if text and not text.endswith('\n'):text=text.rsplit('\n',1)[0]
    valid=[]
    try:
        for r in csv.DictReader(io.StringIO(text)):
            try:int(r['epoch']);float(r['seconds']);float(r['val_loss'])
            except (KeyError,TypeError,ValueError):continue
            valid.append(r)
    except csv.Error:pass
    return valid

def processes():
    found={}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            args=p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if len(args)<2 or Path(args[1]).name not in TRAINERS|{'evaluate_revision.py'}:continue
            status=p.joinpath('status').read_text()
            parent=int(re.search(r'^PPid:\s+(\d+)',status,re.M)[1])
            state=re.search(r'^State:\s+(\S+)',status,re.M)[1]
            found[int(p.name)]=dict(pid=int(p.name),parent=parent,state=state,args=args,cwd=str(p.joinpath('cwd').resolve()))
        except (OSError,UnicodeError,TypeError):continue
    # DataLoader workers inherit argv; count only the main trainer/evaluator.
    return [v for v in found.values() if v['parent'] not in found and v['state']!='Z']

def arg(proc,key):
    args=proc['args']
    try:return args[args.index(key)+1]
    except (ValueError,IndexError):return None

def absolute(proc,value):
    if not value:return None
    p=Path(value);return str((p if p.is_absolute() else Path(proc['cwd'])/p).resolve())

def progress(log):
    parts=[s.strip() for s in re.split(r'[\r\n]',log) if s.strip()]
    for line in reversed(parts):
        m=re.search(r'(Train (\d+)/(\d+)|Initial validation|Validation|Calibration \(validation\)|Locked evaluation):.*?\|\s*(\d+)/(\d+)',line)
        if m:
            return dict(stage=m[1],epoch=int(m[2]) if m[2] else None,
                        batch=int(m[4]),batches=int(m[5]),percent=100*int(m[4])/max(1,int(m[5])))
    return None

def pct(done,total):return 100*done/total if total else 0.

def snapshot(root):
    running=processes();byrun={};evaluators=[]
    for p in running:
        if Path(p['args'][1]).name in TRAINERS:
            out=absolute(p,arg(p,'--output-dir'))
            if not out:
                resume=absolute(p,arg(p,'--resume'));out=str(Path(resume).parent) if resume else None
            if out:byrun[out]=p
        else:
            dest=absolute(p,arg(p,'--output-dir'))
            if dest and Path(dest).is_relative_to(root):evaluators.append(dict(pid=p['pid'],output=dest,state=p['state']))
    rows=[];stages=[]
    phases=[('lr',12),('fewshot',45)]
    if (root/'ablation_fixed/protocol.json').exists():phases.append(('ablation_fixed',33))
    if (root/'ablation_minimal/protocol.json').exists():phases.append(('ablation_minimal',18))
    for phase,expected in phases:
        protocol=read_json(root/phase/'protocol.json',{})
        jobs=protocol.get('jobs',[]);total=len(jobs) if jobs else expected;done=0;active=0
        for job in jobs:
            out=Path(job['run']);cfg=read_json(out/'config.json',{});h=history(out/'history.csv')
            complete=bool(read_json(out/'calibration.json'))
            proc=byrun.get(str(out.resolve()));v=job['variant'];log=tail(out/'train.log')
            state='XONG' if complete else ('TAM DUNG' if proc and proc['state']=='T' else 'DANG CHAY' if proc else 'DO DANG' if (out/'last.pt').exists() else 'LOI/DO DANG' if (out/'suite.json').exists() else 'CHO')
            done+=complete;active+=bool(proc)
            epoch=int(h[-1]['epoch']) if h else 0;cap=cfg.get('epochs',protocol.get('common',{}).get('epochs',150))
            live=progress(log) if proc else None
            # This is used epoch budget, not predicted completion at early stopping.
            used=max(epoch,live['epoch']-1+live['batch']/max(1,live['batches'])) if live and live['epoch'] else epoch
            rows.append(dict(phase=phase,name=out.name,method=v['method'],shots=v['shots'],seed=job['seed'],
                state=state,completed_epochs=epoch,epoch_cap=cap,epoch_budget_percent=min(100,pct(used,cap)),
                complete=complete,pid=proc['pid'] if proc else None,live=live,
                last_epoch_seconds=float(h[-1]['seconds']) if h else None,
                median_recent_epoch_seconds=statistics.median(float(x['seconds']) for x in h[-5:]) if h else None,
                best_val_ce=min(float(x['val_loss']) for x in h) if h else None,
                log_age_seconds=round(time.time()-(out/'train.log').stat().st_mtime) if (out/'train.log').exists() else None))
        stages.append(dict(phase=phase,done=done,total=total,percent=pct(done,total),active=active,plan_locked=bool(jobs)))
    evaluations=[]
    for name,pattern,total in [('Source sau LR','lr/evaluation/*/source/metrics.json',15),
                               ('Source few-shot','fewshot/evaluation/*/source/metrics.json',45),
                               ('PlantDoc + source 7 nhan','cross_domain/*/*/metrics.json',30)]:
        n=sum(bool(read_json(p)) for p in root.glob(pattern));evaluations.append(dict(phase=name,done=n,total=total,percent=pct(n,total)))
    for phase,count in phases[2:]:
        n=sum(bool(read_json(p)) for p in root.glob(phase+'/evaluation/*/metrics.json'))
        evaluations.append(dict(phase=phase+' (+3 anchor)',done=n,total=count+3,percent=pct(n,count+3)))
    try:queue=(root/'queue_status.txt').read_text().strip()
    except OSError:queue='Chua co queue_status'
    done=sum(s['done'] for s in stages);total=sum(s['total'] for s in stages)
    completed=done+sum(e['done'] for e in evaluations);tasks=total+sum(e['total'] for e in evaluations)
    active_rows=[r for r in rows if r['pid']]
    status='DANG TRAIN' if active_rows else 'DANG DANH GIA' if evaluators else 'KHONG THAY TIEN TRINH GPU CUA GOI NAY'
    if completed==tasks:status='HOAN TAT GOI DA THEO DOI'
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=2).stdout.strip()
    except (OSError,subprocess.TimeoutExpired):gpu='Khong doc duoc nvidia-smi'
    return dict(time_vietnam=datetime.now(timezone(timedelta(hours=7))).strftime('%d/%m/%Y %H:%M:%S'),
        status=status,gpu=gpu,queue_status=queue,train_done=done,train_total=total,train_percent=pct(done,total),
        tasks_done=completed,tasks_total=tasks,tasks_percent=pct(completed,tasks),
        stages=stages,evaluations=evaluations,runs=rows,active_evaluations=evaluators,
        ablation_planned=len(phases)>2)

def bar(value):
    n=int(max(0,min(100,value))/5);return '['+'#'*n+'-'*(20-n)+']'

def render(s,all_runs=False):
    lines=[f"THEO DOI BIOCLIP / LoRA v3 | {s['time_vietnam']} (VN)",s['status'],f"GPU: {s['gpu']}",'',
      f"TRAIN: {s['train_done']}/{s['train_total']} luot XONG  {bar(s['train_percent'])} {s['train_percent']:.1f}%",
      f"Ca goi (train + danh gia): {s['tasks_done']}/{s['tasks_total']} tac vu  {s['tasks_percent']:.1f}%",
      'Ty le tren dem tac vu hoan tat, KHONG phai % thoi gian con lai.','',
      'GIAI DOAN                         XONG / TONG       TY LE']
    for stage in s['stages']+s['evaluations']:
        lines.append(f"{stage['phase']:<33} {stage['done']:>3} / {stage['total']:<5} {stage['percent']:>7.1f}%")
    if not s['ablation_planned']:lines.append('Ablation: CHUA XEP CHAY — khong tinh vao tong o tren.')
    lines+=['','LUOT DANG CHAY']
    active=[r for r in s['runs'] if r['pid']]
    for r in active:
        lines.append(f"{r['phase']} | {r['method']} | seed={r['seed']} | K={r['shots'] or 'all'} | PID={r['pid']}")
        lines.append(f"  {r['name']} | {r['state']} | da xong epoch {r['completed_epochs']}/{r['epoch_cap']}")
        p=r['live']
        if p:lines.append(f"  {p['stage']}: batch {p['batch']}/{p['batches']} {bar(p['percent'])} {p['percent']:.1f}%")
        lines.append(f"  Ngan sach epoch da dung: {r['epoch_budget_percent']:.1f}% (co the early-stop truoc 100%)")
        if r['median_recent_epoch_seconds'] is not None:lines.append(f"  Toc do 5 epoch gan nhat: ~{r['median_recent_epoch_seconds']:.1f}s/epoch | log cap nhat {r['log_age_seconds']}s truoc")
    for e in s['active_evaluations']:lines.append(f"Danh gia PID={e['pid']}: {e['output']}")
    if not active and not s['active_evaluations']:lines.append('Khong co trainer/evaluator dang hoat dong; co the dang chuyen buoc hoac da dung. Kiem tra log.')
    lines+=['','TUNG LUOT (XONG = da hieu chuan; % dang chay = ngan sach epoch)',
            'Goi       Phuong phap   Seed    K    Epoch        %     Trang thai']
    display=s['runs'] if all_runs else ([r for r in s['runs'] if r['complete']][-4:]+[r for r in s['runs'] if r['pid']]+[r for r in s['runs'] if r['state']=='CHO'][:3])
    for r in display:
        percent=100 if r['complete'] else r['epoch_budget_percent']
        lines.append(f"{r['phase']:<9} {r['method']:<13} {r['seed']:>4} {str(r['shots'] or 'all'):>4} {r['completed_epochs']:>4}/{r['epoch_cap']:<4} {percent:>6.1f}%  {r['state']}")
    lines+=['',f"Queue ghi lan cuoi: {s['queue_status']}",
      'Epoch/batch lay tu log gan nhat; co the tre 2 giay. Validation/calibration la buoc rieng.',
      'Ctrl+C: chi dong bang theo doi, KHONG dung train. Them --all de xem moi luot da khoa.']
    return '\n'.join(lines)

def dashboard(screen,a):
    import curses
    try:curses.curs_set(0)
    except curses.error:pass
    screen.keypad(True);screen.timeout(200)
    lines=[];offset=0;next_update=0
    while True:
        if time.monotonic()>=next_update:
            lines=render(snapshot(a.root.resolve()),a.all).splitlines()
            next_update=time.monotonic()+max(1,a.interval)
        height,width=screen.getmaxyx()
        available=max(1,height-1)
        offset=min(offset,max(0,len(lines)-available))
        screen.erase()
        for y,line in enumerate(lines[offset:offset+available]):
            try:screen.addnstr(y,0,line,max(0,width-1))
            except curses.error:pass
        footer=f"q/Ctrl+C: dong monitor | Up/Down/PgUp/PgDn: cuon | r: lam moi | {offset+1}-{min(offset+available,len(lines))}/{len(lines)} dong"
        try:screen.addnstr(height-1,0,footer,max(0,width-1),curses.A_REVERSE)
        except curses.error:pass
        screen.refresh()
        key=screen.getch()
        if key in (ord('q'),ord('Q'),3):return
        if key==curses.KEY_DOWN:offset+=1
        elif key==curses.KEY_UP:offset=max(0,offset-1)
        elif key==curses.KEY_NPAGE:offset+=available
        elif key==curses.KEY_PPAGE:offset=max(0,offset-available)
        elif key==curses.KEY_HOME:offset=0
        elif key==curses.KEY_END:offset=max(0,len(lines)-available)
        elif key in (ord('r'),ord('R')):next_update=0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--interval',type=float,default=5)
    p.add_argument('--once',action='store_true');p.add_argument('--all',action='store_true');p.add_argument('--json',action='store_true')
    a=p.parse_args()
    try:
        if a.once or a.json or not (sys.stdout.isatty() and sys.stdin.isatty()):
            s=snapshot(a.root.resolve())
            print(json.dumps(s,ensure_ascii=False,indent=2) if a.json else render(s,a.all),flush=True)
        else:
            import curses
            curses.wrapper(dashboard,a)
    except KeyboardInterrupt:print('\nDa dong bang theo doi. Train van tiep tuc.')

if __name__=='__main__':main()
