"""Locked validation-only extension of v2; fresh pretrained initialization per run."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from experiment_v2 import variants, SEEDS

HERE = Path(__file__).resolve().parent
EXTENDED = {'linear': .03, 'lora': .0003, 'mobilenetv2': .0003, 'full': .0003}


def read(p):
    return json.loads(Path(p).read_text())


def frozen(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True)
    if p.exists() and p.read_text() != content:
        raise RuntimeError(f'Frozen artifact differs: {p}; use a new results root')
    if not p.exists():
        with p.open('x') as f:
            f.write(content)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan(a):
    common = dict(epochs=a.epochs, batch_size=8, workers=4, patience=20,
                  data_root=str(a.data_root.resolve()), reference_manifest=str(a.reference_manifest.resolve()),
                  reference_sha256=digest(a.reference_manifest),
                  code_sha256={p.name: digest(p) for p in sorted(HERE.glob('*.py'))})
    jobs = []
    if a.phase == 'lr':
        specs = [(v, EXTENDED[v['method']]) for v in variants('core') if v['method'] in EXTENDED]
    else:
        selection = read(a.results / 'lr' / 'selection.json')
        common['core_selection_sha256'] = digest(a.results / 'lr' / 'selection.json')
        lrs = {x['variant']['method']: x['head_lr'] for x in selection['selected']}
        if a.phase == 'fewshot':
            specs = [(v, lrs[v['method']]) for v in variants('fewshot')]
            common['policy'] = 'Fixed source-selected LR; no K-specific tuning; 1000 extra validation labels'
        else:
            specs = [(v, lr) for v in variants('ablation') for lr in (.0003, .001, .003, .01)]
            common['policy'] = 'Equal four-LR validation search per ablation variant; 144 runs'
    for i, (v, lr) in enumerate(specs):
        for seed in SEEDS:
            jobs.append(dict(variant=v, head_lr=lr, seed=seed,
                             run=str((a.results / a.phase / f'v{i:02d}_lr{lr}_s{seed}').resolve())))
    return dict(common=common, jobs=jobs)


def command(job, common):
    v = job['variant']
    return [sys.executable, str(HERE/'train_revision.py'), '--method', v['method'],
            '--rank', str(v['rank']), '--placement', v['placement'], '--shots', str(v['shots']),
            '--class-weighting', v['weighting'], '--label-smoothing', str(v['smoothing']),
            '--head-lr', str(job['head_lr']), '--adapter-lr', str(job['head_lr']/10),
            '--seed', str(job['seed']), '--data-root', common['data_root'],
            '--reference-manifest', common['reference_manifest'], '--output-dir', job['run'],
            '--epochs', str(common['epochs']), '--batch-size', '8', '--workers', '4',
            '--patience', '20', '--skip-test']


def score(run):
    run = Path(run)
    if not (run/'calibration.json').exists():
        raise RuntimeError(f'Incomplete run: {run}')
    cfg, suite = read(run/'config.json'), read(run/'suite.json')
    h = list(csv.DictReader((run/'history.csv').open()))
    losses = [(float(read(run/'initial_validation.json')['loss']), 0)]
    losses += [(float(x['val_loss']), int(x['epoch'])) for x in h]
    loss, epoch = min(losses)
    return loss, epoch == cfg['epochs'], cfg, suite


def select(a, protocol):
    grouped = {}
    jobs = list(protocol['jobs'])
    if a.phase == 'lr':
        for i, v in enumerate(variants('core')):
            for lr in (.001, .003, .01):
                for seed in SEEDS:
                    jobs.append(dict(variant=v, head_lr=lr, seed=seed,
                                     run=str((a.old_core/f'v{i:02d}_lr{lr}_s{seed}').resolve())))
    reference = None
    for j in jobs:
        loss, cap, cfg, suite = score(j['run'])
        v = j['variant']
        assert cfg['seed'] == j['seed'] and cfg['head_lr'] == j['head_lr']
        assert all(suite[k] == v[k] for k in ('method', 'rank', 'placement', 'shots'))
        assert cfg['label_smoothing'] == v['smoothing'] and cfg['class_weighting'] == v['weighting']
        assert cfg['init_checkpoint'] is None and cfg['skip_test']
        assert cfg['epochs'] == a.epochs and cfg['batch_size'] == 8
        assert suite['launcher_sha256'] == protocol['common']['code_sha256']['train_revision.py']
        assert suite['base_sha256'] == protocol['common']['code_sha256']['train_tomato_full_base.py']
        m = read(Path(j['run'])/'source_full_manifest.json')
        signature = json.dumps(m['splits'], sort_keys=True)
        if reference is None: reference = signature
        if signature != reference: raise RuntimeError('Source splits differ between runs')
        key = json.dumps(v, sort_keys=True)
        grouped.setdefault(key, {}).setdefault(j['head_lr'], []).append((j, loss, cap))
    chosen = []
    for key, candidates in grouped.items():
        summary = []
        for lr, items in candidates.items():
            if sorted(x[0]['seed'] for x in items) != list(SEEDS):
                raise RuntimeError('Each candidate needs exactly three matching seeds')
            summary.append((sum(x[1] for x in items)/3, lr, items))
        loss, lr, items = min(summary, key=lambda x: (x[0], x[1]))
        if a.phase == 'lr' and any(x[2] for x in items):
            raise RuntimeError('Selected checkpoint reaches epoch limit; inspect validation before proceeding')
        chosen.append(dict(variant=json.loads(key), head_lr=lr, mean_validation_ce=loss,
                           runs=[x[0]['run'] for x in items], lr_grid=sorted(candidates),
                           best_at_epoch_cap=[x[0]['seed'] for x in items if x[2]],
                           candidates=[dict(head_lr=x[1], mean_validation_ce=x[0]) for x in summary],
                           lr_at_boundary=len(candidates)>1 and lr in (min(candidates), max(candidates))))
    frozen(a.results/a.phase/'selection.json', dict(selected=chosen,
           policy='Minimum mean saved-checkpoint validation CE, including epoch 0; historical test already seen'))
    print(json.dumps(chosen, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan','train','select','evaluate','status'])
    p.add_argument('--phase', choices=['lr','fewshot','ablation'], default='lr')
    p.add_argument('--results', type=Path, default=Path('experiments/results/revision_v3'))
    p.add_argument('--old-core', type=Path, default=Path('experiments/results/revision_v2_conservative_d8/core'))
    p.add_argument('--data-root', type=Path, default=Path('data/cls/plantvillage/color'))
    p.add_argument('--reference-manifest', type=Path, default=Path('reference_conservative_d8_v2.json'))
    p.add_argument('--epochs', type=int, default=150)
    a = p.parse_args()
    if a.action == 'status':
        for path in sorted(a.results.glob('*/protocol.json')):
            jobs = read(path)['jobs']
            print(path.parent.name, sum((Path(j['run'])/'calibration.json').exists() for j in jobs), '/', len(jobs))
        return
    protocol = plan(a)
    frozen(a.results/a.phase/'protocol.json', protocol)
    if a.action == 'plan':
        print(f"Locked {len(protocol['jobs'])} runs: {a.results/a.phase/'protocol.json'}")
    elif a.action == 'train':
        for j in protocol['jobs']:
            out = Path(j['run'])
            if (out/'calibration.json').exists(): continue
            out.mkdir(parents=True, exist_ok=True)
            cmd = command(j, protocol['common'])
            if (out/'last.pt').exists():
                cmd = [sys.executable, str(HERE/'train_revision.py'), '--resume', str(out/'last.pt')]
            print('RUN', out, flush=True)
            with (out/'train.log').open('a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not (out/'calibration.json').exists(): raise RuntimeError(f'Run interrupted: {out}')
    elif a.action == 'select': select(a, protocol)
    else:
        for item in read(a.results/a.phase/'selection.json')['selected']:
            for run in item['runs']:
                out = a.results/a.phase/'evaluation'/Path(run).name/'source'
                if (out/'metrics.json').exists(): continue
                subprocess.run([sys.executable, str(HERE/'evaluate_revision.py'), '--run-dir', run,
                                '--output-dir', str(out), '--root', str(a.data_root),
                                '--workers', '4', '--batch-size', '8'], check=True)


if __name__ == '__main__': main()
