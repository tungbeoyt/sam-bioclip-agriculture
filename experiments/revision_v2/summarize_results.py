"""Summarize actual saved predictions; package compact evidence without checkpoints/data."""
import argparse
import csv
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,default=Path('experiments/results/revision_v1'))
    p.add_argument('--zip',type=Path,default=Path('revision_results.zip'))
    a=p.parse_args();records=[];groups=defaultdict(list)
    for file in sorted(a.results_root.rglob('metrics.json')):
        m=json.loads(file.read_text())
        if 'WARNING' in m:continue
        experiment=m.get('experiment',{})
        row=dict(run=str(file.parent.relative_to(a.results_root)),dataset=m.get('dataset','source'),method=m['model'],
                 rank=experiment.get('rank',''),shots=experiment.get('shots',''),seed=m.get('seed',''),
                 placement=experiment.get('placement','mlp'),weighting=experiment.get('weighting','sqrt'),smoothing=experiment.get('smoothing',.05),
                 accuracy=m['test_accuracy'],macro_f1=m['macro_f1'],macro_precision=m['macro_precision'],macro_recall=m['macro_recall'],
                 trainable_parameters=m.get('trainable_parameters',''),best_epoch=m.get('best_epoch',''),
                 elapsed_seconds=m.get('elapsed_seconds',''),ece=m['calibrated']['ece_15_equal_width_bins'])
        records.append(row);groups[(row['dataset'],row['method'],row['rank'],row['shots'],row['placement'],row['weighting'],row['smoothing'])].append(row)
    if not records:raise SystemExit('No completed metrics.json found')
    a.results_root.mkdir(parents=True,exist_ok=True)
    with (a.results_root/'runs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=[]
    for (dataset,method,rank,shots,placement,weighting,smoothing),rs in groups.items():
        if len({r['seed'] for r in rs})!=len(rs):
            raise ValueError(f'Duplicate seed within {dataset}/{method}/r{rank}/K{shots}; remove duplicate evaluation from this summary root')
        for metric in ['accuracy','macro_f1','macro_precision','macro_recall','ece']:
            values=[r[metric] for r in rs]
            summary.append(dict(dataset=dataset,method=method,rank=rank,shots=shots,placement=placement,weighting=weighting,smoothing=smoothing,metric=metric,n=len(values),
                                mean=statistics.mean(values),sample_std=statistics.stdev(values) if len(values)>1 else None,
                                seeds=[r['seed'] for r in rs]))
    (a.results_root/'mean_std.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    paired=[]
    def predictions(run):
        with (a.results_root/run/'predictions.csv').open(encoding='utf-8') as f:return {r['path']:r for r in csv.DictReader(f)}
    for lora in records:
        if (lora['placement'],lora['weighting'],lora['smoothing'])!=('mlp','sqrt',.05):continue
        if lora['method']!='lora' or lora['rank']!=8 or lora['dataset']!='source':continue
        for other in records:
            if other['method'] not in ['linear','resnet50','mobilenetv2'] or (other['seed'],other['shots'],other['dataset'])!=(lora['seed'],lora['shots'],'source'):continue
            x,y=predictions(lora['run']),predictions(other['run'])
            if x.keys()!=y.keys():raise ValueError('Cannot pair different evaluation sets')
            better=worse=0
            for key in x:
                if x[key]['true_class']!=y[key]['true_class']:raise ValueError('True labels changed')
                ca=x[key]['true_class']==x[key]['predicted_class'];cb=y[key]['true_class']==y[key]['predicted_class']
                better+=ca and not cb;worse+=cb and not ca
            n=better+worse
            # Exact two-sided binomial p under the McNemar null, using log-space to avoid overflow.
            logs=[math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2) for k in range(min(better,worse)+1)]
            high=max(logs);pv=min(1.,2*math.exp(high)*sum(math.exp(v-high) for v in logs)) if n else 1.
            paired.append(dict(lora=lora['run'],other=other['run'],corrected=better,regressed=worse,p_exact=pv))
    previous=0.
    for i,r in enumerate(sorted(paired,key=lambda r:r['p_exact'])):
        previous=max(previous,min(1.,r['p_exact']*(len(paired)-i)))
        r['p_holm_all_comparisons']=previous
    (a.results_root/'paired_tests.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    a.zip.parent.mkdir(parents=True,exist_ok=True)
    excluded={'.pt','.pth','.bin'}
    with zipfile.ZipFile(a.zip,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(a.results_root.rglob('*')):
            if file.is_file() and file.suffix not in excluded and file.resolve()!=a.zip.resolve():
                z.write(file,'results/'+str(file.relative_to(a.results_root)))
        for file in Path(__file__).parent.iterdir():
            if file.is_file() and file.suffix in {'.py','.md','.json'}:z.write(file,'scripts/'+file.name)
    print(f'Completed runs: {len(records)}. Send {a.zip.resolve()}. Keep best.pt and last.pt locally.')


if __name__=='__main__':main()
