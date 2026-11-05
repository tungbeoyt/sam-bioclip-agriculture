"""Keep every pHash candidate component in one split; no identity claim."""
import argparse,csv,hashlib,json,random
from collections import Counter
from pathlib import Path
from audit_groups import components

def split_candidates(manifest,pairs,per_class=100,seed=20261030):
    rows=sorted([r for v in manifest['splits'].values() for r in v],key=lambda r:r['path'])
    index={r['path']:i for i,r in enumerate(rows)}
    if len(index)!=len(rows):raise ValueError('Repeated manifest paths')
    groups=components(rows,[(index[r['path_a']],index[r['path_b']]) for r in pairs])
    rng=random.Random(seed);rng.shuffle(groups)
    # Non-singleton components first. Large/awkward components remain in train.
    groups.sort(key=lambda g:len(g)==1)
    splits={s:[] for s in ('train','val','test')}
    counts={s:Counter() for s in splits};group_records=[]
    for group in groups:
        gc=Counter(r['label'] for r in group)
        eligible=[s for s in ('val','test') if all(counts[s][c]+n<=per_class for c,n in gc.items())]
        chosen=min(eligible,key=lambda s:len(splits[s])) if eligible else 'train'
        splits[chosen].extend(group);counts[chosen].update(gc)
        group_records.append(dict(split=chosen,paths=[r['path'] for r in group]))
    for c in range(len(manifest['classes'])):
        if any(counts[s][c]!=per_class for s in ('val','test')):raise ValueError('Cannot fill evaluation counts with whole components')
        if counts['train'][c]<1:raise ValueError('Empty training class')
    assignment={r['path']:s for s,rs in splits.items() for r in rs}
    assert all(assignment[r['path_a']]==assignment[r['path_b']] for r in pairs)
    assert sum(map(len,splits.values()))==len(rows)
    hashes=[r['pixel_sha256'] for r in rows]
    if len(hashes)!=len(set(hashes)):raise ValueError('Input must already be exact-pixel deduplicated')
    for s in splits:splits[s].sort(key=lambda r:r['path'])
    return dict(classes=manifest['classes'],splits=splits,
                policy='Conservative connected components of all pHash candidate pairs; not verified physical-leaf identities; historical repartition, NOT blind',
                seed=seed),dict(candidate_pairs=len(pairs),images=len(rows),components=len(groups),
                largest_component=max(map(len,groups)),cross_split_candidate_pairs=0,
                per_class={s:dict(counts[s]) for s in counts},groups=group_records)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-class',type=int,default=100)
    a=p.parse_args()
    if a.per_class<1:p.error('per-class must be positive')
    audit=a.output.with_suffix('.audit.json')
    if a.output.exists() or audit.exists():raise FileExistsError('Use fresh output paths')
    manifest=json.loads(a.manifest.read_text())
    pairs=list(csv.DictReader(a.candidates.open(encoding='utf-8-sig')))
    result,report=split_candidates(manifest,pairs,a.per_class)
    result.update(source_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                  candidates_sha256=hashlib.sha256(a.candidates.read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2))
    audit.write_text(json.dumps(report,indent=2))
    print('SPLITS', {k:len(v) for k,v in result['splits'].items()})
    print('Candidate pairs across splits:',report['cross_split_candidate_pairs'])
    print('Largest component:',report['largest_component'])

if __name__=='__main__':main()
