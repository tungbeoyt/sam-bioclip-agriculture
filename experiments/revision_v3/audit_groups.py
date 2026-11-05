"""pHash candidate review and group-preserving repartition. This does not create a blind test."""
import argparse, csv, hashlib, json, random
from pathlib import Path

def components(rows, pairs):
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,j in pairs:parent[find(i)]=find(j)
    groups={}
    for i,r in enumerate(rows):groups.setdefault(find(i),[]).append(r)
    return list(groups.values())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['candidates','split'])
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--distance',type=int,default=6)
    p.add_argument('--per-class',type=int,default=100)
    a=p.parse_args();m=json.loads(a.manifest.read_text())
    rows=sorted([r for split in m['splits'].values() for r in split],key=lambda r:r['path'])
    if a.action=='candidates':
        from PIL import Image
        import imagehash
        # BK-tree gives exact Hamming-radius queries without all-pairs image comparisons.
        tree=None;pairs=[]
        for i,r in enumerate(rows):
            with Image.open(a.root/r['path']) as im:
                rgb=im.convert('RGB')
                digest=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                if digest!=r['pixel_sha256']:raise ValueError('Image changed: '+r['path'])
                h=int(str(imagehash.phash(rgb)),16)
            stack=[tree] if tree else []
            while stack:
                old,j,children=stack.pop();d=(h^old).bit_count()
                if d<=a.distance:pairs.append((j,i,d))
                stack.extend(child for edge,child in children.items() if d-a.distance<=edge<=d+a.distance)
            node=(h,i,{})
            if tree is None:tree=node
            else:
                cur=tree
                while True:
                    d=(h^cur[0]).bit_count()
                    if d not in cur[2]:cur[2][d]=node;break
                    cur=cur[2][d]
        if a.review.exists():raise FileExistsError(a.review)
        with a.review.open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['path_a','path_b','hamming','same_object'])
            for i,j,d in pairs:w.writerow([rows[i]['path'],rows[j]['path'],d,''])
        print(f'{len(pairs)} candidates; manually fill same_object with yes/no');return
    index={r['path']:i for i,r in enumerate(rows)};pairs=[]
    for r in csv.DictReader(a.review.open(encoding='utf-8')):
        if r['same_object'] not in ('yes','no'):raise ValueError('Every candidate requires yes/no review')
        if r['same_object']=='yes':pairs.append((index[r['path_a']],index[r['path_b']]))
    groups=components(rows,pairs);random.Random(20261030).shuffle(groups)
    splits={'train':[],'val':[],'test':[]}
    for label in range(len(m['classes'])):
        buckets=[g for g in groups if {r['label'] for r in g}=={label}]
        if any(label in {r['label'] for r in g} and len({r['label'] for r in g})>1 for g in groups):
            raise ValueError('Reviewed object group has conflicting labels; resolve labels before splitting')
        for split in ('test','val'):
            while sum(r['label']==label for r in splits[split])<a.per_class and buckets:
                splits[split].extend(buckets.pop())
            if sum(r['label']==label for r in splits[split])<a.per_class:raise ValueError('Insufficient evaluation support')
        for g in buckets:splits['train'].extend(g)
    if any(not any(r['label']==i for r in splits['train']) for i in range(len(m['classes']))):raise ValueError('No train samples for a class')
    if not a.output or a.output.exists():raise ValueError('Provide a fresh --output path')
    result=dict(classes=m['classes'],splits=splits,policy='Reviewed pHash candidate groups; historical data repartition, NOT blind or proven leaf-independent',
                review_sha256=hashlib.sha256(a.review.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2));print({k:len(v) for k,v in splits.items()})

if __name__=='__main__':main()
