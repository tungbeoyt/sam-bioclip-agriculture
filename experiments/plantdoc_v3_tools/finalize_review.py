"""Apply visually reviewed duplicate decisions; freeze external evaluation inventory."""
import csv,hashlib,json,shutil
from pathlib import Path
from collections import Counter
R=Path('/home/thanhtung/projects/sam-bioclip-agriculture')
O=R/'experiments/plantdoc_v3_review'; PD=R/'data/PlantDoc-Dataset'
rows=json.loads((O/'target_inventory.json').read_text()); report=json.loads((O/'inventory.json').read_text())
pairs=list(csv.DictReader((O/'near_duplicate_candidates.csv').open()))
parent={str(PD/r['path']):str(PD/r['path']) for r in rows}
def find(x):
 while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
 return x
quarantine=set()
for i,p in enumerate(pairs):
 if p['origin']=='target':
  p['decision']='same image or transformed copy; visually reviewed'
  parent[find(p['target'])]=find(p['other'])
 else:
  p['decision']='not confirmed duplicate; conservatively quarantine target candidate'
  quarantine.add(p['target'])
groups={}
for r in rows:groups.setdefault(find(str(PD/r['path'])),[]).append(r)
keep=[];removed=[]
for group in groups.values():
 group=sorted(group,key=lambda r:r['path'])
 conflict=len({r['source_class'] for r in group})>1
 for i,r in enumerate(group):
  reason='conflicting target labels' if conflict else 'source near-duplicate candidate quarantine' if any(str(PD/x['path']) in quarantine for x in group) else 'reviewed target duplicate' if i else None
  if reason:removed.append(dict(r,reason=reason))
  else:keep.append(r)
dest=R/'data/PlantDoc_v3_reviewed'
if dest.exists():raise FileExistsError('Frozen target directory already exists')
for folder in report['mapping']:(dest/folder).mkdir(parents=True,exist_ok=True)
for r in keep:
 source=PD/r['path'];folder=source.parent.name
 filename=r['original_split']+'_'+hashlib.sha256(r['path'].encode()).hexdigest()[:12]+source.suffix.lower()
 shutil.copy2(source,dest/folder/filename);r['evaluation_path']=str(Path(folder)/filename)
report.update(review_complete=True,visual_review='All 22 candidate pairs inspected; target duplicate pairs confirmed; source candidates quarantined, not asserted identical.',
 label_review='Seven unambiguous folder-name disease mappings retained. Healthy and yellow-virus labels excluded as ambiguous; no expert relabeling claimed.',
 retained=len(keep),retained_support=dict(Counter(r['source_class'] for r in keep)),removed_after_review=removed,
 limitations='pHash radius 8 is candidate screening, not proof of absence of related images or physical plant independence. Spider-mite class has very low support.',
 target_root=str(dest),evaluation_inventory_sha256=hashlib.sha256(json.dumps(keep,sort_keys=True).encode()).hexdigest())
(O/'reviewed_inventory.json').write_text(json.dumps(report,indent=2))
(O/'evaluation_rows.json').write_text(json.dumps(keep,indent=2))
(O/'mapping_reviewed.json').write_text(json.dumps(report['mapping'],indent=2))
with (O/'duplicate_decisions.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
print(json.dumps({k:report[k] for k in ['retained','retained_support','evaluation_inventory_sha256']},indent=2))
