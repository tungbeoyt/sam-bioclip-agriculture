"""Inventory and duplicate candidates only. Never trains or evaluates a model."""
import csv, hashlib, json, subprocess
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps, ImageDraw
import imagehash

ROOT=Path('/home/thanhtung/projects/sam-bioclip-agriculture')
PD=ROOT/'data/PlantDoc-Dataset'
PV=ROOT/'data/cls/plantvillage/color'
OUT=ROOT/'experiments/plantdoc_v3_review'
OUT.mkdir(exist_ok=True)
mapping=json.loads((ROOT/'experiments/revision_v3/plantdoc_mapping_review.json').read_text())
manifest=json.loads((ROOT/'experiments/results/revision_v2_conservative_d8/core/v01_lr0.001_s17/source_full_manifest.json').read_text())
source=[r for rows in manifest['splits'].values() for r in rows]
cache=OUT/'source_phash.json'
if cache.exists(): hashed=json.loads(cache.read_text())
else:
 hashed=[]
 for i,r in enumerate(source):
  with Image.open(PV/r['path']) as im: ph=str(imagehash.phash(im.convert('RGB')))
  hashed.append(dict(path=r['path'],pixel_sha256=r['pixel_sha256'],phash=ph))
  if i%2000==0:print('source hash',i,flush=True)
 cache.write_text(json.dumps(hashed))
pixels={r['pixel_sha256'] for r in source}
pool=[(int(r['phash'],16),str(PV/r['path']),'source') for r in hashed]
rows=[];pairs=[];excluded=[];support=Counter()
for split in ['train','test']:
 for folder,target in sorted(mapping.items()):
  for f in sorted((PD/split/folder).glob('*')):
   if not f.is_file() or f.suffix.lower() not in ['.jpg','.jpeg','.png']:continue
   try:
    with Image.open(f) as im:
     rgb=im.convert('RGB');pixel=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest();ph=int(str(imagehash.phash(rgb)),16)
   except Exception as e:
    excluded.append(dict(path=str(f),reason='unreadable',detail=str(e)));continue
   if pixel in pixels:
    excluded.append(dict(path=str(f),reason='exact pixel duplicate of source or earlier target'));continue
   for other,path,origin in pool:
    distance=(ph^other).bit_count()
    if distance<=8:pairs.append(dict(target=str(f),other=path,origin=origin,hamming=distance,decision=''))
   pool.append((ph,str(f),'target'));pixels.add(pixel)
   rows.append(dict(path=str(f.relative_to(PD)),original_split=split,source_class=target,pixel_sha256=pixel,phash=f'{ph:016x}'))
   support[split+' / '+target]+=1
with (OUT/'near_duplicate_candidates.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['target','other','origin','hamming','decision']);w.writeheader();w.writerows(pairs)
report=dict(upstream='https://github.com/pratikkayal/PlantDoc-Dataset',commit=subprocess.check_output(['git','-C',str(PD),'rev-parse','HEAD'],text=True).strip(),
 protocol='Pooled original train and test used ONLY as external evaluation; no target labels for training or tuning. Ten-way output, mapped disease labels only.',
 mapping=mapping,support=dict(support),eligible=len(rows),excluded=excluded,near_duplicate_candidates=len(pairs),
 excluded_labels=['Tomato leaf','Tomato leaf yellow virus'],review_complete=False)
(OUT/'inventory.json').write_text(json.dumps(report,indent=2))
(OUT/'target_inventory.json').write_text(json.dumps(rows,indent=2))
for start in range(0,len(pairs),8):
 sheet=Image.new('RGB',(1000,8*190),'white');draw=ImageDraw.Draw(sheet)
 for j,item in enumerate(pairs[start:start+8]):
  y=j*190;draw.text((5,y),f"Pair {start+j}: d={item['hamming']} {item['origin']}",fill='black')
  for x,key in [(0,'target'),(500,'other')]:
   with Image.open(item[key]) as im:sheet.paste(ImageOps.contain(im.convert('RGB'),(480,160)),(x,y+25))
 sheet.save(OUT/f'candidates_{start:03d}.jpg')
print(json.dumps(report,indent=2),flush=True)
