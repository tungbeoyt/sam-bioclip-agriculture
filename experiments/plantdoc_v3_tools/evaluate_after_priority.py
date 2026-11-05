"""Wait for frozen priority suite, then evaluate locked models on reviewed target."""
import hashlib,json,subprocess,time
from pathlib import Path
R=Path('/home/thanhtung/projects/sam-bioclip-agriculture')
OUT=R/'experiments/results/revision_v3_20261104'
REVIEW=R/'experiments/plantdoc_v3_review'
PY='/home/thanhtung/projects/my-ai-project/.venv/bin/python'
while True:
 status=(OUT/'queue_status.txt').read_text() if (OUT/'queue_status.txt').exists() else ''
 if status.startswith('FAILED'):raise RuntimeError('Priority queue failed; inspect before target evaluation')
 if status.startswith('COMPLETE'):break
 time.sleep(30)
report=json.loads((REVIEW/'reviewed_inventory.json').read_text())
rows=json.loads((REVIEW/'evaluation_rows.json').read_text())
assert report['review_complete']
assert hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()==report['evaluation_inventory_sha256']
from PIL import Image
for row in rows:
 with Image.open(Path(report['target_root'])/row['evaluation_path']) as im:
  rgb=im.convert('RGB');h=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
 assert h==row['pixel_sha256']
for item in json.loads((OUT/'lr/selection.json').read_text())['selected']:
 for run in item['runs']:
  for domain in ['plantdoc','source_matched_labels']:
   dest=OUT/'cross_domain'/Path(run).name/domain
   if (dest/'metrics.json').exists():continue
   cmd=[PY,str(R/'experiments/revision_v3/evaluate_revision.py'),'--run-dir',run,'--output-dir',str(dest),
        '--mapping',str(REVIEW/'mapping_reviewed.json'),'--batch-size','8','--workers','4']
   if domain=='plantdoc':cmd+=['--dataset','plantdoc','--root',report['target_root'],'--mapping-reviewed']
   else:cmd+=['--root',str(R/'data/cls/plantvillage/color')]
   print('EVALUATE',domain,run,flush=True)
   subprocess.run(cmd,check=True)
(OUT/'cross_domain/COMPLETE.txt').write_text('15 source checkpoints evaluated on reviewed PlantDoc and matched source label scope. No target adaptation.')
print('COMPLETE cross_domain',flush=True)
