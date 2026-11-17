from pathlib import Path
import sys,json,zipfile,hashlib,csv,time
import argparse
parser=argparse.ArgumentParser(description='Compare first512 and chunks on identical saved study code; makes no API requests.')
parser.add_argument('--study-zip',type=Path,required=True)
parser.add_argument('--output',type=Path,default=Path('coverage_comparison'))
parser.add_argument('--device',default='cpu')
args=parser.parse_args()
out=args.output
out.mkdir(exist_ok=True)
z=zipfile.ZipFile(args.study_zip)
s=json.loads(z.read('study.json'))
assert len(s['samples'])==48 and all(len(x['analyses'])>=4 for x in s['samples'])
sha=lambda x:hashlib.sha256(x.encode()).hexdigest()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'webapp'))
from inference import Engine
e=Engine(args.device); rows=[]; verification=[]
for condition in ['050','100']:
 for sample in s['samples']:
  for strip in [False,True]:
   r=e.analyze(sample['code'],condition,strip,'first512'); a=r['results'][0]
   saved=next(a for a in sample['analyses'] if a['classifier']==condition and a['comments_removed']==strip and a.get('analysis_mode')=='chunks')
   # Recompute chunked inference locally to avoid device/precision differences.
   c=e.analyze(sample['code'],condition,strip,'chunks'); ca=c['results'][0]
   verification.append({'sample':sample['id'],'classifier':condition,'strip':strip,
      'saved_label':saved['predicted_label'],'local_label':ca['label'],
      'max_score_difference':max(abs(next(t['score'] for t in ca['scores'] if t['label']==v['label'])-v['score']) for v in saved['scores'])})
   for mode,res,meta in [('first512',a,r),('chunks',ca,c)]:
    rows.append({'sample':sample['id'],'task':sample['task'],'repeat':sample['repetition'],
      'generator':sample['model'],'code_sha256':sha(sample['code']),'classifier':condition,
      'comments_removed':strip,'mode':mode,'prediction':res['label'],
      'input_tokens':meta['input_tokens'],'used_tokens':meta['used_tokens'],
      'chunk_count':meta['chunk_count'],'scores':res['scores']})
   (out/'paired_predictions.json').write_text(json.dumps(rows,indent=2))
  print(condition,sample['id'],flush=True)
summary=[]
for mode in ['first512','chunks']:
 for condition in ['050','100']:
  for strip in [False,True]:
   group=[r for r in rows if r['mode']==mode and r['classifier']==condition and r['comments_removed']==strip]
   comp=[r for r in group if not r['generator'].startswith('google/')]
   summary.append({'mode':mode,'classifier':condition,'comments_removed':strip,
     'id_matches':sum(r['prediction']==r['generator'].split('/')[-1] for r in comp),
     'comparable_n':len(comp),'gemini_matches':sum(r['prediction'].startswith('gemini') for r in group if r['generator'].startswith('google/')),
     'gpt_predictions':sum(r['prediction']=='gpt-4.1' for r in group),
     'multichunk':sum(r['chunk_count']>1 for r in group),
     'per_generator':{g:sum(r['prediction']==g.split('/')[-1] for r in group if r['generator']==g) for g in s['models'] if not g.startswith('google/')}})
(out/'summary.json').write_text(json.dumps(summary,indent=2))
(out/'verification.json').write_text(json.dumps({'source_run':s['id'],'paired_rows':len(rows),'saved_chunk_label_disagreements':sum(v['saved_label']!=v['local_label'] for v in verification),
 'maximum_score_difference':max(v['max_score_difference'] for v in verification),
 'chunk_verification':verification},indent=2))
print(json.dumps(summary,indent=2),flush=True)
