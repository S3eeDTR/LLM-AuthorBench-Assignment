"""Read saved experiment evidence without executing or extracting uploaded files."""
from pathlib import Path
import base64, collections, csv, io, json, zipfile, itertools

LABELS=['claude-3.5-haiku','deepseek-chat','gemini-2.5-flash-preview-05-20','gpt-4.1','llama-3.3-70b-instruct']
MODEL_LABELS={'anthropic/claude-3.5-haiku':LABELS[0],'deepseek/deepseek-chat':LABELS[1],
 'google/gemini-2.5-flash-preview-05-20':LABELS[2],'google/gemini-2.5-flash':LABELS[2],
 'openai/gpt-4.1':LABELS[3],'meta-llama/llama-3.3-70b-instruct':LABELS[4]}

def table(content): return list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
def stats(rows, true='true', predicted='predicted'):
    matrix=[[sum(r[true]==a and r[predicted]==b for r in rows) for b in LABELS] for a in LABELS]
    classes=[]
    for i,label in enumerate(LABELS):
        tp=matrix[i][i];support=sum(matrix[i]);selected=sum(row[i] for row in matrix)
        precision=tp/selected if selected else 0; recall=tp/support if support else 0
        classes.append({'label':label,'support':support,'precision':precision,'recall':recall,
                        'f1':2*precision*recall/(precision+recall) if precision+recall else 0})
    return {'n':len(rows),'accuracy':sum(matrix[i][i] for i in range(5))/len(rows) if rows else None,
            'macro_f1':sum(c['f1'] for c in classes)/5,'classes':classes,'matrix':matrix}

def checked_zip(content):
    try: archive=zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile: raise ValueError('Upload a valid results ZIP file.')
    entries=archive.infolist()
    if len(entries)>300 or sum(e.file_size for e in entries)>50_000_000:
        raise ValueError('This evidence ZIP is too large. Upload a results ZIP without checkpoints.')
    return archive

def benchmark(content):
    with checked_zip(content) as z:
        names=z.namelist(); metadata=json.loads(z.read('reference/baseline_metadata.json'))
        output=[]
        for model,key in [('SVM','svm'),('CodeT5 0%','codet5')]:
            metrics={variant:stats(table(z.read(f'reference/{key}_{variant}_predictions.csv'))) for variant in ['original','stripped']}
            output.append({'model':model,**metrics,'training_seconds':metadata[key]['cost']['training_seconds'],
                           'timing_scope':metadata[key]['cost']['timing_scope'],'parameters':metadata[key]['cost']['parameter_count']})
        for condition in ['050','100']:
            prefix=f'results/remove_{condition}_seed42/'
            complete=json.loads(z.read(prefix+'training_complete.json'))
            metrics={variant:stats(table(z.read(prefix+variant+'_predictions.csv'))) for variant in ['original','stripped']}
            history=table(z.read(prefix+'training_history.csv'))
            output.append({'model':f'CodeT5 {int(condition)}%',**metrics,'history':history,
                'best_epoch':complete['best_epoch'],'training_seconds':complete['training_validation_checkpoint_seconds'],
                'timing_scope':'GPU training + validation + checkpoint writes','parameters':complete['parameter_count'],
                'size_mb':complete['checkpoint_bytes']/1e6})
        return {'models':output,'labels':LABELS,'note':'Fixed held-out benchmark: 3,872 C programs, five generators, one training seed.'}

def fresh(state):
    samples=state['samples']
    if not isinstance(samples,list) or len(samples)>1000: raise ValueError('Expected at most 1,000 study samples.')
    conditions=[]
    modes=list(dict.fromkeys(a.get('analysis_mode','first512') for s in samples for a in s.get('analyses',[]))) or ['first512']
    for classifier,strip,mode in itertools.product(['050','100'],[False,True],modes):
        pairs=[(s,a) for s in samples for a in s.get('analyses',[]) if a.get('classifier')==classifier and a.get('comments_removed')==strip and a.get('analysis_mode','first512')==mode]
        good=[(s,a) for s,a in pairs if not a.get('error') and a.get('predicted_label') in LABELS]
        scored=[(s,a) for s,a in good if s['model'] in MODEL_LABELS and s['model']!='google/gemini-2.5-flash']
        gemini=[(s,a) for s,a in good if s['model']=='google/gemini-2.5-flash']
        def grouped(column):
            groups=[]
            for name in dict.fromkeys(s.get(column,'unknown') for s in samples):
                group=[(s,a) for s,a in good if s.get(column,'unknown')==name]
                groups.append({'name':name,'n':len(group),'matches':sum(a['predicted_label']==MODEL_LABELS.get(s['model']) for s,a in group),
                               'predictions':dict(collections.Counter(a['predicted_label'] for s,a in group))})
            return groups
        matrix_rows=[{'true':MODEL_LABELS[s['model']],'predicted':a['predicted_label']} for s,a in good if s['model'] in MODEL_LABELS]
        conditions.append({'analysis_mode':mode,'classifier':classifier,'comments_removed':strip,'n':len(good),'errors':len(pairs)-len(good),
            'id_n':len(scored),'id_matches':sum(a['predicted_label']==MODEL_LABELS[s['model']] for s,a in scored),
            'gemini_n':len(gemini),'gemini_matches':sum(a['predicted_label']==LABELS[2] for s,a in gemini),
            'gpt_predictions':sum(a['predicted_label']=='gpt-4.1' for s,a in good),
            'truncated':sum(bool(a.get('truncated')) for s,a in good),
            'predictions':dict(collections.Counter(a['predicted_label'] for s,a in good)),
            'by_model':grouped('model'),'by_task':grouped('task'),'matrix':stats(matrix_rows)['matrix']})
    return {'id':state.get('id','uploaded'),'status':state.get('status','unknown'),'samples':len(samples),
            'generated':sum(s.get('generation_status')=='completed' for s in samples),
            'generation_failures':sum(s.get('generation_status') in ['failed','uncertain'] for s in samples),
            'generation_output_limits':sum(s.get('generation',{}).get('finish_reason')=='length' for s in samples),
            'conditions':conditions,'labels':LABELS,
            'note':'Exploratory study. Repeated classifications reuse the same programs. Newer Gemini family matches are separate from comparable recorded-ID matches.'}

class ResultStore:
    def __init__(self, root=None):
        self.root=Path(root) if root else Path(__file__).resolve().parents[1]/'evidence'
        self.root.mkdir(exist_ok=True)

    def read(self):
        output={}
        for name,parser in [('course_project_evidence.zip',benchmark),('full_study_results.zip',None)]:
            path=self.root/name
            if not path.exists(): continue
            try:
                if parser: output['benchmark']=parser(path.read_bytes())
                else:
                    with checked_zip(path.read_bytes()) as archive: output['fresh']=fresh(json.loads(archive.read('study.json')))
            except (ValueError,KeyError,OSError,zipfile.BadZipFile): output.setdefault('warnings',[]).append('Could not read '+name)
        return output

    def upload(self,data):
        try: content=base64.b64decode(data.get('data',''),validate=True)
        except Exception: raise ValueError('Invalid upload data.')
        if not content or len(content)>20_000_000: raise ValueError('Upload an evidence ZIP smaller than 20 MB, without model weights.')
        with checked_zip(content) as archive:
            if 'study.json' in archive.namelist():
                fresh(json.loads(archive.read('study.json')));name='full_study_results.zip'
            else: benchmark(content);name='course_project_evidence.zip'
        target=self.root/name
        if target.exists():
            from datetime import datetime,timezone
            backup=self.root/'previous';backup.mkdir(exist_ok=True)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            (backup/(stamp+'_'+name)).write_bytes(target.read_bytes())
        temporary=self.root/(name+'.tmp');temporary.write_bytes(content);temporary.replace(target)
        return self.read()
