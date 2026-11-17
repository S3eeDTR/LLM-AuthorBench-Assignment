"""Persistent six-task study. No API keys are stored in study files."""
from pathlib import Path
import csv, io, json, re, threading, uuid
from datetime import datetime, timezone
from generation import MODELS, generate

PROMPTS = [
('sorting','Write a complete C11 program that reads 10 integers, sorts them in ascending order using insertion sort, and prints the result. Check input errors. Use standard C libraries only. Return only C source code without Markdown fences.'),
('strings','Write a complete C11 program that reads one line of text and counts its vowels, consonants, digits, and whitespace characters. Use standard C libraries only. Return only C source code without Markdown fences.'),
('gcd',"Write a complete C11 program that reads two positive integers and computes their greatest common divisor using Euclid's algorithm. Reject invalid input. Use standard C libraries only. Return only C source code without Markdown fences."),
('recursion','Write a complete C11 program that recursively performs binary search on the sorted array {2, 5, 8, 12, 16, 23, 38, 56}. Read the target integer and print its index or report that it was not found. Return only C source code without Markdown fences.'),
('structures','Write a complete C11 program that stores three students using a struct containing a name and a score. Read their details and print the student with the highest score. Check input errors. Return only C source code without Markdown fences.'),
('memory','Write a complete C11 program that reads an integer n between 1 and 50, dynamically allocates an array of n integers, reads its values, and prints their sum. Handle input and allocation failures and free the memory. Return only C source code without Markdown fences.')]

class StudyRunner:
    def __init__(self, engine, root=None):
        self.engine=engine
        self.root=Path(root) if root else Path(__file__).resolve().parents[1]/'study_results'
        self.root.mkdir(exist_ok=True)
        self.thread=None
        self.active_id=None
        self.pause_event=threading.Event()
        self.lock=threading.Lock()

    def folder(self, run_id):
        if not isinstance(run_id,str) or not re.fullmatch(r'[a-f0-9]{32}',run_id): raise ValueError('Invalid study ID.')
        return self.root/run_id

    def read(self, run_id):
        state=json.loads((self.folder(run_id)/'study.json').read_text(encoding='utf-8'))
        if state['status']=='running' and not (self.active_id==run_id and self.thread and self.thread.is_alive()):
            state['status']='interrupted'
        return state

    def save(self, state):
        folder=self.folder(state['id']);folder.mkdir(exist_ok=True)
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        temporary=folder/'study.tmp'
        temporary.write_text(json.dumps(state,indent=2),encoding='utf-8')
        temporary.replace(folder/'study.json')
        rows=[{'sample':s['id'],'task':s['task'],'repetition':s['repetition'],
               'requested_model':s['model'],'returned_model':s.get('generation',{}).get('returned_model'),
               'provider':s.get('generation',{}).get('provider'),**a,
               'analysis_mode':a.get('analysis_mode','first512')}
              for s in state['samples'] for a in s['analyses']]
        keys=['sample','task','repetition','requested_model','returned_model','provider','classifier','comments_removed','analysis_mode','chunk_count','predicted_label','top_score','gpt_score','input_tokens','used_tokens','truncated','error']
        output=io.StringIO();writer=csv.DictWriter(output,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        csv_temp=folder/'comparison.tmp'
        csv_temp.write_text(output.getvalue(),encoding='utf-8')
        csv_temp.replace(folder/'comparison.csv')

    def start(self, data):
        local_only=data.get('analysis_only') is True
        requested_modes=data.get('analysis_modes',['first512'])
        if not isinstance(requested_modes,list) or not requested_modes or any(m not in ['first512','chunks'] for m in requested_modes):
            raise ValueError('Choose first512, chunks, or both coverage modes.')
        key=data.get('api_key','')
        if not local_only and (not isinstance(key,str) or not 10<=len(key.strip())<=500): raise ValueError('Enter your OpenRouter key.')
        with self.lock:
            if self.thread and self.thread.is_alive(): raise ValueError('A study is already running. Pause it first.')
            run_id=data.get('run_id')
            if local_only and not run_id: raise ValueError('Choose an existing saved study to retest.')
            if run_id:
                state=self.read(run_id)
                state['analysis_modes']=list(dict.fromkeys(state.get('analysis_modes',['first512'])+requested_modes))
                for sample in state['samples']:
                    if sample['generation_status']=='in_progress':
                        sample['generation_status']='uncertain'
                        sample['generation']={'error':'Interrupted request; may have been billed. Not automatically repeated.'}
            else:
                selected=data.get('models',[])
                if not isinstance(selected,list) or not selected or len(selected)!=len(set(selected)) or any(m not in MODELS for m in selected): raise ValueError('Choose supported models.')
                state={'id':uuid.uuid4().hex,'status':'running','created_at':datetime.now(timezone.utc).isoformat(),
                       'models':selected,'prompts':[{'task':t,'prompt':p} for t,p in PROMPTS],
                       'temperature':.7,'max_tokens':768,'repetitions':2,
                       'analysis_modes':list(dict.fromkeys(requested_modes)),'samples':[]}
                for index,(task,prompt) in enumerate(PROMPTS):
                    for repetition in [1,2]:
                        for model in selected:
                            state['samples'].append({'id':f'p{index+1}_r{repetition}_{len(state["samples"])+1}',
                                'task':task,'prompt':prompt,'repetition':repetition,'model':model,
                                'generation_status':'pending','analyses':[]})
            state['status']='running';self.save(state);self.pause_event.clear()
            self.active_id=state['id']
            self.thread=threading.Thread(target=self.work,args=(state,'' if local_only else key.strip(),local_only),daemon=True)
            self.thread.start()
            return {'run_id':state['id'],'folder':str(self.folder(state['id']))}

    def work(self,state,key,local_only=False):
        try:
            for sample in state['samples']:
                if local_only: break
                if self.pause_event.is_set(): break
                if sample['generation_status']!='pending': continue
                sample['generation_status']='in_progress';self.save(state)
                result=generate({'api_key':key,'model':sample['model'],'prompt':sample['prompt']})
                sample['generation']=result
                sample['generation_status']='failed' if result.get('error') else 'completed'
                if not result.get('error'):
                    code=re.sub(r'^```(?:c|C)?\s*\n([\s\S]*?)\n```$',r'\1',result['code'].strip())
                    sample['code']=code
                    (self.folder(state['id'])/(sample['id']+'.c')).write_text(code,encoding='utf-8')
                self.save(state)
                if result.get('error') in ['API key rejected.','Insufficient OpenRouter credits.']:
                    self.pause_event.set();break
            # Keep one classifier loaded across each pass to avoid repeated reloads.
            for condition in ['050','100']:
                for strip in [False,True]:
                    for mode in state.get('analysis_modes',['first512']):
                        for sample in state['samples']:
                            if self.pause_event.is_set(): break
                            if sample['generation_status']!='completed': continue
                            if any(a['classifier']==condition and a['comments_removed']==strip and a.get('analysis_mode','first512')==mode for a in sample['analyses']): continue
                            try:
                                result=self.engine.analyze(sample['code'],condition,strip,mode=mode);prediction=result['results'][0]
                                analysis={'classifier':condition,'comments_removed':strip,'analysis_mode':mode,
                                          'chunk_count':result.get('chunk_count',1),'aggregation':result.get('aggregation','single window'),
                                          'chunks':prediction.get('chunks',[]),'predicted_label':prediction['label'],
                                          'top_score':prediction['scores'][0]['score'],
                                          'gpt_score':next(s['score'] for s in prediction['scores'] if s['label']=='gpt-4.1'),
                                          'input_tokens':result['input_tokens'],'used_tokens':result['used_tokens'],
                                          'truncated':result['truncated'],'scores':prediction['scores']}
                            except Exception as error:
                                analysis={'classifier':condition,'comments_removed':strip,'analysis_mode':mode,'error':type(error).__name__}
                            sample['analyses'].append(analysis);self.save(state)
            state['status']='paused' if self.pause_event.is_set() else 'completed'
        except Exception as error:
            state['status']='interrupted';state['error']=type(error).__name__
        finally:
            self.save(state)
