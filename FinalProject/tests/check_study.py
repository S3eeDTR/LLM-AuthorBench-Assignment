import sys,json,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'webapp'))
import study
calls=[]
def fake_generate(data):
    calls.append((data['model'],data['prompt']))
    return {'code':'int main(void){return 0;}','returned_model':data['model'],'requested_model':data['model']}
class FakeEngine:
    def analyze(self,code,condition,strip,mode='first512'):
        return {'results':[{'label':'gpt-4.1','scores':[{'label':'gpt-4.1','score':1.}]}],
                'input_tokens':10,'used_tokens':10,'truncated':False}
with tempfile.TemporaryDirectory() as folder:
    runner=study.StudyRunner(FakeEngine(),folder)
    with patch.object(study,'generate',fake_generate):
        result=runner.start({'api_key':'secret-not-to-be-saved','models':list(study.MODELS)[:4]})
        runner.thread.join(30)
        state=runner.read(result['run_id'])
        assert state['status']=='completed'
        assert len(calls)==48 and len(state['samples'])==48
        assert sum(len(s['analyses']) for s in state['samples'])==192
        runner.start({'run_id':result['run_id'],'api_key':'secret-not-to-be-saved'})
        runner.thread.join(30)
        assert len(calls)==48
        runner.start({'run_id':result['run_id'],'analysis_only':True,'analysis_modes':['chunks']})
        runner.thread.join(30)
        chunk_state=runner.read(result['run_id'])
        assert len(calls)==48
        assert sum(len(s['analyses']) for s in chunk_state['samples'])==384
        assert sum(a.get('analysis_mode')=='chunks' for s in chunk_state['samples'] for a in s['analyses'])==192
        from results import fresh
        summaries=fresh(chunk_state)['conditions']
        assert len(summaries)==8 and sum(c['n'] for c in summaries)==384
    for file in Path(folder).rglob('*'):
        if file.is_file(): assert b'secret-not-to-be-saved' not in file.read_bytes()
    # Interrupted potentially billed generations must not be automatically repeated.
    state['samples'][0]['generation_status']='in_progress'
    state['samples'][0]['generation']={}
    runner.save(state)
    with patch.object(study,'generate',fake_generate):
        runner.start({'run_id':state['id'],'api_key':'secret-not-to-be-saved'})
        runner.thread.join(30)
    assert len(calls)==48
    assert runner.read(state['id'])['samples'][0]['generation_status']=='uncertain'
print('Passed: 48 mocked generations; 192 classifications per coverage mode; 384 kept separate; key-free saved-code retesting; no duplicate generations or saved key. No paid calls made.')
