"""Optional independent audit against the upstream prompt generator; never execute it."""
import ast,json,re,urllib.request
import pandas as pd
from .utils import ROOT,save_json
from .inspect_dataset import COMMIT

def audit():
    path=ROOT/'data/raw/dataset_creator.ipynb'
    if not path.exists():
        urllib.request.urlretrieve(f'https://raw.githubusercontent.com/LLMauthorbench/LLMauthorbench/{COMMIT}/scripts/1_DATASET_CREATOR_google_colab.ipynb',path)
    notebook=json.loads(path.read_text(encoding='utf-8-sig'))
    source='\n'.join(''.join(c['source']) for c in notebook['cells'])
    templates=ast.literal_eval(re.search(r'c_dynamic_prompts\s*=\s*(\[.*?\n\])',source,re.S)[1])
    groups=pd.read_csv(ROOT/'results/problem_groups.csv')
    def base(s):
        return re.sub(r'\bfunction\b','program',re.sub(r'\b(write|create|make|build|generate)\b','create',s.lower()))
    texts=groups.prompt.map(base); rows=[]
    for i,t in enumerate(templates):
        parts=re.split(r'(\{[a-zA-Z_][a-zA-Z_0-9]*\})',base(t))
        pattern=re.compile('^'+''.join('.+?' if re.fullmatch(r'\{[a-zA-Z_][a-zA-Z_0-9]*\}',v) else re.escape(v) for v in parts)+'$')
        hits=groups[texts.map(lambda p:bool(pattern.match(p)))]
        rows.append({'template_index':i,'matched_exact_prompts':len(hits),'inferred_groups':hits.problem_id.nunique()})
    pd.DataFrame(rows).to_csv(ROOT/'results/upstream_template_audit.csv',index=False)
    violations=sum(r['inferred_groups']>1 for r in rows)
    save_json('results/upstream_template_audit.json',{'templates_inspected':len(rows),'templates_with_matches':sum(r['matched_exact_prompts']>0 for r in rows),'templates_spanning_multiple_groups':violations,'scope':'Exact template regex with wildcard parameters and normalized instruction verbs/program-function wording; not proof covering arbitrary semantic paraphrases.'})
    assert violations==0,'Upstream task template split across inferred groups'
    print('Upstream template audit PASSED',flush=True)
if __name__=='__main__': audit()
