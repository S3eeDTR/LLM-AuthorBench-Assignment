"""Group prompt variants conservatively, before any model fitting."""
import re,itertools
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from .utils import ROOT,SEED,digest,save_json

def canonical(prompt):
    s=prompt.lower()
    # Audited task aliases: deliberately broad families avoid paraphrase leakage.
    # These rules use prompt text only, never model labels or outcomes.
    families = [
        ('histogram', r'histogram from data'),
        ('password_strength', r'(?:strength.*password|password.*strength)'),
        ('number_guessing', r'number guessing game'),
        ('traveling_salesman', r'traveling salesman'),
        ('balanced_brackets', r'(?:balanced.*(?:brackets|parentheses)|(?:brackets|parentheses).*balanced)'),
        ('file_sha_hash', r'(?:sha.*hash of a file|sha.*hash.*file given)'),
        ('pipe_chat', r'(?:chat.*pipes|pipes.*chat)'),
        ('array_extrema', r'(?:largest and smallest|maximum and minimum).*array'),
        ('maze', r'maze'),
        ('chatbot', r'(?:chatbot|chats with the user)'),
        ('factorial', r'factorial'),
        ('web_server', r'(?:http server|web server)'),
        ('memory_allocator', r'custom memory allocator'),
        ('dice', r'(?:dice|die with unfair)'),
        ('traffic_light', r'traffic light'),
        ('sudoku', r'sudoku'),
        ('xor_encryption', r'(?:encrypt.*xor|xor.*encrypt)'),
        ('temperature_conversion', r'(?:celsius.*fahrenheit|fahrenheit.*celsius)'),
        ('huffman', r'huffman'),
        ('matrix_multiplication', r'(?:matrix multiplication|multiplies two.*matrices)'),
        ('basic_shell', r'(?:basic shell|simple shell)'),
        ('memory_leaks', r'memory leaks'),
        ('elevator', r'elevator'),
        ('rsa', r'\brsa\b'),
        ('calculator', r'(?:calculator|parse.*(?:arithmetic expression|evaluate the expression))'),
        ('binary_search_tree', r'(?:binary search tree|avl tree)'),
        ('sorting', r'(?:bubble sort|merge sort|quicksort|stalin sort|compare sorting)'),
        ('banking', r'(?:bank transactions|banking system|basic atm)'),
        ('histogram', r'histogram'),
        ('trie', r'inserts words into a trie'),
    ]
    for family, pattern in families:
        if re.search(pattern,s): return 'audited_family:'+family
    s=re.sub(r'https?://\S+', ' VALUE ', s)
    s=re.sub(r'(?<=for the word ).*?(?= and returns)', 'VALUE',s)
    s=re.sub(r'(?<=if the file name is ).*?(?=, then)', 'VALUE',s)
    s=re.sub(r"'[^']*'",' VALUE ',s)
    s=re.sub(r'\{[^{}]*\}|\[[^\[\]]*\]|\d+(?:\.\d+)?',' VALUE ',s)
    s=re.sub(r'\b(write|create|make|build|generate)\b','create',s)
    s=re.sub(r'\bfunction\b','program',s)
    return ' '.join(s.split())

def prepare(df):
    keys=sorted({canonical(s) for s in df.prompt})
    parent=list(range(len(keys)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    edges=[]
    for i,a in enumerate(keys):
        for j in range(i):
            b=keys[j]
            # Conservative union of near-identical task descriptions, including
            # shortened/expanded variants from different generator versions.
            ratio=SequenceMatcher(None,a,b,autojunk=False).ratio()
            if ratio>=0.84 or (min(len(a),len(b))>=45 and (a.rstrip('.') in b or b.rstrip('.') in a)):
                parent[find(i)]=find(j); edges.append((a,b,ratio))
    mapping={s:digest(keys[find(i)])[:16] for i,s in enumerate(keys)}
    df=df.copy(); df['problem_id']=df.prompt.map(lambda s:mapping[canonical(s)])
    groups=np.array(sorted(df.problem_id.unique())); np.random.default_rng(SEED).shuffle(groups)
    n=len(groups); a=int(0.7*n); b=a+int(0.15*n)
    assignments={g:split for part,split in ((groups[:a],'train'),(groups[a:b],'validation'),(groups[b:],'test')) for g in part}
    df['split']=df.problem_id.map(assignments)
    for field in ('problem_id','prompt','source_hash'):
        sets=[set(df.loc[df.split==s,field]) for s in ('train','validation','test')]
        assert all(not x.intersection(y) for x,y in itertools.combinations(sets,2)),field
    assert all(set(df.loc[df.split==s,'label'])==set(df.label) for s in ('train','validation','test'))
    df[['row_id','source_hash','problem_id','split','label']].to_csv(ROOT/'data/processed/splits.csv',index=False)
    df[['prompt','problem_id']].drop_duplicates().sort_values(['problem_id','prompt']).to_csv(ROOT/'results/problem_groups.csv',index=False)
    pd.DataFrame(edges,columns=['prompt_a','prompt_b','similarity']).to_csv(ROOT/'results/grouping_edges.csv',index=False)
    df.groupby('problem_id').agg(samples=('row_id','size'),exact_prompts=('prompt','nunique')).to_csv(ROOT/'results/samples_per_problem.csv')
    summary=df.groupby('split').agg(samples=('row_id','size'),problems=('problem_id','nunique')).reindex(['train','validation','test'])
    summary.to_csv(ROOT/'results/split_summary.csv')
    pd.crosstab(df.split,df.label).to_csv(ROOT/'results/split_class_distribution.csv')
    save_json('results/split_metadata.json',{'seed':SEED,'canonical_prompts':len(keys),'inferred_problem_families':n,'grouping':'manually audited semantic aliases; normalize numeric/quoted/list/placeholder parameters and instruction verbs; union similarity >=0.84 and substantial substring variants','limitation':'No authoritative task IDs. Assertions prove separation of inferred families and exact prompts, not all possible semantic equivalences. Grouping is label-independent and deliberately conservative.'})
    print(summary.to_string(),flush=True); print('Leakage check PASSED',flush=True)
    return df
