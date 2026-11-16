from pathlib import Path
import hashlib, json, re, zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parent
# Colab keeps data beside the scripts; the repository reuses Assignment 1 data.
DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT.parent / "data"
LABELS = sorted(['claude-3.5-haiku','deepseek-chat','gemini-2.5-flash-preview-05-20','gpt-4.1','llama-3.3-70b-instruct'])
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*(?:\n|$)|/\*.*?\*/', re.S)

def remove_comments(code):
    code = code.replace('\\\r\n','').replace('\\\n','')
    return TOKEN.sub(lambda m: ' ' + '\n'*m.group().count('\n') if m.group().startswith(('//','/*')) else m.group(), code)

def load_data():
    with zipfile.ZipFile(DATA_DIR/'LLM-AuthorBench.json.zip') as z:
        df = pd.DataFrame(json.loads(z.read('LLM-AuthorBench.json')))
    df = df.rename(columns={'c_code':'code','model_name':'label'})
    df['row_id'] = range(len(df))
    df = df[df.label.isin(LABELS)].copy()
    df['source_hash'] = df.code.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    splits = pd.read_csv(DATA_DIR/'splits.csv')
    df = df.merge(splits,on=['row_id','label','source_hash'],validate='one_to_one')
    assert len(df)==20000 and not df.source_hash.duplicated().any()
    assert df.groupby('split').size().to_dict()=={'train':13150,'validation':2978,'test':3872}
    assert df.groupby('label').size().eq(4000).all()
    for part in ['train','validation','test']:
        for column in ['problem_id','prompt','source_hash']:
            assert set(df.loc[df.split==part,column]).isdisjoint(df.loc[df.split!=part,column])
    return df
