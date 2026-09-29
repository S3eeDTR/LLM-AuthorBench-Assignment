"""Extend the ablation using existing original-run artifacts, without retraining originals."""
import os
from pathlib import Path
root=Path(__file__).resolve().parent
for key in ('TEMP','TMP','TMPDIR','JOBLIB_TEMP_FOLDER'): os.environ[key]=str(root/'tmp')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
os.environ['MPLCONFIGDIR']=str(root/'tmp/matplotlib')
import json,joblib,pandas as pd
from src.inspect_dataset import inspect
from src.ablation import ablate
from src.report import report

def main():
    df,stats=inspect(); splits=pd.read_csv(root/'data/processed/splits.csv')
    assert len(splits)==len(df) and set(splits.source_hash)==set(df.source_hash)
    df=df.merge(splits[['row_id','source_hash','problem_id','split']],on=['row_id','source_hash'],validate='one_to_one')
    for split in ('train','validation','test'):
        assert not set(df.loc[df.split==split,'problem_id']) & set(df.loc[df.split!=split,'problem_id'])
    winner=json.loads((root/'results/selection.json').read_text())['winner']
    results=pd.read_csv(root/'results/model_results.csv')
    comparison=ablate(df,joblib.load(root/f'models/{winner}.joblib'),winner,results)
    report(df,stats,winner,results,comparison)
if __name__=='__main__': main()
