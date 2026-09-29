"""Comment-only ablation, preserving strings, chars, and token separation."""
import re
from time import perf_counter
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from .utils import ROOT,save_json
from .features import features
from .evaluate import evaluate
TOKEN=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*(?:\n|$)|/\*.*?\*/',re.S)
def remove_comments(code):
    # C translation phase 2: join escaped physical newlines before lexing.
    code=code.replace('\\\r\n','').replace('\\\n','')
    def replace(m):
        t=m.group()
        if t.startswith('//') or t.startswith('/*'): return ' '+('\n'*t.count('\n'))
        return t
    return TOKEN.sub(replace,code)

def ablate(df,model,winner,results):
    altered=df.copy(); altered['code']=altered.code.map(remove_comments)
    save_json('results/comment_removal.json',{'changed_programs':int((altered.code!=df.code).sum()),'removed_characters':int(df.code.str.len().sum()-altered.code.str.len().sum()),'normalization':'comments only; C line splicing applied; strings and character literals preserved; identifiers unchanged'})
    vectorizer,X=features(altered,'ablated'); estimator=clone(model)
    start=perf_counter(); estimator.fit(X['train'],altered.loc[altered.split=='train','label']); elapsed=perf_counter()-start
    joblib.dump(estimator,ROOT/'models/ablated_classifier.joblib',compress=3)
    row,pred=evaluate(estimator,X['test'],altered.loc[altered.split=='test','label'],sorted(df.label.unique()),'Ablated '+winner,'test')
    original=results[(results.model==winner)&(results.split=='test')].iloc[0]
    comparison=[]
    for metric in ('accuracy','macro_f1'):
        delta=row[metric]-original[metric]
        comparison.append({'metric':metric,'original':original[metric],'ablated':row[metric],'signed_difference':delta,'absolute_difference':abs(delta),'percentage_point_difference':100*delta,'relative_percentage_difference':100*delta/original[metric]})
    pd.DataFrame(comparison).to_csv(ROOT/'results/ablation_results.csv',index=False)
    save_json('results/ablation_run.json',{**row,'training_seconds':elapsed,'classifier':winner})
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)
    return pd.DataFrame(comparison)

def top_features(model,vectorizer):
    names=vectorizer.get_feature_names_out(); rows=[]
    if hasattr(model,'coef_'): weights=model.coef_; classes=model.classes_
    elif hasattr(model,'feature_log_prob_'): weights=model.feature_log_prob_; classes=model.classes_
    else: weights=model.feature_importances_[None,:]; classes=['global']
    for label,values in zip(classes,weights):
        for i in np.argsort(values)[-15:][::-1]: rows.append({'class':label,'feature':names[i],'feature_display':repr(names[i]),'weight':values[i]})
    pd.DataFrame(rows).to_csv(ROOT/'results/top_features.csv',index=False)
