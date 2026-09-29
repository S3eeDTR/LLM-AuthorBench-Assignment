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
    """One ablation condition, shared by all four original classifier settings."""
    from .train_models import estimators
    altered=df.copy(); altered['code']=altered.code.map(remove_comments)
    save_json('results/comment_removal.json',{'changed_programs':int((altered.code!=df.code).sum()),'removed_characters':int(df.code.str.len().sum()-altered.code.str.len().sum()),'normalization':'comments only; C line splicing applied; strings and character literals preserved; identifiers unchanged'})
    vectorizer,X=features(altered,'ablated')
    labels=sorted(df.label.unique()); rows=[]
    for name,base in estimators().items():
        estimator=clone(model if name==winner else base)
        original_model=joblib.load(ROOT/f'models/{name}.joblib')
        assert estimator.get_params()==original_model.get_params(), 'Ablation settings changed'
        print(f'Ablation: training {name} ...',flush=True)
        start=perf_counter(); estimator.fit(X['train'],altered.loc[altered.split=='train','label']); elapsed=perf_counter()-start
        path=ROOT/f'models/ablated_{name}.joblib'; joblib.dump(estimator,path,compress=3)
        if name==winner: joblib.dump(estimator,ROOT/'models/ablated_classifier.joblib',compress=3)
        for split in ('validation','test'):
            row,pred=evaluate(estimator,X[split],altered.loc[altered.split==split,'label'],labels,'Ablated '+name,split)
            row.update(model=name,training_seconds=elapsed,model_MB=path.stat().st_size/1e6)
            rows.append(row)
            pd.DataFrame({'row_id':altered.loc[altered.split==split,'row_id'],'true':altered.loc[altered.split==split,'label'],'predicted':pred}).to_csv(ROOT/f'results/Ablated {name}_{split}_predictions.csv',index=False)
        pd.DataFrame(rows).to_csv(ROOT/'results/ablation_model_results.csv',index=False)
        print(f'{name}: validation F1={rows[-2]["macro_f1"]:.4f}, test F1={rows[-1]["macro_f1"]:.4f}',flush=True)
    ablated=pd.DataFrame(rows)
    ranked=results[['model','split','accuracy','macro_f1']].merge(ablated[['model','split','accuracy','macro_f1']],on=['model','split'],suffixes=('_original','_ablated'),validate='one_to_one')
    for condition in ('original','ablated'):
        ranked[f'rank_{condition}']=ranked.groupby('split')[f'macro_f1_{condition}'].rank(ascending=False,method='min').astype(int)
    ranked['f1_change_pp']=100*(ranked.macro_f1_ablated-ranked.macro_f1_original)
    ranked.to_csv(ROOT/'results/ablation_ranking.csv',index=False)
    original=results[(results.model==winner)&(results.split=='test')].iloc[0]
    row=ablated[(ablated.model==winner)&(ablated.split=='test')].iloc[0]
    comparison=[]
    for metric in ('accuracy','macro_f1'):
        delta=row[metric]-original[metric]
        comparison.append({'metric':metric,'original':original[metric],'ablated':row[metric],'signed_difference':delta,'absolute_difference':abs(delta),'percentage_point_difference':100*delta,'relative_percentage_difference':100*delta/original[metric]})
    pd.DataFrame(comparison).to_csv(ROOT/'results/ablation_results.csv',index=False)
    save_json('results/ablation_run.json',{**row.to_dict(),'classifier':winner,'scope':'All four original algorithms, same common ablated features and settings'})
    print(ranked.to_string(index=False),flush=True)
    return pd.DataFrame(comparison)

def top_features(model,vectorizer):
    names=vectorizer.get_feature_names_out(); rows=[]
    if hasattr(model,'coef_'): weights=model.coef_; classes=model.classes_
    elif hasattr(model,'feature_log_prob_'): weights=model.feature_log_prob_; classes=model.classes_
    else: weights=model.feature_importances_[None,:]; classes=['global']
    for label,values in zip(classes,weights):
        for i in np.argsort(values)[-15:][::-1]: rows.append({'class':label,'feature':names[i],'feature_display':repr(names[i]),'weight':values[i]})
    pd.DataFrame(rows).to_csv(ROOT/'results/top_features.csv',index=False)
