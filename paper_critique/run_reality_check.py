"""CPU reality check of the public classical baseline; no neural claims."""
from pathlib import Path
import json, time, hashlib, importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from project_utils import ROOT, DATA_DIR, LABELS, load_data, remove_comments

def scores(y,p):
    return {'accuracy':accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0),'weighted_f1':f1_score(y,p,average='weighted',zero_division=0)}

def main():
    df=load_data(); out=ROOT/'results/reality_check'; out.mkdir(parents=True,exist_ok=True)
    metrics=[]; audit=[]
    train=df[df.split=='train']; test=df[df.split=='test']
    # Match the published notebook's seeded permutation before stratified splitting.
    perm=np.random.RandomState(42).choice(len(df),size=len(df),replace=False)
    random_train,random_test=train_test_split(df.iloc[perm],test_size=.2,random_state=42,stratify=df.iloc[perm].label)
    for condition,tr,te,fullfit in [
        ('grouped_train_fit',train,test,False),
        ('random_train_fit',random_train,random_test,False),
        ('random_full_fit_DIAGNOSTIC',random_train,random_test,True)]:
        vector=TfidfVectorizer(max_features=400,token_pattern=r'(?u)\b\w+\b')
        vector.fit(df.code if fullfit else tr.code)
        model=LinearSVC(max_iter=2000,random_state=42)
        started=time.perf_counter(); model.fit(vector.transform(tr.code),tr.label); elapsed=time.perf_counter()-started
        pred=model.predict(vector.transform(te.code))
        metrics.append({'condition':condition,'model':'Authors word-400 Linear SVM','split':'test','training_seconds':elapsed,**scores(te.label,pred)})
        pd.DataFrame({'row_id':te.row_id,'true':te.label,'predicted':pred}).to_csv(out/f'{condition}_predictions.csv',index=False)
        pd.DataFrame(classification_report(te.label,pred,labels=LABELS,output_dict=True,zero_division=0)).T.to_csv(out/f'{condition}_class_scores.csv')
        overlap=set(tr.problem_id)&set(te.problem_id)
        audit.append({'condition':condition,'train_programs':len(tr),'test_programs':len(te),'shared_inferred_task_groups':len(overlap),'test_fraction_with_group_in_train':float(te.problem_id.isin(set(tr.problem_id)).mean()),'test_prompts_seen_in_train_fraction':float(te.prompt.isin(set(tr.prompt)).mean()),'vocabulary_fit_includes_test':fullfit})
        if condition=='grouped_train_fit':
            for part in ['validation','test']:
                rows=df[df.split==part]
                for variant,text in [('original',rows.code),('comments_removed_at_test',rows.code.map(remove_comments))]:
                    p=model.predict(vector.transform(text))
                    metrics.append({'condition':f'grouped_{variant}','model':'Authors word-400 Linear SVM','split':part,'training_seconds':elapsed,**scores(rows.label,p)})
    # Reuse trusted local Assignment 1 models when present; otherwise reproduce
    # their fixed configuration so a fresh clone does not need uploaded weights.
    model_dir=ROOT.parent/'models/experiment_1_original'
    reused=(model_dir/'vectorizer.joblib').exists() and (model_dir/'Linear SVM.joblib').exists()
    if reused:
        v=joblib.load(model_dir/'vectorizer.joblib')
        m=joblib.load(model_dir/'Linear SVM.joblib')
    else:
        v=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=50000,
                         sublinear_tf=True,lowercase=False,dtype=np.float32)
        x=v.fit_transform(train.code)
        m=LinearSVC(C=1.0,dual='auto',max_iter=10000,random_state=42)
        m.fit(x,train.label)
    print('Assignment 1 baseline:', 'reused trusted local artifacts' if reused else 'retrained fixed configuration')
    for part in ['validation','test']:
        rows=df[df.split==part]; clean=m.predict(v.transform(rows.code)); stripped=m.predict(v.transform(rows.code.map(remove_comments)))
        if part=='test' and reused: assert abs(accuracy_score(rows.label,clean)-0.7975206611570248)<1e-12
        for variant,p in [('original',clean),('comments_removed_at_test',stripped)]:
            changed=rows.code.ne(rows.code.map(remove_comments)).to_numpy()
            correct=(clean==rows.label.to_numpy()); eligible=correct & changed
            record={'condition':f'grouped_{variant}','model':'Assignment 1 char TF-IDF Linear SVM','split':part,**scores(rows.label,p)}
            if variant!='original':
                record.update({'attack_success_rate_correct_changed':float((stripped[eligible]!=rows.label.to_numpy()[eligible]).mean()),'eligible_correct_changed':int(eligible.sum())})
            metrics.append(record)
            pd.DataFrame({'row_id':rows.row_id,'true':rows.label,'predicted':p}).to_csv(out/f'svm_{part}_{variant}_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'split_audit.csv',index=False)
    (out/'environment.json').write_text(json.dumps({'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scikit-learn','joblib']},'seed':42,'assignment1_baseline_mode':'reused' if reused else 'retrained','data_sha256':hashlib.sha256((DATA_DIR/'LLM-AuthorBench.json.zip').read_bytes()).hexdigest(),'split_sha256':hashlib.sha256((DATA_DIR/'splits.csv').read_bytes()).hexdigest()},indent=2))
    print(pd.DataFrame(metrics).to_string(index=False)); print(pd.DataFrame(audit).to_string(index=False))

if __name__=='__main__': main()
