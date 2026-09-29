"""Equal shared inputs, fixed hyperparameters; selection uses validation only."""
from time import perf_counter
import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from .utils import ROOT,SEED,save_json
from .evaluate import evaluate

def estimators():
    return {'Naive Bayes':MultinomialNB(alpha=1.0),'Logistic Regression':LogisticRegression(max_iter=3000,C=1.0,solver='lbfgs',random_state=SEED),'Linear SVM':LinearSVC(C=1.0,dual='auto',max_iter=10000,random_state=SEED),'Random Forest':RandomForestClassifier(n_estimators=150,max_depth=40,min_samples_leaf=2,max_features='sqrt',n_jobs=4,random_state=SEED)}

def train(df,X):
    labels=sorted(df.label.unique()); ys={s:df.loc[df.split==s,'label'] for s in X}
    rows=[]; fitted={}
    for name,model in estimators().items():
        print(f'Training {name} ...',flush=True)
        start=perf_counter(); model.fit(X['train'],ys['train']); elapsed=perf_counter()-start
        path=ROOT/f'models/{name}.joblib'; joblib.dump(model,path,compress=3)
        for split in ('validation','test'):
            row,pred=evaluate(model,X[split],ys[split],labels,name,split)
            row.update(training_seconds=elapsed,model_MB=path.stat().st_size/1e6)
            rows.append(row)
            pd.DataFrame({'row_id':df.loc[df.split==split,'row_id'],'true':ys[split],'predicted':pred}).to_csv(ROOT/f'results/{name}_{split}_predictions.csv',index=False)
        fitted[name]=model
        pd.DataFrame(rows).to_csv(ROOT/'results/model_results.csv',index=False)
        print(f'{name}: validation F1={rows[-2]["macro_f1"]:.4f}, test F1={rows[-1]["macro_f1"]:.4f}, fit {elapsed:.1f}s',flush=True)
    results=pd.DataFrame(rows)
    winner=results[results.split=='validation'].sort_values(['macro_f1','model'],ascending=[False,True]).iloc[0]['model']
    dummy=DummyClassifier(strategy='most_frequent').fit(X['train'],ys['train'])
    baseline,_=evaluate(dummy,X['test'],ys['test'],labels,'Majority baseline','test')
    save_json('results/baselines.json',{'random_chance_accuracy':1/len(labels),'majority_training_class':dummy.classes_[dummy.class_prior_.argmax()],'majority_test':baseline})
    save_json('results/selection.json',{'winner':winner,'criterion':'Highest validation Macro-F1; test results never used for selection','parameters':fitted[winner].get_params()})
    return fitted,winner,results
