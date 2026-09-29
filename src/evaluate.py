"""Metrics use every class, including classes with zero predictions."""
from time import perf_counter
import pandas as pd
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix
from .utils import ROOT

def evaluate(model,X,y,labels,name,split):
    start=perf_counter(); predictions=model.predict(X); seconds=perf_counter()-start
    p,r,f,_=precision_recall_fscore_support(y,predictions,labels=labels,average='macro',zero_division=0)
    pd.DataFrame(classification_report(y,predictions,labels=labels,output_dict=True,zero_division=0)).T.to_csv(ROOT/f'results/{name}_{split}_per_class.csv')
    pd.DataFrame(confusion_matrix(y,predictions,labels=labels),index=labels,columns=labels).to_csv(ROOT/f'results/confusion_matrices/{name}_{split}.csv')
    return {'model':name,'split':split,'accuracy':accuracy_score(y,predictions),'macro_f1':f,'macro_precision':p,'macro_recall':r,'inference_seconds':seconds},predictions
