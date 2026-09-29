"""One shared training-only representation for the main comparison."""
from time import perf_counter
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import ROOT,save_json

def features(df,condition='original'):
    # sklearn's char analyzer normalizes repeated whitespace internally.
    # lowercase=False preserves C identifier case, unlike the sklearn default.
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=50000,sublinear_tf=True,lowercase=False,dtype=np.float32)
    start=perf_counter()
    matrices={'train':vectorizer.fit_transform(df.loc[df.split=='train','code'])}
    fit_seconds=perf_counter()-start
    for s in ('validation','test'): matrices[s]=vectorizer.transform(df.loc[df.split==s,'code'])
    path=ROOT/f'models/{condition}_tfidf.joblib'; joblib.dump(vectorizer,path,compress=3)
    save_json(f'results/{condition}_features.json',{'features':len(vectorizer.vocabulary_),'fit_transform_seconds':fit_seconds,'vectorizer_MB':path.stat().st_size/1e6,'train_shape':matrices['train'].shape,'train_nonzero':matrices['train'].nnz,'train_density':matrices['train'].nnz/(matrices['train'].shape[0]*matrices['train'].shape[1]),'settings':{'analyzer':'char','ngram_range':[3,5],'max_features':50000,'sublinear_tf':True,'lowercase':False,'dtype':'float32'}})
    print(f'{condition}: {len(vectorizer.vocabulary_):,} features, training feature extraction {fit_seconds:.1f}s',flush=True)
    return vectorizer,matrices
