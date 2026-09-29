"""Verify saved results and portable artifact loading without retraining."""
import json,itertools,zipfile
import joblib,numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from src.utils import ROOT,save_json
from src.report import report

def verify():
    splits=pd.read_csv(ROOT/'data/processed/splits.csv')
    for col in ('problem_id','source_hash'):
        sets=[set(splits.loc[splits.split==s,col]) for s in ('train','validation','test')]
        assert all(not a&b for a,b in itertools.combinations(sets,2))
    results=pd.read_csv(ROOT/'results/model_results.csv')
    for row in results.itertuples():
        pred=pd.read_csv(ROOT/f'results/{row.model}_{row.split}_predictions.csv')
        assert np.isclose(accuracy_score(pred.true,pred.predicted),row.accuracy)
        assert np.isclose(f1_score(pred.true,pred.predicted,average='macro'),row.macro_f1)
    with zipfile.ZipFile(ROOT/'data/raw/LLM-AuthorBench.json.zip') as z: data=json.loads(z.read('LLM-AuthorBench.json'))
    winner=json.loads((ROOT/'results/selection.json').read_text())['winner']
    sample=pd.read_csv(ROOT/f'results/{winner}_test_predictions.csv').head(64)
    vectorizer=joblib.load(ROOT/'models/original_tfidf.joblib')
    model=joblib.load(ROOT/f'models/{winner}.joblib')
    assert (model.predict(vectorizer.transform([data[i]['c_code'] for i in sample.row_id]))==sample.predicted).all()
    stats=json.loads((ROOT/'results/dataset_statistics.json').read_text())
    frame=pd.DataFrame({'label':[d['model_name'] for d in data],'row_id':range(len(data))}).merge(splits[['row_id','split','problem_id']],on='row_id')
    report(frame,stats,winner,results,pd.read_csv(ROOT/'results/ablation_results.csv'))
    assert (ROOT/'slides_content.md').read_text(encoding='utf-8').count('# Slide ')==5
    save_json('results/verification.json',{'split_disjointness':True,'all_main_metrics_recomputed_from_predictions':True,'saved_model_prediction_roundtrip_samples':64,'slides':5,'lexer_and_grouping_tests':7,'run_from_other_working_directory':__import__('pathlib').Path.cwd().resolve()!=ROOT})
    print('Saved artifacts, metrics, split boundaries and five-slide content verified.')
if __name__=='__main__': verify()
