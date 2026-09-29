"""Run the complete source-only benchmark from any working directory."""
import os
from pathlib import Path
root=Path(__file__).resolve().parent
(root/'tmp').mkdir(exist_ok=True)
for key in ('TEMP','TMP','TMPDIR','JOBLIB_TEMP_FOLDER'): os.environ[key]=str(root/'tmp')
os.environ['MPLCONFIGDIR']=str(root/'tmp/matplotlib')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
from src.inspect_dataset import inspect
from src.prepare_data import prepare
from src.features import features
from src.train_models import train
from src.ablation import ablate,top_features
from src.report import report
from src.utils import save_json
import platform,importlib.metadata,datetime

def main():
    df,stats=inspect(); df=prepare(df)
    vectorizer,X=features(df)
    fitted,winner,results=train(df,X)
    top_features(fitted[winner],vectorizer)
    comparison=ablate(df,fitted[winner],winner,results)
    save_json('results/environment.json',{'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packages':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','matplotlib','joblib','scipy')}})
    report(df,stats,winner,results,comparison)
    print('Complete: results, models, README.md and slides_content.md saved.',flush=True)
if __name__=='__main__': main()
