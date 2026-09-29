"""Shared portable paths, provenance, and serialization."""
from pathlib import Path
import hashlib,json,os
ROOT=Path(__file__).resolve().parents[1]
SEED=42
for folder in ('data/raw','data/processed','models','results/figures','results/confusion_matrices','tmp'):
    (ROOT/folder).mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/matplotlib')
os.environ['JOBLIB_TEMP_FOLDER']=str(ROOT/'tmp')
def digest(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def save_json(path,obj): (ROOT/path).write_text(json.dumps(obj,indent=2,ensure_ascii=True,default=str),encoding='utf-8')
