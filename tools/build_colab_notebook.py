"""Build the self-contained, unexecuted Colab notebook from reviewed cells."""
from pathlib import Path
import json,uuid
root=Path(__file__).resolve().parents[1]
cells=[]
def md(s): cells.append({'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},'source':s.strip().splitlines(keepends=True)})
def code(s): cells.append({'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{},'execution_count':None,'outputs':[],'source':s.strip().splitlines(keepends=True)})
md(r'''
# CodeT5-Authorship on our eight-class assignment split

**Run this later in Google Colab with a GPU. No neural results have been generated yet.**

This notebook adapts the authors' architecture: the pretrained `Salesforce/codet5p-770m` **encoder**, first-token embedding, a 768-unit linear layer, GELU, dropout 0.2, and a new eight-class output layer. It does **not** load an attribution checkpoint trained on this dataset. The original encoder-decoder checkpoint name is 770M; the encoder-only classifier has fewer parameters, counted below.

1. Choose **Runtime → Change runtime type → GPU**.
2. Review the configuration, especially comment ablation, checkpoint storage and token limit.
3. Run the cells in order. Training uses ten epochs by default and may exceed a Colab session. Actual runtime depends on the GPU. A smaller smoke test checks execution but cannot produce reportable scores.
4. Download the small results archive at the end. Large model/checkpoint files stay in the selected output directory.

The default experiment uses all eight labels and our saved problem-family split. It is an **architecture adaptation, not a reproduction of the paper's five-class protocol**. Code tokenization, truncation, pretraining and GPU compute differ from the four TF-IDF systems, so this is an additional whole-system comparison.

Sources: [paper](https://arxiv.org/abs/2506.17323), [released training notebook](https://github.com/LLMauthorbench/LLMauthorbench/blob/6a1c2ac173c774cfbd012f4c81e8d0a51bb61eb7/scripts/5_CodeT5-Authorship_5-class_google_colab.ipynb), [base model](https://huggingface.co/Salesforce/codet5p-770m), [our repository](https://github.com/S3eeDTR/LLM-AuthorBench-Assignment).

The dataset research-use licence remains unverified. The base model's listed BSD-3-Clause licence does not establish the dataset's licence. No paper scores are used as our experimental outputs.
''')
md('''## 1. Configuration
The optional comment-removal run doubles the number of training runs. Keep it enabled if you want to test the explanation under the same ablation as our four baselines. A100-class BF16-capable hardware is useful; the notebook also permits other CUDA GPUs, using FP32 when BF16 is unavailable rather than unsafe FP16 for T5. Memory and speed are not guaranteed on every Colab GPU.

`MAX_LENGTH=512` follows the paper's results table. Their released notebook instead uses 1024. This discrepancy is explicit, not silently treated as exact reproduction. Effective training batch size is 32, matching their 4 × 8 accumulation setup. Our microbatch is one for memory safety, with gradient checkpointing.

Optional Drive checkpoints survive runtime resets; mounting asks you for permission in Colab. Without Drive, download outputs before the runtime disappears. Do not change settings inside an existing run name.''')
code(r'''
from pathlib import Path
import os, json, hashlib, sys, subprocess, time, urllib.request, zipfile, platform, gc
SEED = 42
MAX_LENGTH = 512              # paper table: 512; released notebook: 1024
EPOCHS = 10
MICROBATCH = 1
GRAD_ACCUM = 32               # effective batch size = 32
LEARNING_RATE = 5e-5
RUN_COMMENT_ABLATION = True   # False trains original code only
SMOKE_TEST = False            # True: tiny execution check, never an assignment result
USE_GOOGLE_DRIVE = False      # True enables persistent output/checkpoints
RESUME = True
RUN_NAME = 'codet5p770m-eight-class-512-seed42'
BASE_MODEL = 'Salesforce/codet5p-770m'
MODEL_REVISION = '3f7cc6e80aee6612f41253b5ee28f09f86faa35f'
BASELINE_COMMIT = '33b42098d3bfdeddd8d94c77326a22fb8847be0a'
DATASET_COMMIT = '6a1c2ac173c774cfbd012f4c81e8d0a51bb61eb7'
WORKSPACE = Path.cwd() / 'LLM-AuthorBench-Colab'
WORKSPACE.mkdir(exist_ok=True)
for key,folder in [('HF_HOME','cache/huggingface'),('TORCH_HOME','cache/torch'),('MPLCONFIGDIR','cache/matplotlib'),('TMPDIR','tmp'),('PIP_CACHE_DIR','cache/pip')]:
    target=WORKSPACE/folder; target.mkdir(parents=True,exist_ok=True); os.environ[key]=str(target)
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
OUTPUT_BASE=WORKSPACE/'runs'
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUTPUT_BASE=Path('/content/drive/MyDrive/LLM-AuthorBench-Colab/runs')
RUN_ROOT=OUTPUT_BASE/(RUN_NAME+('-SMOKE-NOT-REPORTABLE' if SMOKE_TEST else ''))
RUN_ROOT.mkdir(parents=True,exist_ok=True)
print('Output:',RUN_ROOT)
print('Conditions:', ['original','comments_removed'] if RUN_COMMENT_ABLATION else ['original'])
''')
md('''## 2. Install notebook dependencies
This uses Colab's existing CUDA-enabled PyTorch. It does not install our workstation's package pins or alter its environment. Run installation before importing these packages. If Colab requests a runtime restart after installation, restart and rerun from configuration. Versions and hardware are recorded with the results.''')
code(r'''
subprocess.check_call([sys.executable,'-m','pip','install','--quiet','--no-cache-dir',
    'transformers==4.57.1','accelerate==1.10.1','sentencepiece==0.2.1','safetensors==0.6.2',
    'scikit-learn>=1.5,<2','pandas>=2.2,<4','matplotlib>=3.9,<4'])
import importlib.metadata
import numpy as np
import pandas as pd
import torch
from packaging.version import Version
assert Version(torch.__version__.split('+')[0])>=Version('2.6'), 'Use a current Colab runtime with torch >=2.6 for safe pretrained .bin loading.'
assert torch.cuda.is_available(), 'Select a GPU runtime before continuing.'
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5EncoderModel, Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
set_seed(SEED)
torch.backends.cudnn.benchmark=False
USE_BF16=torch.cuda.is_bf16_supported()
ENVIRONMENT={'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),
             'gpu_memory_GB':torch.cuda.get_device_properties(0).total_memory/1e9,
             'precision':'bf16 mixed precision' if USE_BF16 else 'float32',
             'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','numpy','pandas','scikit-learn','safetensors']}}
print(json.dumps(ENVIRONMENT,indent=2))
(RUN_ROOT/'environment.json').write_text(json.dumps(ENVIRONMENT,indent=2))
''')
md('''## 3. Download the exact dataset, split and baseline results
Downloads are pinned to recorded versions. Hash checks stop on changed bytes. The split and metrics come from the completed four-model experiment, not from a new random partition. The raw archive remains local to this Colab workspace.''')
code(r'''
INPUTS=WORKSPACE/'inputs'; INPUTS.mkdir(exist_ok=True)
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def fetch(url,name,expected):
    path=INPUTS/name
    if not path.exists():
        temp=path.with_suffix(path.suffix+'.part')
        urllib.request.urlretrieve(url,temp)
        assert sha256(temp)==expected, f'Download checksum mismatch: {name}'
        temp.replace(path)
    assert sha256(path)==expected, f'Input checksum mismatch: {name}'
    return path
base=f'https://raw.githubusercontent.com/S3eeDTR/LLM-AuthorBench-Assignment/{BASELINE_COMMIT}'
archive=fetch(f'https://raw.githubusercontent.com/LLMauthorbench/LLMauthorbench/{DATASET_COMMIT}/LLM-AuthorBench.json.zip',
              'LLM-AuthorBench.json.zip','e24399b7b05b812c5a1148ef44245348859eaa74b1140523cb695f984337f24c')
split_path=fetch(base+'/data/processed/splits.csv','splits.csv','c214d8473e346b3d8f0d430386c812bb114eb0ba658df197c6fc3716302efbb1')
main_path=fetch(base+'/results/model_results.csv','model_results.csv','9a387cf26adce9cb796a2aa13388d9012c370fa72f2ea42c8a0535ae1ef0d99a')
ablated_path=fetch(base+'/results/ablation_model_results.csv','ablation_model_results.csv','b0a1db34354731fdbfbf292603f9420ca03933bb2bddce1bf8714eb18659cfc9')
with zipfile.ZipFile(archive) as z:
    records=json.loads(z.read('LLM-AuthorBench.json'))
df=pd.DataFrame({'row_id':range(len(records)),'label':[r['model_name'] for r in records],
                 'code':[r['c_code'] for r in records],'prompt':[r['prompt'] for r in records]})
df['source_hash']=df.code.map(lambda s:hashlib.sha256(s.encode('utf-8')).hexdigest())
splits=pd.read_csv(split_path)
assert len(df)==len(splits)==32000
assert not splits.row_id.duplicated().any()
df=df.merge(splits,on=['row_id','source_hash','label'],validate='one_to_one')
assert len(df)==32000 and df.split.notna().all()
labels=sorted(df.label.unique()); label2id={v:i for i,v in enumerate(labels)}
assert len(labels)==8
for field in ['problem_id','prompt','source_hash']:
    for split in ['train','validation','test']:
        assert not set(df.loc[df.split==split,field]) & set(df.loc[df.split!=split,field]), field
for split in ['train','validation','test']:
    assert set(df.loc[df.split==split,'label'])==set(labels)
df['target']=df.label.map(label2id)
print('Leakage check PASSED for saved inferred groups, exact prompts and source hashes')
print('No authoritative task IDs exist; this is not proof against every semantic paraphrase.')
display(df.groupby('split').agg(samples=('row_id','size'),families=('problem_id','nunique')))
display(pd.crosstab(df.split,df.label))
(RUN_ROOT/'input_manifest.json').write_text(json.dumps({'baseline_commit':BASELINE_COMMIT,'dataset_commit':DATASET_COMMIT,
  'input_sha256':{p.name:sha256(p) for p in [archive,split_path,main_path,ablated_path]},'labels':labels},indent=2))
''')
md('''## 4. Define the model and the single ablation
Only source code reaches the tokenizer. The `code: ` prefix matches the released neural notebook; prompts and labels never enter the input text. The classifier head is newly initialized for eight labels. All encoder layers are trainable.

Comments are removed with the same string/character-aware scanner as the classical experiment. Identifiers remain unchanged. Each condition starts again from the same pretrained encoder and seed, rather than continuing from the other condition's trained weights.''')
code(r'''
import re
COMMENT_TOKEN=re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*(?:\n|$)|/\*.*?\*/',re.S)
def remove_comments(code):
    code=code.replace('\\\r\n','').replace('\\\n','')
    def replace(m):
        text=m.group()
        return ' '+('\n'*text.count('\n')) if text.startswith('//') or text.startswith('/*') else text
    return COMMENT_TOKEN.sub(replace,code)
assert remove_comments('int/**/x; // comment\n')=='int x;  \n'
assert remove_comments('"https://a/*b*/"')=='"https://a/*b*/"'

class CodeT5Authorship(nn.Module):
    accepts_loss_kwargs=False
    def __init__(self,encoder,classes):
        super().__init__()
        self.encoder=encoder
        self.config=encoder.config
        self.config.num_labels=len(classes)
        self.config.id2label=dict(enumerate(classes))
        self.config.label2id={v:i for i,v in enumerate(classes)}
        self.pre_classifier=nn.Linear(encoder.config.d_model,768)
        self.activation=nn.GELU()
        self.dropout=nn.Dropout(0.2)
        self.classifier=nn.Linear(768,len(classes))
    def forward(self,input_ids,attention_mask=None,labels=None):
        first=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0,:]
        logits=self.classifier(self.dropout(self.activation(self.pre_classifier(first))))
        loss=nn.functional.cross_entropy(logits.float(),labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss,logits=logits)

class EncodedCodeDataset(Dataset):
    def __init__(self,rows,tokenizer):
        self.records=[]; at_limit=0
        texts=rows.code.tolist()
        for start in range(0,len(texts),64):
            batch=tokenizer(['code: '+s for s in texts[start:start+64]],truncation=True,max_length=MAX_LENGTH,padding=False)
            targets=rows.target.iloc[start:start+64].tolist()
            for ids,mask,label in zip(batch['input_ids'],batch['attention_mask'],targets):
                at_limit+=int(len(ids)==MAX_LENGTH)
                self.records.append({'input_ids':ids,'attention_mask':mask,'labels':int(label)})
        self.at_token_limit_fraction=at_limit/len(self.records)
    def __len__(self): return len(self.records)
    def __getitem__(self,i): return self.records[i]

def metrics(evaluation):
    pred=evaluation.predictions
    if isinstance(pred,tuple): pred=pred[0]
    if np.ndim(pred)>1: pred=np.argmax(pred,axis=-1)
    p,r,f,_=precision_recall_fscore_support(evaluation.label_ids,pred,labels=range(len(labels)),average='macro',zero_division=0)
    _,_,weighted,_=precision_recall_fscore_support(evaluation.label_ids,pred,labels=range(len(labels)),average='weighted',zero_division=0)
    return {'accuracy':accuracy_score(evaluation.label_ids,pred),'macro_f1':f,'macro_precision':p,'macro_recall':r,'weighted_f1':weighted}
def keep_predictions_only(logits,unused_labels):
    if isinstance(logits,tuple): logits=logits[0]
    return logits.argmax(dim=-1)
''')
md('''## 5. Train, select on validation, then evaluate test
The original released notebook selects its best epoch on the same 20% split it later calls test. This notebook instead uses our independent validation split for selection and touches test only after training.

Training: AdamW, learning rate 5e-5, weight decay 0.01, 200 warmup steps, cosine-with-restarts schedule, ten epochs. These follow the released settings, except the documented class/split/token/microbatch/precision changes. Gradient checkpointing trades speed for lower memory use. Do not adjust settings after looking at test results.

A saved full checkpoint includes optimizer state and can be large. Check available disk space. If a runtime ends, reconnect the same Drive directory and rerun with the same settings and `RESUME=True`. A resumed run reports session training time separately and leaves total training time blank rather than inventing the duration of an interrupted session.

If you hit out-of-memory, restart the runtime and reduce MAX_LENGTH with a **new RUN_NAME**, documenting the deviation. Do not quietly switch to a smaller model and call it the authors' model.''')
code(r'''
from datetime import datetime,timezone

def run_condition(condition):
    set_seed(SEED)
    work=RUN_ROOT/condition; work.mkdir(exist_ok=True)
    checkpoint_dir=work/'checkpoints'; checkpoint_dir.mkdir(exist_ok=True)
    settings={'seed':SEED,'condition':condition,'base_model':BASE_MODEL,'model_revision':MODEL_REVISION,
              'baseline_commit':BASELINE_COMMIT,'max_length':MAX_LENGTH,'epochs':EPOCHS,'microbatch':MICROBATCH,
              'gradient_accumulation':GRAD_ACCUM,'learning_rate':LEARNING_RATE,'warmup_steps':200,
              'weight_decay':0.01,'scheduler':'cosine_with_restarts','labels':labels,'smoke_test':SMOKE_TEST,
              'mixed_bf16':USE_BF16,'fp16':False,'gradient_checkpointing':True,'input_prefix':'code: ',
              'selection_metric':'validation_macro_f1'}
    config_file=work/'run_config.json'
    if config_file.exists():
        assert json.loads(config_file.read_text())==settings, 'Settings changed. Use a NEW RUN_NAME.'
    config_file.write_text(json.dumps(settings,indent=2))
    last=get_last_checkpoint(str(checkpoint_dir))
    if last and not RESUME: raise RuntimeError('Existing checkpoints found. Use RESUME=True or a new run name.')
    completed=work/'metrics.csv'
    if completed.exists():
        print('Completed condition already exists:',work,'; reusing its measured outputs.')
        return pd.read_csv(completed)
    token_start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,revision=MODEL_REVISION,trust_remote_code=False)
    source=df.copy()
    if condition=='comments_removed': source['code']=source.code.map(remove_comments)
    rows={s:source[source.split==s].reset_index(drop=True) for s in ['train','validation','test']}
    if SMOKE_TEST:
        rows={s:pd.concat([g.head(2 if s=='train' else 1) for _,g in frame.groupby('label')]).reset_index(drop=True) for s,frame in rows.items()}
    sets={s:EncodedCodeDataset(frame,tokenizer) for s,frame in rows.items()}
    token_seconds=time.perf_counter()-token_start
    (work/'tokenization.json').write_text(json.dumps({'seconds':token_seconds,'counts':{s:len(x) for s,x in sets.items()},
      'fraction_reaching_token_limit':{s:x.at_token_limit_fraction for s,x in sets.items()},
      'limit_note':'Includes sequences exactly at the limit, not only truncated sequences.'},indent=2))
    encoder=T5EncoderModel.from_pretrained(BASE_MODEL,revision=MODEL_REVISION,torch_dtype=torch.float32,trust_remote_code=False)
    encoder.config.use_cache=False
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model=CodeT5Authorship(encoder,labels)
    param_count=sum(p.numel() for p in model.parameters())
    trainable_count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count==trainable_count
    arguments=TrainingArguments(output_dir=str(checkpoint_dir),num_train_epochs=EPOCHS,
      max_steps=2 if SMOKE_TEST else -1,per_device_train_batch_size=MICROBATCH,
      per_device_eval_batch_size=MICROBATCH,gradient_accumulation_steps=GRAD_ACCUM,
      learning_rate=LEARNING_RATE,weight_decay=0.01,warmup_steps=0 if SMOKE_TEST else 200,
      lr_scheduler_type='cosine_with_restarts',optim='adamw_torch',max_grad_norm=1.0,
      eval_strategy='epoch',save_strategy='epoch',logging_steps=1 if SMOKE_TEST else 25,logging_nan_inf_filter=False,
      load_best_model_at_end=True,metric_for_best_model='macro_f1',greater_is_better=True,
      save_total_limit=2,save_safetensors=False,report_to='none',seed=SEED,data_seed=SEED,
      bf16=USE_BF16,fp16=False,tf32=False,dataloader_num_workers=0,
      remove_unused_columns=False,eval_accumulation_steps=16)
    trainer=Trainer(model=model,args=arguments,train_dataset=sets['train'],eval_dataset=sets['validation'],
      data_collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8),compute_metrics=metrics,
      preprocess_logits_for_metrics=keep_predictions_only)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
    training=trainer.train(resume_from_checkpoint=last if RESUME else None)
    torch.cuda.synchronize(); fit_seconds=time.perf_counter()-start
    history=pd.DataFrame(trainer.state.log_history); history.to_csv(work/'training_history.csv',index=False)
    if 'loss' in history and not np.isfinite(history['loss'].dropna()).all():
        raise RuntimeError('Non-finite training loss. Do not report these results.')
    best_dir=work/'best_model'; trainer.save_model(str(best_dir)); tokenizer.save_pretrained(best_dir)
    (best_dir/'authorship_config.json').write_text(json.dumps(settings,indent=2))
    model_bytes=sum(p.stat().st_size for p in best_dir.rglob('*') if p.is_file())
    all_rows=[]
    for split in ['validation','test']:
        torch.cuda.synchronize(); start=time.perf_counter()
        prediction=trainer.predict(sets[split],metric_key_prefix=split)
        torch.cuda.synchronize(); inference_seconds=time.perf_counter()-start
        ytrue=prediction.label_ids; ypred=np.asarray(prediction.predictions)
        if ypred.ndim>1: ypred=ypred.argmax(-1)
        measured=metrics(prediction)
        row={'model':'CodeT5-Authorship (8-class adaptation)','condition':condition,'split':split,
             **measured,'training_seconds':fit_seconds if last is None else None,
             'fit_wall_seconds_current_session':fit_seconds,'inference_seconds':inference_seconds,
             'model_MB':model_bytes/1e6,'parameters':param_count,'trainable_parameters':trainable_count,
             'gpu_peak_allocated_MB':torch.cuda.max_memory_allocated()/1e6,'resumed':last is not None,
             'smoke_test':SMOKE_TEST,'reportable':not SMOKE_TEST}
        all_rows.append(row)
        pd.DataFrame({'row_id':rows[split].row_id,'problem_id':rows[split].problem_id,
                      'true':[labels[int(i)] for i in ytrue],
                      'predicted':[labels[int(i)] for i in ypred]}).to_csv(work/f'{split}_predictions.csv',index=False)
        pd.DataFrame(classification_report(ytrue,ypred,labels=range(len(labels)),target_names=labels,output_dict=True,zero_division=0)).T.to_csv(work/f'{split}_per_class.csv')
        pd.DataFrame(confusion_matrix(ytrue,ypred,labels=range(len(labels))),index=labels,columns=labels).to_csv(work/f'{split}_confusion_matrix.csv')
    result=pd.DataFrame(all_rows)
    result.to_csv(completed,index=False)
    (work/'training_run.json').write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),
      'best_model_checkpoint':trainer.state.best_model_checkpoint,'best_validation_macro_f1':trainer.state.best_metric,
      'trainer_metrics':training.metrics,'fit_wall_seconds_current_session':fit_seconds,'resumed':last is not None,
      'timing_note':'Training includes validation and checkpoint I/O. Prediction includes data loading and scoring but excludes pre-tokenization. Different hardware and timing boundaries from CPU baselines.'},indent=2,default=str))
    del trainer,model,encoder,sets,source,rows
    gc.collect(); torch.cuda.empty_cache()
    return result

conditions=['original','comments_removed'] if RUN_COMMENT_ABLATION else ['original']
neural_results=pd.concat([run_condition(condition) for condition in conditions],ignore_index=True)
display(neural_results)
if SMOKE_TEST: print('SMOKE TEST ONLY. These scores must not be used in the assignment.')
''')
md('''## 6. Compare our own runs and write the conclusion from evidence
The comparison imports our actual classical results, not the paper's scores. The original classical models use full-source TF-IDF; CodeT5 sees a token-limited prefix, learned features and a pretrained initialization. Model size is the saved bundle on disk (uncompressed neural weights versus compressed classical classifiers). Training hardware and timing boundaries differ, so time and size need those qualifications.

If CodeT5 wins, report that it won under this protocol. Pretraining and contextual token representations are **possible explanations**, not demonstrated causes. The comment-removal comparison tests sensitivity to comments, not the causal value of pretraining, attention or semantic understanding. If it does not win, report that result without changing the test split or trying configurations until one does.''')
code(r'''
if SMOKE_TEST:
    print('Skipping assignment comparison for smoke-test outputs.')
else:
    original=pd.read_csv(main_path); original['condition']='original'
    ablated=pd.read_csv(ablated_path); ablated['condition']='comments_removed'
    classical=pd.concat([original,ablated],ignore_index=True)
    classical=classical[classical.condition.isin(conditions)]
    classical['reportable']=True; classical['representation']='character TF-IDF'
    neural_results['representation']='pretrained CodeT5+ encoder, token limit '+str(MAX_LENGTH)
    comparison=pd.concat([classical,neural_results],ignore_index=True)
    comparison['rank']=comparison.groupby(['condition','split']).macro_f1.rank(ascending=False,method='min').astype(int)
    comparison.to_csv(RUN_ROOT/'five_model_comparison.csv',index=False)
    display(comparison[comparison.split=='test'][['condition','model','accuracy','macro_f1','rank','training_seconds','model_MB']])
    validation=comparison[(comparison.condition=='original')&(comparison.split=='validation')]
    selected=validation.sort_values(['macro_f1','model'],ascending=[False,True]).iloc[0]
    test=comparison[(comparison.condition=='original')&(comparison.split=='test')]
    selected_test=test[test.model==selected.model].iloc[0]
    conclusion=f"Validation-selected winner: {selected.model}. Validation Macro-F1={selected.macro_f1:.4f}; test Macro-F1={selected_test.macro_f1:.4f}.\n"
    if RUN_COMMENT_ABLATION:
        paired=comparison.pivot(index=['model','split'],columns='condition',values='macro_f1').reset_index()
        paired['f1_change_pp']=100*(paired.comments_removed-paired.original)
        paired.to_csv(RUN_ROOT/'five_model_ablation.csv',index=False)
        display(paired)
        conclusion+='Comment-removal effects and rankings are saved for all five systems. Their direction must be read from these outputs.\n'
    else:
        conclusion+='Neural comment ablation was not run. Do not claim the neural winner has an ablation-tested explanation.\n'
    conclusion+='This compares our five systems on eight labels and inferred problem groups. It is not a reproduction or numerical confirmation of the paper. Pretraining/context explanations are hypotheses, not conclusions proven by the comments ablation.'
    (RUN_ROOT/'interpretation.txt').write_text(conclusion)
    print(conclusion)
''')
md('''## 7. Export results
The download contains metrics, predictions, provenance and interpretation, excluding multi-gigabyte model weights and checkpoints. Keep your best-model directory separately if you need future inference. Reopen this notebook with the same configuration and output storage to resume interrupted training.

Bring the exported ZIP back to the assignment project before updating slides. Until then, the fifth model remains **not run**, and the existing four-model results remain unchanged.''')
code(r'''
export_path=WORKSPACE/(RUN_ROOT.name+'-results.zip')
with zipfile.ZipFile(export_path,'w',compression=zipfile.ZIP_DEFLATED) as output:
    for path in RUN_ROOT.rglob('*'):
        if path.is_file() and not {'checkpoints','best_model'}.intersection(path.relative_to(RUN_ROOT).parts):
            output.write(path,path.relative_to(RUN_ROOT))
print('Results archive:',export_path)
print('Model/checkpoint location:',RUN_ROOT)
try:
    from google.colab import files
    files.download(str(export_path))
except ImportError:
    print('Download the ZIP using your notebook file browser.')
''')
md('''## How this differs from the paper and released code

| Item | Our assignment / this notebook | Authors' released notebook or table |
|---|---|---|
| Classes | Eight, matching our four baselines | Five for the headline multiclass experiment |
| Split | Saved inferred task groups, train/validation/test | Stratified random samples, 80/20 |
| Epoch selection | Validation Macro-F1, separate test | Best accuracy on the same held-out partition later evaluated |
| Classical input | 50,000 character n-grams fitted on training only | Released ML notebook uses 400 word features and fits TF-IDF before its split |
| Neural architecture | Same encoder/first-token/768/GELU/dropout structure, new 8-way head | 5-way head |
| Token limit | Configurable, default 512 | Paper table says 512; released code uses 1024 |
| Neural precision | BF16 mixed precision when available; FP32 otherwise | Released code loads encoder/head in BF16 |
| Batch | Microbatch 1, accumulation 32 | Microbatch 4, accumulation 8 |

The paper/official README reports five-class accuracies of 74.6% (linear SVM), 88.0% (Random Forest), and 95.4% (CodeT5-Authorship). Our existing eight-class grouped-test accuracies are 58.96% (SVM) and 50.34% (Random Forest). These different protocols cannot establish a replication gap or an architecture advantage. The authors' F1 implementation is weighted; ours emphasizes Macro-F1. Both are saved for the new model.

Also, our four algorithms are **our assignment baselines**, not an exact copy of the paper's baseline set: their published classical table does not include our Naive Bayes and Logistic Regression choices. Cite the study, and describe this work as an adaptation with a stricter split design, not proof that their reported score was reproduced.
''')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'colab':{'name':'CodeT5_Authorship_Colab.ipynb','provenance':[]},'accelerator':'GPU'},'cells':cells}
path=root/'notebooks/CodeT5_Authorship_Colab.ipynb'
path.write_text(json.dumps(nb,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Created',path,'with',len(cells),'cells. All execution counts are null.')
