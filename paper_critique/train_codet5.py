"""CodeT5-Authorship adaptation. Test data never selects a checkpoint."""
import argparse, hashlib, importlib.metadata, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel, DataCollatorWithPadding, get_cosine_with_hard_restarts_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from project_utils import ROOT, DATA_DIR, LABELS, load_data, remove_comments

def metrics(y,p):
    return {'accuracy':accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0),'weighted_f1':f1_score(y,p,average='weighted',zero_division=0)}

class Authorship(nn.Module):
    def __init__(self,revision=None):
        super().__init__()
        self.encoder=T5EncoderModel.from_pretrained('Salesforce/codet5p-770m',revision=revision)
        self.head=nn.Sequential(nn.Linear(self.encoder.config.d_model,768),nn.GELU(),nn.Dropout(.2),nn.Linear(768,5))
    def forward(self,input_ids,attention_mask):
        return self.head(self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--max-length',type=int,default=512)
    parser.add_argument('--micro-batch',type=int,default=4)
    parser.add_argument('--accumulation',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--revision',default=None)
    parser.add_argument('--timing-only',action='store_true')
    parser.add_argument('--gradient-checkpointing',action='store_true')
    args=parser.parse_args()
    assert torch.cuda.is_available(),'A CUDA GPU is required; use the supplied Colab notebook.'
    assert torch.cuda.is_bf16_supported(),'This run requires bfloat16 support.'
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out=ROOT/'results'/f'codet5_authorship_L{args.max_length}_seed{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists() and not args.timing_only:
        raise RuntimeError('Completed results already exist. Archive them before intentionally repeating this run.')
    df=load_data(); label_id={s:i for i,s in enumerate(LABELS)}
    tokenizer=AutoTokenizer.from_pretrained('Salesforce/codet5p-770m',revision=args.revision)
    collate=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8)
    rows={part:df[df.split==part].reset_index(drop=True) for part in ['train','validation','test']}
    def loader(frame,strip=False,shuffle=False):
        texts=frame.code.map(remove_comments) if strip else frame.code
        tokens=tokenizer(['code: '+s for s in texts],truncation=True,max_length=args.max_length,padding=False,return_token_type_ids=False)
        items=[{'input_ids':tokens['input_ids'][i],'attention_mask':tokens['attention_mask'][i],'labels':label_id[s]} for i,s in enumerate(frame.label)]
        return DataLoader(items,batch_size=args.micro_batch,shuffle=shuffle,collate_fn=collate,num_workers=0,pin_memory=True)
    train=loader(rows['train'],shuffle=True); validation=loader(rows['validation'])
    model=Authorship(args.revision).cuda()
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(); model.encoder.config.use_cache=False
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01)
    steps_per_epoch=(len(train)+args.accumulation-1)//args.accumulation
    scheduler=get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,num_warmup_steps=200,num_training_steps=steps_per_epoch*args.epochs,num_cycles=1)
    settings={**vars(args),'model':'Salesforce/codet5p-770m','resolved_revision':getattr(model.encoder.config,'_commit_hash',None),'prefix':'code: ','pooling':'first token','head':'d_model -> 768 -> GELU -> dropout 0.2 -> 5','optimizer':'AdamW','learning_rate':5e-5,'weight_decay':.01,'warmup_steps':200,'scheduler':'cosine_with_restarts, one cycle','checkpoint_selection':'validation accuracy','precision':'fp32 parameters + bf16 autocast (adaptation from authors bf16 parameters)','gpu':torch.cuda.get_device_name(0),'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','pandas','scikit-learn']},'data_sha256':hashlib.sha256((DATA_DIR/'LLM-AuthorBench.json.zip').read_bytes()).hexdigest(),'split_sha256':hashlib.sha256((DATA_DIR/'splits.csv').read_bytes()).hexdigest()}
    (out/'settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    @torch.no_grad()
    def evaluate(dl):
        model.eval(); truth=[]; pred=[]; probabilities=[]
        for batch in dl:
            truth.extend(batch.pop('labels').tolist())
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(**{k:v.cuda() for k,v in batch.items()})
            pred.extend(logits.argmax(-1).cpu().tolist()); probabilities.extend(logits.float().softmax(-1).cpu().tolist())
        return metrics(truth,pred),truth,pred,probabilities
    print('Trainable encoder parameters:',sum(p.numel() for p in model.encoder.parameters()))
    print('Effective batch:',args.micro_batch*args.accumulation,'; primary run uses 512 tokens; notebook discrepancy is documented.')
    history=[]; best=-1.; start=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]; epoch_start=time.perf_counter()
        for i,batch in enumerate(train):
            y=batch.pop('labels').cuda(); batch={k:v.cuda() for k,v in batch.items()}
            group_start=(i//args.accumulation)*args.accumulation
            group_size=min(args.accumulation,len(train)-group_start)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=nn.functional.cross_entropy(model(**batch),y)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss; stop and inspect the run.')
            (loss/group_size).backward(); losses.append(loss.item())
            if (i+1)%args.accumulation==0 or i+1==len(train):
                nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if args.timing_only and i==24:
                torch.cuda.synchronize(); measured=time.perf_counter()-epoch_start
                estimate=measured/25*len(train)*args.epochs/60
                info={'measured_batches':25,'seconds':measured,'rough_training_minutes':estimate,'note':'Includes warm-up; excludes validation and checkpoint writing. No completed training result.'}
                (out/'timing_only.json').write_text(json.dumps(info,indent=2)); print(info); return
        val,*_=evaluate(validation)
        record={'epoch':epoch,'mean_batch_training_loss':float(np.mean(losses)),'validation_accuracy':val['accuracy'],'validation_macro_f1':val['macro_f1'],'epoch_seconds':time.perf_counter()-epoch_start}
        history.append(record); pd.DataFrame(history).to_csv(out/'training_history.csv',index=False); print(record,flush=True)
        if val['accuracy']>best:
            best=val['accuracy']; best_epoch=epoch
            torch.save(model.state_dict(),out/'best_model.pt')
    training_seconds=time.perf_counter()-start
    model.load_state_dict(torch.load(out/'best_model.pt',map_location='cpu',weights_only=True))
    result={'status':'completed','best_epoch':best_epoch,'best_validation_accuracy':best,'training_validation_checkpoint_seconds':training_seconds}
    for variant,strip in [('original',False),('comments_removed_at_test',True)]:
        score,y,p,prob=evaluate(loader(rows['test'],strip=strip)); result[variant]=score
        table=pd.DataFrame({'row_id':rows['test'].row_id,'true':[LABELS[i] for i in y],'predicted':[LABELS[i] for i in p]})
        for i,label in enumerate(LABELS): table['prob_'+label]=[v[i] for v in prob]
        table.to_csv(out/f'{variant}_predictions.csv',index=False)
        pd.DataFrame(classification_report(y,p,labels=range(5),target_names=LABELS,output_dict=True,zero_division=0)).T.to_csv(out/f'{variant}_class_scores.csv')
        pd.DataFrame(confusion_matrix(y,p,labels=range(5)),index=LABELS,columns=LABELS).to_csv(out/f'{variant}_confusion.csv')
    result['comment_removal_accuracy_drop_pp']=100*(result['original']['accuracy']-result['comments_removed_at_test']['accuracy'])
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    tokenizer.save_pretrained(out/'tokenizer'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
