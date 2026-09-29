"""Create the two presentation figures from saved experiment results."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
(ROOT / 'tmp').mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(ROOT / 'tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
out=ROOT / 'results/figures'; out.mkdir(exist_ok=True)
cm=pd.read_csv(ROOT / 'results/experiment_1_original/Linear SVM_test_confusion.csv',index_col=0)
short={'claude-3.5-haiku':'Claude','deepseek-chat':'DeepSeek','gemini-2.5-flash-preview-05-20':'Gemini','gpt-4.1':'GPT-4.1','llama-3.3-70b-instruct':'Llama'}
plt.rcParams.update({'font.family':'Arial','font.size':17})
fig,ax=plt.subplots(figsize=(6.4,6.4),layout='constrained')
fig.patch.set_facecolor('#F7FAFA'); ax.imshow(cm.values,cmap='Blues',vmin=0,vmax=cm.values.max())
ax.set_xticks(range(5),[short[x] for x in cm.columns],rotation=35,ha='right'); ax.set_yticks(range(5),[short[x] for x in cm.index])
ax.set_xlabel('Predicted LLM'); ax.set_ylabel('Actual LLM')
for r in range(5):
 for c in range(5): ax.text(c,r,str(cm.iloc[r,c]),ha='center',va='center',fontsize=21,color='white' if cm.iloc[r,c]>cm.values.max()/2 else '#152E38')
ax.set_title('SVM test confusion matrix',fontsize=22,pad=16)
fig.savefig(out/'svm_confusion_matrix.png',dpi=200); plt.close(fig)
a=pd.read_csv(ROOT / 'results/feature_comparison.csv'); a=a[a.split=='validation'].sort_values('original_rank')
fig,ax=plt.subplots(figsize=(12,3.4),layout='constrained'); fig.patch.set_facecolor('#F7FAFA'); ax.set_facecolor('#F7FAFA')
y=np.arange(4); ax.barh(y-.18,a.original,height=.32,label='50,000 features',color='#117C83'); ax.barh(y+.18,a.limited_features,height=.32,label='500 features',color='#9CABB7')
ax.set_yticks(y,a.model); ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_xlabel('Validation Macro-F1'); ax.legend(loc='lower right',frameon=False,fontsize=15)
for i,r in enumerate(a.itertuples()):
 ax.text(r.original+.012,i-.18,f'{r.original:.4f}',va='center',fontsize=14)
 ax.text(r.limited_features+.012,i+.18,f'{r.limited_features:.4f}',va='center',fontsize=14)
ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'feature_ablation.png',dpi=200); plt.close(fig)
assert cm.values.sum()==3872
print('Matrix total:',cm.values.sum(),'Correct:',np.trace(cm.values))
