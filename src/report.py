"""Generate figures and documentation from measured outputs."""
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .utils import ROOT

def table(df):
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))

def report(df,stats,winner,results,ablation):
    plt.rcParams.update({'font.size':10,'figure.dpi':150})
    tests=results[results.split=='test'].copy()
    def bars(values,title,ylabel,filename):
        fig,ax=plt.subplots(figsize=(10,6)); values.plot.bar(ax=ax,color='#28788d',rot=25)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel(''); ax.grid(axis='y',alpha=.2)
        fig.tight_layout(); fig.savefig(ROOT/f'results/figures/{filename}.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    df.label.value_counts().sort_index().plot.barh(ax=ax,color='#28788d')
    ax.set_title('Class distribution'); ax.set_xlabel('Programs'); ax.set_ylabel(''); fig.tight_layout()
    fig.savefig(ROOT/'results/figures/class_distribution.png'); plt.close(fig)
    bars(tests.set_index('model').macro_f1,'Test Macro-F1 (selection uses validation)','Macro-F1','macro_f1_comparison')
    bars(tests.set_index('model').training_seconds,'Classifier training time','Seconds','training_time_comparison')
    cm=pd.read_csv(ROOT/f'results/confusion_matrices/{winner}_test.csv',index_col=0)
    fig,ax=plt.subplots(figsize=(10,9)); im=ax.imshow(cm,cmap='Blues'); fig.colorbar(im,ax=ax,fraction=.04)
    ax.set_xticks(range(len(cm)),cm.columns,rotation=55,ha='right'); ax.set_yticks(range(len(cm)),cm.index)
    for i in range(len(cm)):
        for j in range(len(cm)): ax.text(j,i,str(cm.iloc[i,j]),ha='center',va='center',fontsize=8,color='white' if cm.iloc[i,j]>cm.values.max()/2 else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(f'{winner}: held-out task families'); fig.tight_layout(); fig.savefig(ROOT/'results/figures/best_confusion_matrix.png'); plt.close(fig)
    f=ablation.set_index('metric').loc['macro_f1']
    bars(pd.Series({'Original':f.original,'Comments removed':f.ablated}),'Comment-removal ablation','Test Macro-F1','ablation_macro_f1')
    compact=pd.DataFrame({'Model':tests.model,'Accuracy':tests.accuracy.map(lambda x:f'{x:.4f}'),'Macro-F1':tests.macro_f1.map(lambda x:f'{x:.4f}'),'Training (s)':tests.training_seconds.map(lambda x:f'{x:.2f}'),'Model (MB)':tests.model_MB.map(lambda x:f'{x:.3f}')})
    compact.to_csv(ROOT/'results/final_comparison.csv',index=False)
    comparison=table(compact); splits=pd.read_csv(ROOT/'results/split_summary.csv'); split_table=table(splits)
    feature_info=json.loads((ROOT/'results/original_features.json').read_text())
    baselines=json.loads((ROOT/'results/baselines.json').read_text())
    top=pd.read_csv(ROOT/'results/top_features.csv'); examples='; '.join(f'{label}: '+', '.join(group.feature_display.head(3)) for label,group in top.groupby('class'))
    off=cm.copy()
    for i in range(len(off)): off.iloc[i,i]=0
    actual,predicted=max(((i,j) for i in range(len(off)) for j in range(len(off)) if i!=j),key=lambda x:off.iloc[x[0],x[1]])
    confusion=f'The largest directional confusion was {cm.index[actual]} to {cm.columns[predicted]} ({off.iloc[actual,predicted]} programs).'
    change=f.ablated-f.original
    interpretation=('Removing comments reduced Macro-F1, consistent with useful information in comment wording/style.' if change < -0.005 else 'Removing comments improved Macro-F1, consistent with comments adding noise under this task split.' if change>0.005 else 'Macro-F1 changed by less than 0.5 percentage points; non-comment lexical patterns may carry much of the signal.')
    rationale=f'The training matrix has {feature_info["features"]:,} character features and density {feature_info["train_density"]:.3%}. '
    if winner in ('Linear SVM','Logistic Regression'): rationale+='A regularized linear classifier can combine many sparse lexical cues without requiring the hierarchical feature partitions used by a forest. '
    elif winner=='Naive Bayes': rationale+='Aggregating character-pattern evidence with a simple probabilistic model generalized best on validation. '
    else: rationale+='The forest may capture useful nonlinear combinations of lexical cues. '
    rationale+="The leading SVM features include comment openings by braces, inline comments after semicolons, and compact if(/for( forms (see the manual feature review). " if winner=='Linear SVM' else ''
    rationale+=interpretation+' This single split and comment-only ablation do not prove a causal explanation or test identifier style independently.'
    readme=f'''# LLM-Generated Code Authorship Attribution: Algorithm Comparison on LLM-AuthorBench

## Research Question
How effectively can standard machine-learning algorithms identify which LLM generated a C source-code program?

## Dataset
[Official dataset]({stats['source']}). Local archive: **{stats['total_records']:,} programs**, **{stats['classes']} classes**, **{stats['exact_prompts']:,} exact prompts**. Each class contains 4,000 samples. Exact labels: {', '.join(sorted(df.label.unique()))}.

License: **{stats['license']}**. Raw data is excluded from Git. Original download date is unknown because the archive was supplied locally. Inspection: {stats['inspection_utc']}. Archive SHA-256: `{stats['archive_sha256']}`. Upstream reference commit: `{stats.get('upstream_reference_commit','not recorded')}`; the archive Git blob hash was verified against this commit. Full provenance and field inventory: `results/dataset_statistics.json`.

## Why This Dataset
Our research interests include attribution of LLM-generated code. This assignment evaluates the simpler source-code attribution problem using standard machine-learning algorithms. Programs are treated only as text: no compilation, execution, binary analysis, or paper-result reuse.

## Data Preparation
The loader reads only `c_code` as input and `model_name` as target; prompts are used only for grouping. It checks required fields and empty code, computes exact source hashes, removes repeated copies, and excludes conflicting-label duplicates. Findings: {stats['malformed_or_missing_or_empty']} invalid records, {stats['exact_duplicate_excess']} exact duplicate excess records, {stats['cross_label_duplicate_hashes']} cross-label duplicate hashes. Programs are embedded in JSON, so there are no per-program files to check. C syntax validity is not tested.

There is no explicit task ID. Numeric, quoted, list, and placeholder prompt parameters are canonicalized, along with instruction verbs and program/function wording. Manually audited semantic aliases (such as Sudoku, dice simulation, and number guessing) are first collapsed. Near-identical descriptions (similarity >=0.84) and substantial substring variants are joined transitively before splitting. This intentionally groups more conservatively than exact prompt identity. All {df.problem_id.nunique()} inferred task families are shuffled with seed 42 and partitioned approximately 70/15/15 by group count.

{split_table}

Assertions check disjoint inferred problem families, exact prompts, and exact code hashes across all splits and all classes present in each split. **The same inferred programming-problem group never appears in both training and testing.** This is not proof that every semantic paraphrase is recognized. Review `results/problem_groups.csv` and `results/grouping_edges.csv`. Saved sample assignments: `data/processed/splits.csv`; per-group counts and per-split class counts are in results.

## Features
Shared TF-IDF character 3-5-grams, maximum 50,000 features, sublinear term frequency, float32, fitted only on training code. Case is retained (`lowercase=False`, a deliberate change from the default). Comments, identifiers, and formatting remain in loaded input. The sklearn character analyzer internally collapses repeated whitespace, so exact indentation width is not represented. Validation/test are transformed without refitting. Feature extraction time and shared vectorizer size: `results/original_features.json`.

## Algorithms
- Multinomial Naive Bayes: alpha=1, simple baseline.
- Logistic Regression: C=1, lbfgs, max_iter=3000.
- Linear SVM: C=1, dual=auto, max_iter=10000.
- Random Forest: 150 trees, max_depth=40, min_samples_leaf=2, sqrt feature sampling, four workers.

Seed 42 wherever supported; no hyperparameter search. Identical split, preprocessing, and feature matrices for all models. Winner **{winner}** selected by validation Macro-F1. Test performance does not change selection. Estimators are not refitted on train+validation.

## Evaluation
Accuracy, macro precision/recall/F1, per-class metrics, confusion matrices, classifier fit time, batch prediction time, and compressed classifier size are measured. Macro-F1 weights all classes equally. Timings are single-run wall-clock measurements, exclude feature extraction, and depend on hardware. Model size excludes the separately saved vectorizer. No uncertainty interval is claimed.

Uniform random expected accuracy: {baselines['random_chance_accuracy']:.2%}. Training-majority test accuracy: {baselines['majority_test']['accuracy']:.2%}; Macro-F1: {baselines['majority_test']['macro_f1']:.4f}. The overall dataset is balanced; grouped splits can be imbalanced.

## Results
{comparison}

Detailed validation/test results: `results/model_results.csv`. Per-class reports, predictions, and all confusion matrices are saved. {confusion}

## Ablation and Interpretation
Same winning estimator/settings/split, newly fitted training-only TF-IDF after comment removal. The scanner protects strings and character literals, replaces comments with whitespace, and handles escaped physical newlines. Identifiers are unchanged. This does not isolate all aspects of source style.

Original test Macro-F1 **{f.original:.4f}**; ablated **{f.ablated:.4f}**; signed change **{100*change:+.2f} percentage points** ({f.relative_percentage_difference:+.2f}% relative). Accuracy and absolute/relative differences: `results/ablation_results.csv`.

{rationale}

Strong feature examples: {examples}. Overlapping character fragments are cues, not independent semantic explanations. Full table: `results/top_features.csv`.

## Leakage Audit
Only code enters the feature extractor. Metadata, prompts, hashes and labels are excluded. The source audit searches model-name fragments and attribution phrases. Initial manual inspection found API model names in LLM-client task implementations, ordinary 'generated by' comments, an author placeholder, and a Hungarian substring matching 'llama'; these are not direct generator labels. See `results/source_leakage_audit.csv` and `results/feature_review.md`.

## Reproduction
Python 3.12 was used. From your clone:

```sh
python -m venv .venv
# Activate .venv for your shell, then:
pip install -r requirements.txt
python run_all.py
python -m unittest discover -s tests
python -m src.audit_templates
```

Paths resolve from the project root and are portable. The runner uses `data/raw/LLM-AuthorBench.json.zip`, copies a supplied root archive there, or downloads the archive from the pinned upstream commit if absent. An SHA-256 check rejects any different archive. Generated datasets, models, cache and temporary files stay in the project. For installation scratch files, set TEMP/TMP (Windows) or TMPDIR (Unix) to the project's `tmp` folder and install with `pip --no-cache-dir`.

Dependencies are pinned; environment details: `results/environment.json`. Timings vary. Exact data reproduction requires the recorded archive hash. Outputs are overwritten on rerun.

## AI Usage
AI assistants supported code development, debugging, experiment execution, and documentation. Numerical results came from actual local runs. Group members must independently review and understand the work; no human contribution or execution claim is fabricated.

## Contributions
Member 1: [name] - dataset inspection and grouped splitting.

Member 2: [name] - training-only feature engineering.

Member 3: [name] - models and evaluation.

Member 4: [name] - ablation and feature analysis.

Member 5: [name] - documentation, figures and five-slide presentation.

These are proposed responsibilities, not completed human contributions. Each member must make real commits for their own substantive work. See `CONTRIBUTING.md`.

Repository: https://github.com/S3eeDTR/LLM-AuthorBench-Assignment
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    slides=f'''# Slide 1 - Dataset & Story
**LLM-Generated Code Authorship Attribution**
- LLM-AuthorBench: {len(df):,} C programs, eight LLMs, {df.problem_id.nunique()} inferred task families.
- Can standard ML identify the generating LLM from C source?
- Connection: source-level foundation for our code-attribution research.
- Source: {stats['source']}
- Repository: https://github.com/S3eeDTR/LLM-AuthorBench-Assignment

# Slide 2 - Data Handling
- 4,000 programs per class; {stats['exact_prompts']:,} exact prompt strings.
- {stats['malformed_or_missing_or_empty']} invalid records; {stats['exact_duplicate_excess']} exact duplicates.
- Seed 42; conservative prompt-family grouping; saved assignments.
{split_table}
- Training-only TF-IDF; comments and identifiers retained.
- **The same inferred programming problem never appears in both training and testing.**
- Limitation: no authoritative IDs; semantic paraphrase leakage cannot be ruled out.

# Slide 3 - Algorithms
```text
C source -> shared character TF-IDF (3-5 grams; 50,000 features)
                  +-- Multinomial Naive Bayes (simple baseline)
                  +-- Logistic Regression
                  +-- Linear SVM
                  +-- Random Forest
```
- Same split, preprocessing, features and seed where supported.
- No heavy tuning; select by validation Macro-F1.
- Chance accuracy: 12.5%; majority test accuracy: {baselines['majority_test']['accuracy']:.2%}.

# Slide 4 - Results
{comparison}
- **Validation-selected winner: {winner}.** Table shows held-out test results.
- Time excludes feature extraction; size is compressed classifier only.
- Optional chart: `results/figures/macro_f1_comparison.png`.

# Slide 5 - Why + Ablation
- Winner: **{winner}**; {feature_info['features']:,} sparse lexical features (density {feature_info['train_density']:.2%}).
- Sparse character cues suit a regularized linear model; top features include comment placement and compact control syntax.
- {interpretation}
- Comments removed; same classifier/settings/split, new training-only TF-IDF.
- Test Macro-F1: **{f.original:.4f} -> {f.ablated:.4f}** ({100*change:+.2f} pp).
- Conclusion: attribution is measurable on held-out inferred task families, with limited generalization claims.
'''
    (ROOT/'slides_content.md').write_text(slides,encoding='utf-8')
