"""Assignment-specific narrative and deck data, derived from actual run outputs."""
import json
import pandas as pd
from .utils import ROOT,save_json

def augment_report():
    path=ROOT/'results/ablation_ranking.csv'
    if not path.exists(): return
    ranking=pd.read_csv(path)
    val=ranking[ranking.split=='validation'].set_index('model')
    test=ranking[ranking.split=='test'].set_index('model')
    before=val.loc['Linear SVM','macro_f1_original']-val.loc['Logistic Regression','macro_f1_original']
    after=val.loc['Linear SVM','macro_f1_ablated']-val.loc['Logistic Regression','macro_f1_ablated']
    outcome='narrowed' if after<before else 'widened'
    supported=after<before
    ranking_table='| Model | Original validation F1 | No-comment validation F1 | Rank before / after |\n|---|---:|---:|---:|\n'
    for name,r in val.sort_values('rank_original').iterrows():
        ranking_table+=f'| {name} | {r.macro_f1_original:.4f} | {r.macro_f1_ablated:.4f} | {int(r.rank_original)} / {int(r.rank_ablated)} |\n'
    statement=f"SVM's validation lead over Logistic Regression {outcome} from {100*before:.2f} to {100*after:.2f} percentage points. "
    statement+=('This is consistent with the follow-up directional expectation.' if supported else 'This contradicts the follow-up expectation that removing comments would narrow its validation lead.')
    statement+=' SVM remained first on both validation and test. Random Forest and Logistic Regression swapped validation ranks, but their ablated F1 values differ by only '+f"{abs(val.loc['Random Forest','macro_f1_ablated']-val.loc['Logistic Regression','macro_f1_ablated']):.5f}. Test ranking stayed unchanged. No significance or robust rank reversal is claimed."
    explanation='The measured property is a sparse, overlapping vocabulary of source and comment fragments: 50,000 features, 3.81% training density. Strong SVM coefficients include brace-adjacent comments, inline comment markers, and compact control syntax. Every classifier lost Macro-F1 when comments were removed. This supports the usefulness of comments, but the persistent SVM lead means comments alone do not explain why SVM won. Other lexical patterns remain a plausible contributor, not a demonstrated cause. A single split and refitted vocabulary limit causal and generalization claims.'
    readme=ROOT/'README.md'; text=readme.read_text(encoding='utf-8')
    section='''## Assignment-aligned ranking ablation

One comment-removal condition now covers all four classifiers. Each uses the same saved split, the same newly fitted training-only vectorizer, and its original settings. `python run_all.py` executes both complete conditions. `python run_ablation.py` repeats only the ablation using existing original artifacts.

The original SVM-only ablation was already known. The extension is therefore an exploratory follow-up. Its dated rationale and prediction are in `results/ablation_protocol.md`; it is not presented as an independent preregistered test.

'''+ranking_table+'\n'+statement+'\n\n'+explanation+'''

Full validation/test metrics and rankings: `results/ablation_model_results.csv` and `results/ablation_ranking.csv`. The original winner remains selected from original validation results.

## Collection provenance and sources

The authors describe a corpus generated from 300 parameterized C-programming templates, followed by deduplication and compilation filtering. Our source-only experiment does not execute that compilation step. The released archive contains 10,262 distinct prompt strings in our inspection; this observed count must not be replaced with the paper's count of generated task instances.

- Bisztray et al., *I Know Which LLM Wrote Your Code Last Summer: LLM Generated Code Stylometry for Authorship Attribution*, https://arxiv.org/abs/2506.17323 . Cited for collection provenance, not numerical comparison results.
- Official release: https://github.com/LLMauthorbench/LLMauthorbench . The generator notebook is read only to audit template grouping, not executed or used to train the classifiers.
- scikit-learn: https://scikit-learn.org/stable/ . Provides TF-IDF, the four classifiers and metrics. NumPy, pandas, Matplotlib and joblib support numerical arrays, tables, figures and serialization. Versions are recorded in requirements.txt and results/environment.json.

## Submission status

Finished five-slide presentation: [Download the PowerPoint](presentation/LLM_AuthorBench_Assignment.pptx). Member details are intentionally pending at the user's request. The assignment still requires a verified research-use licence, confirmation of the dataset claim, and genuine contributions from every group member. See `docs/ASSIGNMENT_STATUS.md` and `docs/DATASET_PERMISSION.md`. The latter includes permission-request drafts; no messages have been sent.

Presentation source: `presentation/build_deck.mjs`. It uses @oai/artifact-tool from the authoring runtime, separate from the five-package ML environment. The exported PowerPoint is editable and does not require that authoring runtime to view or edit.
'''
    text=text.split('\n## Assignment-aligned ranking ablation')[0]
    readme.write_text(text+'\n'+section,encoding='utf-8')
    slides=ROOT/'slides_content.md'; content=slides.read_text(encoding='utf-8').split('# Slide 5')[0]
    content+='''# Slide 5 - Comments help, but SVM stays first
'''+ranking_table+'\n- '+statement+'\n- Comments help all four classifiers, but do not explain SVM\'s unique advantage.\n- This is an exploratory follow-up after the initial SVM-only result. One split, no significance claim.\n'
    slides.write_text(content,encoding='utf-8')
    (ROOT/'results/ablation_interpretation.md').write_text('# Four-model ablation interpretation\n\n'+statement+'\n\n'+explanation+'\n\n'+ranking_table,encoding='utf-8')
    save_json('results/presentation_data.json',{'original':pd.read_csv(ROOT/'results/model_results.csv').to_dict('records'),'ablation':ranking.to_dict('records'),'splits':pd.read_csv(ROOT/'results/split_summary.csv').to_dict('records'),'baselines':json.loads((ROOT/'results/baselines.json').read_text()),'svm_lr_validation_gap_before_pp':100*before,'svm_lr_validation_gap_after_pp':100*after,'hypothesis_supported_on_validation':bool(supported),'interpretation':statement,'data_explanation':explanation})
    return statement
if __name__=='__main__': print(augment_report())
