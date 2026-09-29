# LLM-Generated Code Authorship Attribution: Algorithm Comparison on LLM-AuthorBench

## Research Question
How effectively can standard machine-learning algorithms identify which LLM generated a C source-code program?

## Dataset
[Official dataset](https://github.com/LLMauthorbench/LLMauthorbench). Local archive: **32,000 programs**, **8 classes**, **10,262 exact prompts**. Each class contains 4,000 samples. Exact labels: claude-3.5-haiku, deepseek-chat, gemini-2.5-flash-preview-05-20, gpt-4.1, gpt-4o, gpt-4o-mini, llama-3.3-70b-instruct, qwen-2.5-72b-instruct.

License: **No license file or explicit dataset license found in inspected upstream repository; redistribution permission unverified**. Raw data is excluded from Git. Original download date is unknown because the archive was supplied locally. Inspection: 2026-09-29T05:35:17.396709+00:00. Archive SHA-256: `e24399b7b05b812c5a1148ef44245348859eaa74b1140523cb695f984337f24c`. Upstream reference commit: `6a1c2ac173c774cfbd012f4c81e8d0a51bb61eb7`; the archive Git blob hash was verified against this commit. Full provenance and field inventory: `results/dataset_statistics.json`.

## Why This Dataset
Our research interests include attribution of LLM-generated code. This assignment evaluates the simpler source-code attribution problem using standard machine-learning algorithms. Programs are treated only as text: no compilation, execution, binary analysis, or paper-result reuse.

## Data Preparation
The loader reads only `c_code` as input and `model_name` as target; prompts are used only for grouping. It checks required fields and empty code, computes exact source hashes, removes repeated copies, and excludes conflicting-label duplicates. Findings: 0 invalid records, 0 exact duplicate excess records, 0 cross-label duplicate hashes. Programs are embedded in JSON, so there are no per-program files to check. C syntax validity is not tested.

There is no explicit task ID. Numeric, quoted, list, and placeholder prompt parameters are canonicalized, along with instruction verbs and program/function wording. Manually audited semantic aliases (such as Sudoku, dice simulation, and number guessing) are first collapsed. Near-identical descriptions (similarity >=0.84) and substantial substring variants are joined transitively before splitting. This intentionally groups more conservatively than exact prompt identity. All 259 inferred task families are shuffled with seed 42 and partitioned approximately 70/15/15 by group count.

| split | samples | problems |
| --- | --- | --- |
| train | 21115 | 181 |
| validation | 4777 | 38 |
| test | 6108 | 40 |

Assertions check disjoint inferred problem families, exact prompts, and exact code hashes across all splits and all classes present in each split. **The same inferred programming-problem group never appears in both training and testing.** This is not proof that every semantic paraphrase is recognized. Review `results/problem_groups.csv` and `results/grouping_edges.csv`. Saved sample assignments: `data/processed/splits.csv`; per-group counts and per-split class counts are in results.

## Features
Shared TF-IDF character 3-5-grams, maximum 50,000 features, sublinear term frequency, float32, fitted only on training code. Case is retained (`lowercase=False`, a deliberate change from the default). Comments, identifiers, and formatting remain in loaded input. The sklearn character analyzer internally collapses repeated whitespace, so exact indentation width is not represented. Validation/test are transformed without refitting. Feature extraction time and shared vectorizer size: `results/original_features.json`.

## Algorithms
- Multinomial Naive Bayes: alpha=1, simple baseline.
- Logistic Regression: C=1, lbfgs, max_iter=3000.
- Linear SVM: C=1, dual=auto, max_iter=10000.
- Random Forest: 150 trees, max_depth=40, min_samples_leaf=2, sqrt feature sampling, four workers.

Seed 42 wherever supported; no hyperparameter search. Identical split, preprocessing, and feature matrices for all models. Winner **Linear SVM** selected by validation Macro-F1. Test performance does not change selection. Estimators are not refitted on train+validation.

## Evaluation
Accuracy, macro precision/recall/F1, per-class metrics, confusion matrices, classifier fit time, batch prediction time, and compressed classifier size are measured. Macro-F1 weights all classes equally. Timings are single-run wall-clock measurements, exclude feature extraction, and depend on hardware. Model size excludes the separately saved vectorizer. No uncertainty interval is claimed.

Uniform random expected accuracy: 12.50%. Training-majority test accuracy: 11.90%; Macro-F1: 0.0266. The overall dataset is balanced; grouped splits can be imbalanced.

## Results
| Model | Accuracy | Macro-F1 | Training (s) | Model (MB) |
| --- | --- | --- | --- | --- |
| Naive Bayes | 0.4468 | 0.4309 | 0.22 | 4.640 |
| Logistic Regression | 0.5787 | 0.5731 | 22.78 | 1.383 |
| Linear SVM | 0.5896 | 0.5797 | 11.36 | 2.835 |
| Random Forest | 0.5034 | 0.4645 | 53.34 | 24.326 |

Detailed validation/test results: `results/model_results.csv`. Per-class reports, predictions, and all confusion matrices are saved. The largest directional confusion was qwen-2.5-72b-instruct to llama-3.3-70b-instruct (171 programs).

## Ablation and Interpretation
Same winning estimator/settings/split, newly fitted training-only TF-IDF after comment removal. The scanner protects strings and character literals, replaces comments with whitespace, and handles escaped physical newlines. Identifiers are unchanged. This does not isolate all aspects of source style.

Original test Macro-F1 **0.5797**; ablated **0.4287**; signed change **-15.10 percentage points** (-26.05% relative). Accuracy and absolute/relative differences: `results/ablation_results.csv`.

The training matrix has 50,000 character features and density 3.813%. A regularized linear classifier can combine many sparse lexical cues without requiring the hierarchical feature partitions used by a forest. The leading SVM features include comment openings by braces, inline comments after semicolons, and compact if(/for( forms (see the manual feature review). Removing comments reduced Macro-F1, consistent with useful information in comment wording/style. This single split and comment-only ablation do not prove a causal explanation or test identifier style independently.

Strong feature examples: claude-3.5-haiku: ' { //', '{ // ', '{ //'; deepseek-chat: '() f', 's (', 'to 0 '; gemini-2.5-flash-preview-05-20: 'For ', ' For ', ' For'; gpt-4.1: ') p', ') pr', ') pri'; gpt-4o: '0;\n} ', ' th', ' ++'; gpt-4o-mini: '; // ', '; //', '; /'; llama-3.3-70b-instruct: '*) ', '.h> /', 'h> /'; qwen-2.5-72b-instruct: 'nt th', 't the', 't th'. Overlapping character fragments are cues, not independent semantic explanations. Full table: `results/top_features.csv`.

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

## Assignment-aligned ranking ablation

One comment-removal condition now covers all four classifiers. Each uses the same saved split, the same newly fitted training-only vectorizer, and its original settings. `python run_all.py` executes both complete conditions. `python run_ablation.py` repeats only the ablation using existing original artifacts.

The original SVM-only ablation was already known. The extension is therefore an exploratory follow-up. Its dated rationale and prediction are in `results/ablation_protocol.md`; it is not presented as an independent preregistered test.

| Model | Original validation F1 | No-comment validation F1 | Rank before / after |
|---|---:|---:|---:|
| Linear SVM | 0.6182 | 0.4903 | 1 / 1 |
| Logistic Regression | 0.5942 | 0.4622 | 2 / 3 |
| Random Forest | 0.5429 | 0.4625 | 3 / 2 |
| Naive Bayes | 0.4670 | 0.3335 | 4 / 4 |

SVM's validation lead over Logistic Regression widened from 2.40 to 2.81 percentage points. This contradicts the follow-up expectation that removing comments would narrow its validation lead. SVM remained first on both validation and test. Random Forest and Logistic Regression swapped validation ranks, but their ablated F1 values differ by only 0.00028. Test ranking stayed unchanged. No significance or robust rank reversal is claimed.

The measured property is a sparse, overlapping vocabulary of source and comment fragments: 50,000 features, 3.81% training density. Strong SVM coefficients include brace-adjacent comments, inline comment markers, and compact control syntax. Every classifier lost Macro-F1 when comments were removed. This supports the usefulness of comments, but the persistent SVM lead means comments alone do not explain why SVM won. Other lexical patterns remain a plausible contributor, not a demonstrated cause. A single split and refitted vocabulary limit causal and generalization claims.

Full validation/test metrics and rankings: `results/ablation_model_results.csv` and `results/ablation_ranking.csv`. The original winner remains selected from original validation results.

## Collection provenance and sources

The authors describe a corpus generated from 300 parameterized C-programming templates, followed by deduplication and compilation filtering. Our source-only experiment does not execute that compilation step. The released archive contains 10,262 distinct prompt strings in our inspection; this observed count must not be replaced with the paper's count of generated task instances.

- Bisztray et al., *I Know Which LLM Wrote Your Code Last Summer: LLM Generated Code Stylometry for Authorship Attribution*, https://arxiv.org/abs/2506.17323 . Cited for collection provenance, not numerical comparison results.
- Official release: https://github.com/LLMauthorbench/LLMauthorbench . The generator notebook is read only to audit template grouping, not executed or used to train the classifiers.
- scikit-learn: https://scikit-learn.org/stable/ . Provides TF-IDF, the four classifiers and metrics. NumPy, pandas, Matplotlib and joblib support numerical arrays, tables, figures and serialization. Versions are recorded in requirements.txt and results/environment.json.

## Submission status

Finished five-slide presentation: [Download the PowerPoint](presentation/LLM_AuthorBench_Assignment.pptx). Member details are intentionally pending at the user's request. The assignment still requires a verified research-use licence, confirmation of the dataset claim, and genuine contributions from every group member. See `docs/ASSIGNMENT_STATUS.md` and `docs/DATASET_PERMISSION.md`. The latter includes permission-request drafts; no messages have been sent.

Presentation source: `presentation/build_deck.mjs`. It uses @oai/artifact-tool from the authoring runtime, separate from the five-package ML environment. The exported PowerPoint is editable and does not require that authoring runtime to view or edit.
