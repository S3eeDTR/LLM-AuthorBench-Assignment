# LLM authorship assignment

**Five LLM classes. Two simple experiments, plus optional CodeT5.**

We use 20,000 C programs to test whether code patterns identify the LLM that wrote them. This connects to our research topic of LLM-generated code attribution.

## Run

[Open in Google Colab](https://colab.research.google.com/github/S3eeDTR/LLM-AuthorBench-Assignment/blob/main/assignment.ipynb) for Experiments 1 and 2 on CPU. Optional Experiment 3 requires a GPU.

Or use Python 3.12 locally:

```text
python -m pip install -r requirements.txt
python assignment.py
```

The dataset downloads automatically if missing. Start with [assignment.py](assignment.py). The notebook adds optional Experiment 3 to the same first two experiments.

## Experiments

**Experiment 1: original code.** Train Naive Bayes, Logistic Regression, Linear SVM and Random Forest on the same character TF-IDF features. Choose the highest validation Macro-F1, then report test scores.

**Experiment 2: remove comments.** Repeat with the same split, seed and model settings. Compare scores and rankings. The prediction and its exploratory status are in [PLAN.md](PLAN.md).

**Experiment 3 (Colab only): CodeT5.** Fine-tune the authors' encoder-based architecture on the same five classes and split. Run Setup, Shared code and Data preparation, then the four labeled Experiment 3 steps. GPU training can take many hours; Drive checkpoints are enabled. This experiment has not been trained here, so no CodeT5 scores are claimed.

## Results

| Model | Accuracy | Macro-F1 | Fit seconds | Size MB |
|---|---:|---:|---:|---:|
| Naive Bayes | 62.68% | 0.6278 | 0.12 | 2.898 |
| Logistic Regression | 76.89% | 0.7704 | 15.17 | 0.863 |
| Linear SVM | 79.75% | 0.7970 | 5.36 | 1.767 |
| Random Forest | 69.01% | 0.6796 | 20.11 | 9.935 |

Linear SVM won on validation with Macro-F1 **0.8062**. Removing comments lowered every model's score but left the ranking unchanged. SVM test Macro-F1 fell from **0.7970 to 0.6032**. Its validation lead over Logistic Regression grew from **1.49 to 3.03 percentage points**, contrary to our prediction. Comments help attribution, but do not establish why SVM uniquely wins.

Fit time excludes TF-IDF. Model size is the compressed classifier, excluding the shared vectorizer. These are single-run measurements without confidence intervals.

## Where things are

- [assignment.py](assignment.py): all experiment code, clearly labeled.
- [assignment.ipynb](assignment.ipynb): the same steps in Colab.
- [results/](results/): separate `experiment_1_original` and `experiment_2_no_comments` folders, plus combined tables.
- [Five slides](submission/Assignment.pptx).

<details>
<summary>Dataset, preprocessing and reproducibility details</summary>


[LLM-AuthorBench](https://github.com/LLMauthorbench/LLMauthorbench) contains 32,000 programs from eight LLMs. We select the paper's five labels: `gpt-4.1`, `deepseek-chat`, `claude-3.5-haiku`, `gemini-2.5-flash-preview-05-20` and `llama-3.3-70b-instruct`. Each contributes 4,000 programs. The authors describe generation from 300 parameterized C-programming templates followed by deduplication and compilation filtering. We read the programs as text and do not execute them.

Version: upstream commit `6a1c2ac173c774cfbd012f4c81e8d0a51bb61eb7`. Archive SHA-256: `e24399b7b05b812c5a1148ef44245348859eaa74b1140523cb695f984337f24c`. Inspected September 29, 2026. The original user-supplied download date is unknown.

**Licence: unverified.** Public download availability does not establish research-use permission. No explicit dataset licence was found during the earlier repository review. Confirm permission before submission.

The five-class data has zero missing required values, empty programs or exact source duplicates. We reuse the existing task-family split, filtered to the five labels: 13,150 training, 2,978 validation and 3,872 test programs. All five classes occur in every split. Counts are in `results/class_counts.csv`.

`data/splits.csv` is a committed preprocessing artifact, so rerunning does not require the earlier grouping code. That preparation normalized prompt parameters, joined audited task aliases and similar descriptions, then assigned 259 inferred families using seed 42. Its source remains available at [the earlier preparation commit](https://github.com/S3eeDTR/LLM-AuthorBench-Assignment/blob/33b42098d3bfdeddd8d94c77326a22fb8847be0a/src/prepare_data.py). We check separation of families, exact prompts and source hashes. The dataset has no official task IDs: inferred groups may overmerge tasks, and unrecognized semantic overlap remains a limitation.


Both experiments use training-only character TF-IDF: 3–5 grams, up to 50,000 features, sublinear frequency, case preserved and float32 values. The analyzer normalizes repeated whitespace. Model parameters are together in `run_experiment`. Stochastic models use seed 42. There is no tuning search. Uniform-chance accuracy is 20%.

The five labels match the [paper](https://arxiv.org/abs/2506.17323), but our group split, validation partition, features and algorithm choices differ. This is not a reproduction of the paper's scores. Earlier eight-class results were known before this rebuild; the ablation remains exploratory.

Local versions are pinned in `requirements.txt` and recorded in `results/environment.json`. Colab keeps its preinstalled scientific libraries to avoid dependency conflicts and records them in `colab_environment.json`; numerical differences between environments are possible. The recorded run used Python 3.12.14. Trained models, raw data and temporary files stay local and are ignored by Git. The old implementation is recoverable from Git history.

</details>

## Before submission

Confirm the dataset's research-use permission and instructor claim. **Licence remains unverified.** Member names and actual contributions will be added by the group; each member must contribute under their own Git identity. The ablation's failed prediction is reported honestly and does not fully establish the requested explanation of the winner.

## AI use and sources

AI assisted with writing, simplifying, checking and explaining the code and documentation. Group members must review and understand it. Sources: [LLM-AuthorBench](https://github.com/LLMauthorbench/LLMauthorbench), [paper](https://arxiv.org/abs/2506.17323), and [scikit-learn](https://scikit-learn.org/stable/). Supporting libraries are NumPy, pandas and joblib.
