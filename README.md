# LLM authorship classification

Predict which LLM generated a C program. We compare four classical classifiers and an additional CodeT5 encoder classifier on the same five-class split.

[Download the five slides](submission/Assignment.pptx) · [Experiment code](assignment.py) · [Saved classical results](results/metrics.csv)

## Dataset

Data comes from **[LLM-AuthorBench](https://github.com/LLMauthorbench/LLMauthorbench)**, associated with the paper [I Know Which LLM Wrote Your Code Last Summer](https://arxiv.org/abs/2506.17323). Our instructor approved the dataset by email.

We use **20,000 C programs**, with 4,000 from each author: Claude 3.5 Haiku, DeepSeek Chat, Gemini 2.5 Flash Preview, GPT-4.1 and Llama 3.3 70B Instruct. Known author labels, balanced classes and a common programming language make this dataset suitable for comparing code-attribution methods.

| Partition | Programs | Inferred task groups |
|---|---:|---:|
| Training | 13,150 | 181 |
| Validation | 2,978 | 38 |
| Test | 3,872 | 40 |

The saved split keeps related task groups together. Checks found no empty programs or exact source duplicates in the selected subset, and no shared groups, exact prompts or source hashes across partitions. Groups are inferred, so unrecognized semantic overlap may remain. Matching the paper's five labels does not reproduce its experimental setup.

## Experiments

1. **Original code:** compare Naive Bayes, Logistic Regression, Linear SVM and Random Forest using character TF-IDF with up to 50,000 patterns of 3–5 characters. Fit the vocabulary on training data only.
2. **Remove comments:** repeat the four-model comparison with the same split and settings.
3. **Restrict features (2B):** retain comments but reduce the vocabulary to 500 patterns.
4. **CodeT5:** fine-tune the pretrained encoder with mean pooling and a five-class head. Use the first 256 tokens, batch size 16 and learning rate 0.00002. Five epochs were completed on an A100; validation Macro-F1 selected epoch 3. This optional experiment ran in a separate local Colab notebook.

## Results

Held-out test results on the same **3,872 programs**:

| Model | Accuracy | Macro-F1 | Time (s) | Saved size (MB) |
|---|---:|---:|---:|---:|
| Naive Bayes | 62.68% | 0.6278 | 0.12 | 2.898 |
| Logistic Regression | 76.89% | 0.7704 | 15.17 | 0.863 |
| **Linear SVM** | **79.75%** | **0.7970** | **5.36** | **1.767** |
| Random Forest | 69.01% | 0.6796 | 20.11 | 9.935 |
| CodeT5 encoder + mean pooling | 76.96% | 0.7725 | 291.71* | 438.494* |

**Linear SVM performed best.** Its validation Macro-F1 was 0.8062, compared with CodeT5's 0.7998. CodeT5 ranked second on test Macro-F1. Its predictions were verified against the saved test IDs and labels. CodeT5's tokenizer, pretraining and truncation differ from TF-IDF, so this comparison does not isolate one cause of performance differences.

Classical times measure CPU fitting only, excluding TF-IDF. Their sizes are compressed classifier files. *CodeT5 time includes GPU training, validation and saving; its size is an uncompressed encoder-and-head checkpoint. These are different timing and storage conventions, not a controlled speed or memory comparison.*

### SVM confusion matrix

![SVM test confusion matrix](results/figures/svm_confusion_matrix.png)

Rows are actual authors; columns are predicted authors. The diagonal contains **3,088 correct predictions**. The largest error cell is 199 DeepSeek programs predicted as Llama.

### What the ablations showed

![Validation scores with 50,000 and 500 features](results/figures/feature_ablation.png)

Removing comments reduced every model's score, but SVM's validation lead over Logistic Regression grew from **1.49 to 3.03 percentage points**, contrary to the prediction. Limiting features to 500 narrowed that lead to **1.39 points**, a change of only **0.10 points**. Neither test changed the rankings.

Comments and a richer vocabulary helped performance. These exploratory, single-split results do not conclusively explain why SVM won or establish statistical significance.

<details>
<summary>Mathematics: TF-IDF, SVM and Macro-F1</summary>

**TF-IDF** means Term Frequency–Inverse Document Frequency. It weights character patterns by their repetition in one program and their frequency across training programs. We use logarithmic term frequency, smoothed IDF and L2 normalization.

**SVM scoring:** multiply each feature by its learned weight, sum the products, add the intercept and select the author with the largest score.

![SVM scores and prediction](results/figures/svm_score.png)

Here x is the feature vector, w contains learned weights, b is the intercept, and k identifies an author. We train five one-vs-rest classifiers.

![SVM training objective](results/figures/svm_objective.png)

The first term penalizes large weights and intercepts. The sum is squared-hinge loss across training examples. The binary label y is +1 for this author and −1 for the others. C = 1 balances loss against regularization. Our LinearSVC uses intercept scaling 1, so the intercept is regularized too.

![F1 and Macro-F1](results/figures/f1_equation.png)

P is precision (correct predictions of a class divided by all predictions of that class). R is recall (correct predictions of a class divided by its actual examples). Macro-F1 averages the five class F1 scores equally. Undefined scores are set to zero. Models and checkpoints are selected using validation scores, not test scores.

Reference: [scikit-learn LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

</details>

## Run the classical experiments

Use Python 3.12 in the project folder:

```text
python -m pip install -r requirements.txt
python assignment.py
python make_figures.py
```

The dataset downloads automatically if missing. `assignment.py` contains data checks, shared training steps and labelled experiments. `make_figures.py` creates charts and equation images from saved results. These commands do not run CodeT5. Rerunning replaces generated classical outputs.

<details>
<summary>Reproducibility and source details</summary>

Dataset snapshot: upstream commit `6a1c2ac173c774cfbd012f4c81e8d0a51bb61eb7`, inspected September 29, 2026. Archive SHA-256: `e24399b7b05b812c5a1148ef44245348859eaa74b1140523cb695f984337f24c`.

`data/splits.csv` stores the seed-42 task-family assignments. The earlier preparation normalized prompt parameters and merged audited aliases and similar descriptions. Its [source](https://github.com/S3eeDTR/LLM-AuthorBench-Assignment/blob/33b42098d3bfdeddd8d94c77326a22fb8847be0a/src/prepare_data.py) remains available. The current script reuses and verifies the split.

Model settings appear in `assignment.py`; package versions appear in `requirements.txt` and `results/environment.json`. Classical results are saved separately for each condition. Earlier outcomes were known before the ablations, so their interpretation is exploratory.

Instructor approval confirms the dataset choice for this assignment. An explicit dataset reuse licence has not been verified. Raw data, model weights, the personal study PDF and the optional notebook are not uploaded here.

</details>

## Group members

- Saeed Alshehhi
- Khalifa Alblooshi
- Maitha Alblooshi
- Fatima Alnuaimi
- Fatima Alsadi

## AI assistance

AI assisted with code, figures, documentation and explanations. Group members review the work and are responsible for understanding and presenting it.
