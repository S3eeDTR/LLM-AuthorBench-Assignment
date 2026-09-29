# Comparing the authors' architecture with our models

## What the Colab experiment matches

The fifth model uses the CodeT5+ encoder, first-token representation, a 768-unit hidden layer, GELU and dropout 0.2, with an eight-class output head. All encoder layers are fine-tuned. It uses precisely our saved rows, labels and inferred problem-family split. Validation Macro-F1 selects the epoch; the separate test set supplies the final comparison. Our four trained baselines remain unchanged.

The notebook exports a five-model comparison, predictions, class metrics, provenance, and (when enabled) comment-removal results. It has not been trained. A win must be observed rather than assumed. Pretraining and contextual representations are possible explanations, but the comments ablation does not isolate their causal contribution.

## Why existing scores do not reproduce the paper

| Measurement | Our measured grouped eight-class test | Authors' five-class reference accuracy |
|---|---:|---:|
| Linear SVM accuracy | 58.96% | 74.60% |
| Random Forest accuracy | 50.34% | 88.00% |
| CodeT5-Authorship accuracy | Not run | 95.40% |

These columns describe different experiments, not a replication gap. Our primary metric is Macro-F1; the released classical code computes weighted F1. Our four assignment choices also differ from their baseline set: Naive Bayes and Logistic Regression are not in their released classical table.

The released code uses five labels and a stratified random 80/20 row split. Its classical TF-IDF has 400 word features and is fitted before that split. Ours fits 50,000 character features on training only. Its neural notebook selects the best epoch on the partition later evaluated as test; ours reserves an independent validation partition. These observations concern the released implementation and do not establish how every paper result was produced.

The paper's table lists a 512-token limit while the released neural notebook uses 1024. Our configurable default is 512. We retain ten epochs, learning rate 5e-5, effective batch size 32, weight decay 0.01, and 200 warmup steps. Memory-conscious microbatching and precision differ. Neural and classical representations, input coverage, CPU/GPU hardware, timing boundaries and saved-model formats differ; equal splits do not make these controlled architecture-only comparisons.

## Reporting after Colab

Report the actual validation-selected winner and its held-out test result, even if CodeT5 loses. Compare comment-removal effects to test sensitivity to comments. Do not substitute the published 95.40% for our measured output. Describe this as an eight-class adaptation and comparison, not an exact reproduction of the headline experiment.

## Sources

- [Paper](https://arxiv.org/abs/2506.17323)
- [Official repository and reported scores](https://github.com/LLMauthorbench/LLMauthorbench)
- [Released neural notebook](https://github.com/LLMauthorbench/LLMauthorbench/blob/6a1c2ac173c774cfbd012f4c81e8d0a51bb61eb7/scripts/5_CodeT5-Authorship_5-class_google_colab.ipynb)
- [Pretrained CodeT5+ model](https://huggingface.co/Salesforce/codet5p-770m)

Local checks: all notebook code cells compile; pinned input checksums and 32,000 split joins pass; comment removal equals the original implementation on all programs. Neural dependency installation and GPU execution have not been tested locally. Dataset licensing and member contributions remain pending as previously documented.
