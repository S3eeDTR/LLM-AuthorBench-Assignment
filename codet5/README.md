# Exploratory CodeT5 run

[Open in Google Colab](https://colab.research.google.com/github/S3eeDTR/LLM-AuthorBench-Assignment/blob/main/codet5/CodeT5_Exploratory.ipynb) · [Notebook](CodeT5_Exploratory.ipynb) · [Saved results](results/)

This is a separate exploratory experiment using **Salesforce/codet5-base**, masked mean pooling and a five-class head. It uses original code with comments and the same saved dataset split. It is not a reproduction of the authors' CodeT5-Authorship or a full neural ablation study.

**No CodeT5 comment-removal run was completed.** Comment removal and the 500-feature ablation in the main project apply only to the four classical classifiers.

## Run

1. Open the notebook in a fresh Colab A100 GPU runtime (or another GPU supporting bfloat16).
2. Run installation section 1, restart the session once, then run sections 2-10 in order.
3. Review the timing estimate before starting training in section 8. Five epochs are requested; early stopping uses validation Macro-F1.
4. Download results and optionally the checkpoint before the runtime disconnects.

The notebook uses a working directory relative to the runtime, downloads verified data and split files, and does not require a personal drive path. Outputs are cleared in this rerunnable source copy; the actual completed run's evidence is supplied separately in `results/`. It has not been retrained for this publication.

## Completed run

| Measure | Result |
|---|---:|
| Epochs completed | 5 |
| Selected epoch | 3 |
| Validation Macro-F1 | 0.799756 |
| Test accuracy | 0.769628 |
| Test Macro-F1 | 0.772514 |
| Test programs | 3,872 |

`settings.json` records the original model revision, package versions, GPU and settings. `training_history.csv` records all five epochs. Predictions, class scores and the confusion matrix support the test metrics. Model weights and raw source data are not uploaded.

The input is limited to 256 tokenizer tokens. A seed aids repeatability but does not guarantee identical GPU results. Do not tune using the reported test scores.
