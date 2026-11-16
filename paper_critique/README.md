# Paper Critique: code

This folder extends [Assignment 1](../README.md) for the **CSBP711 Paper Critique**. It is separate from the later Course Project. It contains runnable code and a clean Colab notebook; reports, slides, result bundles, datasets and model weights are excluded.

[Open the neural notebook in Colab](https://colab.research.google.com/github/S3eeDTR/LLM-AuthorBench-Assignment/blob/main/paper_critique/CodeT5_Reality_Check_Colab.ipynb)

## Sources and adaptations

Primary paper: Bisztray et al., *I Know Which LLM Wrote Your Code Last Summer*, [DOI](https://doi.org/10.1145/3733799.3762964).

Authors' [public implementation](https://github.com/LLMauthorbench/LLMauthorbench/tree/6a1c2ac173c774cfbd012f4c81e8d0a51bb61eb7), revision `6a1c2ac173c774cfbd012f4c81e8d0a51bb61eb7`, especially `scripts/5_CodeT5-Authorship_5-class_google_colab.ipynb` and `scripts/4_TRAIN_Machine_learning_google_colab.ipynb`.

The neural adaptation keeps `Salesforce/codet5p-770m`, the first-token representation, a two-layer head with GELU and dropout 0.2, the `code: ` prefix and the main shared training settings. Changes are explicit:

- Use the saved Assignment 1 train/validation/test partitions instead of the authors' random 80/20 split.
- Select checkpoints by **validation accuracy**; test predictions do not select checkpoints.
- Use **512 tokens**, as reported in the paper. The shared notebook defaults to 1,024.
- Use fp32 parameters with bf16 autocast instead of the authors' bf16 parameters; the optimizer is implemented directly with AdamW.
- Save accuracy, Macro-F1, weighted F1, predictions, confusion matrices, training history, package versions, model revision and data hashes.
- Test fixed models after removing comments from test inputs only. This is a distribution-shift check, **not** retraining on comment-free code.

This is a controlled adaptation, not an exact reproduction of the published score. The notebook is an output-cleared source copy, not the executed evidence notebook.

## Files

| File | Purpose |
|---|---|
| `CodeT5_Reality_Check_Colab.ipynb` | Self-contained A100 setup, verified downloads, timing trial, training and evidence download |
| `train_codet5.py` | Neural adaptation; ten epochs, effective batch 32, seed 42 |
| `project_utils.py` | Saved-split checks and comment removal preserving quoted literals |
| `prepare_data.py` | Download and hash-check the public dataset when running locally |
| `run_reality_check.py` | Shared word-400 SVM diagnostics and Assignment 1 SVM comparison |
| `requirements-gpu.txt` | Neural dependency versions; CUDA PyTorch comes from the runtime |

## Run in Colab

1. Open the notebook above and select an **A100 GPU** runtime. The script requires CUDA and bfloat16 support.
2. Run the setup and verified data-loading cells in order.
3. Run the timing trial; it does not count as completed training.
4. Run the full training cell. The best checkpoint is selected using validation accuracy.
5. Download the evidence bundle from the final cell. Download `best_model.pt` separately if you want to retain weights.

The notebook downloads the pinned dataset and saved split automatically. It does not require uploading private files or mounting Google Drive. It embeds the helper and training scripts, so it also works as a standalone notebook.

## Run locally from the repository root

CPU comparison:

```powershell
python -m pip install -r requirements.txt
python paper_critique/prepare_data.py
python paper_critique/run_reality_check.py
```

The CPU script reuses trusted local Assignment 1 model files when present. On a fresh clone it retrains the identical fixed character TF-IDF/LinearSVC configuration and records that choice. Package versions can affect rerun results. Only load model files you trust.

For neural training, use a separate environment with compatible CUDA-enabled PyTorch:

```powershell
python -m pip install -r paper_critique/requirements-gpu.txt
python paper_critique/prepare_data.py
python paper_critique/train_codet5.py --timing-only
python paper_critique/train_codet5.py
```

An optional `--gradient-checkpointing` flag reduces activation memory. If you change micro-batch size, adjust `--accumulation` to preserve the effective batch size. Record every change rather than silently altering the experiment. The original model cache download is large.

## Evaluation boundaries

The CPU `random_full_fit_DIAGNOSTIC` condition deliberately recreates the authors' vocabulary-fitting order. It is diagnostic and must not be treated as an unbiased benchmark. The corrected condition fits vocabulary on training samples only. Random and grouped partitions also differ in training volume and test composition, so their difference cannot isolate a causal task-overlap effect.

Use training/validation to fix settings before viewing final test results. Do not select the best test seed or input length. Results and weights are written under `paper_critique/results/`, which is ignored by Git.

## AI use

An AI assistant helped audit the authors' code, prepare this adaptation and notebook, execute CPU comparisons, and verify uploaded evidence. The student executed the neural experiment in Colab. Review the code and cite the original implementation when using it.
