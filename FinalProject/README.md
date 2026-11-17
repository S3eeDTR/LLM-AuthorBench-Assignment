# FinalProject: LLM code attribution

Training and local inference code extending Assignment 1 and the paper critique. Reports and slides are intentionally excluded.

## Results: accuracy percentages

### Held-out benchmark

All conditions use the same 3,872 test programs from five generator classes, with seed 42 and first-512-token inference. Training removal percentages describe how often training inputs have comments removed. Input columns describe whether test comments are kept or removed.

| Model / training condition | Comments kept: accuracy | Comments removed: accuracy |
|---|---:|---:|
| Character TF-IDF SVM baseline | 79.75% | 40.81% |
| CodeT5 baseline: 0% removal | 85.90% | 49.66% |
| CodeT5: 50% removal | 83.24% | 79.57% |
| CodeT5: 100% removal | 60.77% | 81.74% |

The 50% model improves stripped-code accuracy by **29.91 percentage points** over the CodeT5 baseline, with a **2.66-point** decrease on original code. Stripped-code Macro-F1 rises from **0.4349 to 0.7919**. The clean-accuracy loss exceeds the prespecified two-point tolerance, so the combined target was not fully met. The 100% model has the highest stripped-code accuracy (**81.74%**) but is substantially weaker when comments are kept. These are single-seed results on a previously examined test set.

### Fresh generated code

The earlier first-512 study recorded **75.00% (27/36)** comparable-ID matches for the 50% model with comments kept or removed. The 100% model recorded **66.67% (24/36)** with comments kept and **69.44% (25/36)** after removal. Newer Gemini family matches were 12/12 for both 50% conditions, and 0/12 kept versus 12/12 removed for the 100% model.

The newer 48-program study is compared fairly below: both coverage methods use identical code. These are small exploratory label-match rates, not benchmark accuracy estimates. Gemini's newer version is evaluated separately, leaving 36 comparable-ID programs. Repeated inference conditions do not add independent programs.

## Colab notebooks

- [Final project training](https://colab.research.google.com/github/S3eeDTR/LLM-AuthorBench-Assignment/blob/main/FinalProject/FinalProject_Colab.ipynb): select an A100 runtime and run the numbered cells in order.
- [Paper critique baseline](https://colab.research.google.com/github/S3eeDTR/LLM-AuthorBench-Assignment/blob/main/paper_critique/CodeT5_Reality_Check_Colab.ipynb): separate baseline experiment.

The final notebook pins training revision `2e9104d86defa2e67c32154ff9687bfde8821d6a` to reproduce the completed experiment. Web updates do not change that training implementation. Notebooks have outputs cleared; keep executed notebooks and evidence locally. Retraining is unnecessary when using existing checkpoints.

## Setup

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m pip install torch
python -m pip install -r FinalProject/requirements-colab.txt
python FinalProject/webapp/setup_models.py --mixed "path/to/best_model (1).pt" --stripped "path/to/best_model.pt"
python FinalProject/webapp/server.py --device cpu
```

Weights are excluded from GitHub. `best_model (1).pt` is the verified 50% removal checkpoint; `best_model.pt` is the 100% checkpoint. Setup checks SHA256 values and downloads pinned tokenizer/config assets. Add `--tokenizer-source "path/to/tokenizer"` for offline assets, including `config.json`.

On Mac, open Terminal inside `FinalProject`, run `bash setup.command`, prepare weights, then `bash start.command`. Windows users may use `START_WEB_APP.cmd`.

## Platform pages

Visit http://127.0.0.1:8765 after startup.

| Page | Purpose |
|---|---|
| `/` | Navigation hub |
| `/live` | Analyze one C program with either classifier or both |
| `/generate` | Shared OpenRouter prompt, six prompt presets, local batch classification |
| `/study` | Six tasks, two requests per generator, saved progress and local retesting |
| `/results` | Upload evidence ZIPs and explore results |

OpenRouter generation requires your key and consumes credits. Local classification/retesting makes no generation requests. Keys are not saved to study files or downloads. Manual inference input is never executed or saved; generated study programs are saved as experiment evidence. The service binds to localhost.

**Test held-out examples** replaces the active batch with five benchmark samples. **Test all** analyzes the current batch. Import a previously downloaded generation JSON to restore your OpenRouter outputs without paying to regenerate them.

## Full-code chunks and saved studies

Both classifiers support comments kept/removed and first512/full-code coverage. Chunk mode uses up to 508 code tokens per window, 64-token overlap, and the trained prefix plus special tokens within 512 model tokens. Scores are averaged equally across chunks. Full coverage does not guarantee higher accuracy.

To reuse a saved study: select its run, **View saved study**, choose coverage, then **Retest saved code locally (no API key)**. Existing generations are preserved, and only missing analyses are added. Both coverage methods stay separate. Progress saves under `study_results/<run_id>/`; download the ZIP as a backup. With four generators, the study uses 48 generation requests and 192 local classifications per coverage method (384 for both).

A fresh clone excludes private evidence and weights. Upload your downloaded benchmark/study ZIPs on `/results`. Keep the Python service running while a study runs.

## Paired comparison and checks

After model setup, recompute both coverage methods on identical code from a completed 48-program study containing chunked results:

```bash
python FinalProject/tools/compare_study_coverage.py --study-zip "path/to/study_results.zip" --output "coverage_comparison"
python FinalProject/tests/check_study.py
```

The comparison makes no API requests. The test uses mocked generations to check persistence, key-free retesting, coverage separation and avoidance of duplicate requests.

`results/chunk_comparison/` holds small verified summaries and paired labels with code hashes. Earlier run `376530e8...` and newer run `a83921e9...` contain different generated code; comparing runs directly cannot isolate chunking. Local reruns instead use the newer run's identical code:

| Classifier / comments | First512 ID matches | Chunk ID matches | First512 Gemini family | Chunk Gemini family |
|---|---:|---:|---:|---:|
| 50% / kept | 75.00% (27/36) | 75.00% (27/36) | 100.00% (12/12) | 100.00% (12/12) |
| 50% / removed | 77.78% (28/36) | 75.00% (27/36) | 100.00% (12/12) | 100.00% (12/12) |
| 100% / kept | 77.78% (28/36) | 77.78% (28/36) | 16.67% (2/12) | 25.00% (3/12) |
| 100% / removed | 77.78% (28/36) | 77.78% (28/36) | 100.00% (12/12) | 100.00% (12/12) |

These exploratory results do not show a consistent accuracy improvement from chunking. Gemini is a newer version and is reported separately.

## Scope and sources

The classifier always chooses among five known LLM labels for C code. Scores are uncalibrated. It cannot prove authorship, detect human code or reject unknown generators.

- [Primary paper](https://doi.org/10.1145/3733799.3762964)
- [Authors' implementation](https://github.com/LLMauthorbench/LLMauthorbench)
- [CodeT5+](https://aclanthology.org/2023.emnlp-main.68/)
