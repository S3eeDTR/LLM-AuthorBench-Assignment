# FinalProject: LLM code attribution

Training and local inference code extending Assignment 1 and the paper critique. Reports and slides are intentionally excluded.

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
| 50% / kept | 27/36 | 27/36 | 12/12 | 12/12 |
| 50% / removed | 28/36 | 27/36 | 12/12 | 12/12 |
| 100% / kept | 28/36 | 28/36 | 2/12 | 3/12 |
| 100% / removed | 28/36 | 28/36 | 12/12 | 12/12 |

These exploratory results do not show a consistent accuracy improvement from chunking. Gemini is a newer version and is reported separately.

## Scope and sources

The classifier always chooses among five known LLM labels for C code. Scores are uncalibrated. It cannot prove authorship, detect human code or reject unknown generators.

- [Primary paper](https://doi.org/10.1145/3733799.3762964)
- [Authors' implementation](https://github.com/LLMauthorbench/LLMauthorbench)
- [CodeT5+](https://aclanthology.org/2023.emnlp-main.68/)
