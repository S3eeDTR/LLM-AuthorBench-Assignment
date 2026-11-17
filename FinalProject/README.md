# FinalProject

Comment-removal training for five-LLM C-code authorship classification, extending Assignment 1 and the paper critique.

## Local web platform

The web app accepts pasted C code, a local `.c` file, or one of five original benchmark samples. It supports the 50% and 100% training conditions, optional input-comment removal, ranked scores, and a 512-token truncation warning.

### Setup after cloning

Install PyTorch for your machine, then install the project dependencies:

```text
python -m pip install -r FinalProject/requirements-colab.txt
```

The large trained checkpoints are **not included in GitHub**. Use the downloaded project weights. Their verified mapping is:

- `best_model (1).pt`: 50% removal.
- `best_model.pt`: 100% removal.

Prepare the weights and tokenizer from the repository root:

```text
python FinalProject/webapp/setup_models.py --mixed "path/to/best_model (1).pt" --stripped "path/to/best_model.pt"
```

The setup script checks checkpoint SHA256 values, copies weights into the ignored `FinalProject/models/` folder, and downloads only the pinned tokenizer/config files. For offline setup, add `--tokenizer-source "path/to/existing/tokenizer"` pointing to the project's tokenizer folder, including `config.json`.

Start the platform:

```text
python FinalProject/webapp/server.py --device cpu
```

Open **http://127.0.0.1:8765**. On Windows, `FinalProject/START_WEB_APP.cmd` also starts the app. It checks for an existing project Python environment before falling back to system Python. Optional compatible CUDA inference: `--device cuda`. CPU inference has been tested with the saved models.

The server binds to localhost. Submitted code is not executed or saved. Scores are uncalibrated and the classifier always chooses among five known LLM labels. It cannot establish provenance, detect human authorship, or reject an unseen generator.

## Benchmark examples

Use **Load a benchmark sample** above the editor to choose one original test-set program per model:

| Recorded generator | Test row |
|---|---:|
| Claude 3.5 Haiku | 74 |
| DeepSeek Chat | 23 |
| Gemini 2.5 Flash preview | 87 |
| GPT-4.1 | 18 |
| Llama 3.3 70B Instruct | 13 |

`test_samples/` contains the unedited code, a label manifest and source hashes. Selection uses the first test row per label, without consulting classifier predictions. These are previously evaluated benchmark examples, not fresh independent tests. Some exceed the 512-token input window.

## Colab experiment

[Open the notebook in Colab](https://colab.research.google.com/github/S3eeDTR/LLM-AuthorBench-Assignment/blob/main/FinalProject/FinalProject_Colab.ipynb)

Select A100 and run the numbered cells. The notebook downloads a pinned version of `experiment.py` and the fixed baseline reference. It trains both conditions independently and evaluates after both finish. It does not resume interrupted training.

Completed test results on the saved 3,872-row partition:

| Training condition | Original accuracy | Stripped accuracy | Stripped Macro-F1 |
|---|---:|---:|---:|
| Character SVM | 79.75% | 40.81% | .3514 |
| CodeT5 0% | 85.90% | 49.66% | .4349 |
| CodeT5 50% | 83.24% | 79.57% | .7919 |
| CodeT5 100% | 60.77% | 81.74% | .8185 |

Mixed training improves stripped-code robustness, with a 2.66 percentage point original-code accuracy loss. This exceeds the planned two-point tolerance. Results cover one seed, five known generators, C code, and a previously examined test set.

The code and test samples are shared here. Reports, presentation files, local environment files and trained weights are excluded from this folder's publication.

## Sources

- [Primary paper](https://doi.org/10.1145/3733799.3762964)
- [CodeT5+](https://aclanthology.org/2023.emnlp-main.68/)
- [Authors' repository](https://github.com/LLMauthorbench/LLMauthorbench)
- [Assignment and critique code](https://github.com/S3eeDTR/LLM-AuthorBench-Assignment)
