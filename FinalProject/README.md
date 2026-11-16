# Final Project

Training-time comment removal for five-LLM code authorship classification.

## Run in Colab

Open `FinalProject_Colab.ipynb`, select **A100 GPU**, then run the numbered cells in order.
The notebook downloads a pinned version of the Python code and the baseline bundle from this repository. No manual ZIP upload is needed.

1. Get the project and install packages.
2. Check the GPU, load the fixed data split, and tokenize.
3. Train the 50% removal method and the 100% removal ablation.
4. Evaluate both, compare with the completed SVM/CodeT5 baselines, and download evidence.

Allow roughly 2–3 hours for both runs on A100; timing varies. `/content` is temporary.
Interrupted training cannot resume. Save the evidence ZIP when finished; optionally download both large checkpoints as well.

## Files

- `FinalProject_Colab.ipynb`: short Colab front end.
- `experiment.py`: data checks, comment removal, model, training and evaluation.
- `requirements-colab.txt`: dependencies; uses Colab's existing GPU PyTorch.
- `baseline_reference.zip`: completed baseline predictions, metadata and SHA256 manifest. Contains no trained weights or new experiment results.
- `run_project.py`: optional GPU command-line runner (`python run_project.py --all`). Install the requirements and GPU PyTorch first.
- `EXPERIMENT_PLAN.txt`: controls, comparisons, limitations and source links.

The 50% and 100% experiments start independently from the pinned pretrained CodeT5 checkpoint. Checkpoints are selected by original-code validation accuracy. Both runs must finish before the notebook evaluates test results.

This extends the adapted paper-critique experiment. New full training runs have not been performed as part of preparing these files. The fixed test set was already examined in the critique, and only one seed is planned. CPU SVM fit time and full neural GPU training time measure different work.

## Sources

- [Paper](https://doi.org/10.1145/3733799.3762964)
- [Authors' code and dataset](https://github.com/LLMauthorbench/LLMauthorbench)
- [Assignment 1 and paper critique](https://github.com/S3eeDTR/LLM-AuthorBench-Assignment)
