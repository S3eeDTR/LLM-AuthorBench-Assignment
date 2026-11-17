CODE ORIGIN — LOCAL WEB DEMO

Double-click START_WEB_APP.cmd in the FinalProject folder, then open:
http://127.0.0.1:8765

The launcher uses the existing GPU-project Python environment, with CPU inference
by default. It also checks .venv in FinalProject or system Python if needed.
For another machine, install PyTorch and requirements-colab.txt first.
From this folder: python server.py --device cpu
Optional compatible GPU: python server.py --device cuda

First analysis loads a 1.34 GB checkpoint. Comparing both models loads each in turn.
Keep the server window open. Close it or press Ctrl+C to stop the app.
Input is never compiled, executed, saved, or sent to a remote service.
The local server binds only to 127.0.0.1. This is a research demo, not a public deployment.

SUPPORTED TASK
C code, five known LLM labels, 512 model tokens. Long inputs are truncated with a warning.
Scores are softmax outputs, not calibrated confidence or evidence of provenance.
There is no human-code detector or unknown-model rejection.
The 50% model is the default for balancing original and stripped input performance.
Both checkpoints were identified against saved test predictions; see models/checkpoint_manifest.json.

The example button inserts an illustrative program with unknown authorship.
It is not a measured test case or evidence that a prediction is correct.

BENCHMARK SELECTOR
Use Load a benchmark sample above the editor to choose one original test-set
program per model. The recorded label and row ID are shown separately from
the source. Selection does not depend on classifier correctness.
