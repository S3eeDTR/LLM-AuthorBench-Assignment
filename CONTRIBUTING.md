# Genuine group contributions

This initial implementation is AI-assisted. Do not attribute it to five fabricated authors or manufacture commits.

Each member should take a responsibility below, inspect the module and outputs, make a substantive improvement or verification contribution, and commit using their own configured Git identity. Review changes as a group. A single initial scaffolding commit does not satisfy five-member participation.

1. Dataset inspection and split review: src/inspect_dataset.py and src/prepare_data.py. Review prompt families manually, especially shortened or paraphrased variants.
2. Feature engineering: src/features.py. Verify training-only fitting and assess representation limitations.
3. Models and evaluation: src/train_models.py and src/evaluate.py. Review per-class errors and timing boundaries.
4. Ablation and features: src/ablation.py. Extend lexical edge-case tests and interpret top features.
5. Reporting: src/report.py, README.md, slides_content.md. Review the five-slide story and reproduction instructions.

Record actual review and changes, and rerun affected experiments if the method changes. Do not merely split generated files into commits under other people's names.
