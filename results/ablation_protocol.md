# Follow-up ablation protocol

Recorded before running the extended four-model ablation on 29 September 2026.

Known already: original results for all four algorithms, and an SVM-only comment-removal test Macro-F1 of 0.4287 (original 0.5797). Therefore this is an exploratory follow-up, not a preregistered or independent confirmatory experiment.

Data-property explanation: model-specific comment wording and placement create many overlapping sparse character n-grams. The main SVM coefficients show comment openings and inline comment boundaries among strong class cues. These cues may explain part of its advantage over alternative classifiers.

Directional expectation for the extension: removing comments will reduce SVM's advantage over Logistic Regression and may change their ranking. Compare their Macro-F1 gap before and after removal, using validation as the primary ranking comparison and test as a descriptive check. Also report all four ranks and absolute changes, including an unchanged or opposite ranking.

Freeze seed 42, saved problem-family split, all classifier hyperparameters, 50,000 character features, and the lexical comment-removal procedure. Fit one new vectorizer on ablated training code only and share those matrices across all four algorithms. No tuning, alternative ablation search or re-splitting in response to outcomes. Retain the original validation-selected winner regardless of the follow-up ranking.

A ranking change would support a relative dependence explanation under this split. If the SVM lead persists or widens, comments alone do not explain its advantage. The ablation can also change the vocabulary, so it cannot isolate a single comment feature or prove causality. Single-split point estimates carry no significance claim.
