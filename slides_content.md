# Slide 1 - Dataset & Story
**LLM-Generated Code Authorship Attribution**
- LLM-AuthorBench: 32,000 C programs, eight LLMs, 259 inferred task families.
- Can standard ML identify the generating LLM from C source?
- Connection: source-level foundation for our code-attribution research.
- Source: https://github.com/LLMauthorbench/LLMauthorbench
- Repository: https://github.com/S3eeDTR/LLM-AuthorBench-Assignment

# Slide 2 - Data Handling
- 4,000 programs per class; 10,262 exact prompt strings.
- 0 invalid records; 0 exact duplicates.
- Seed 42; conservative prompt-family grouping; saved assignments.
| split | samples | problems |
| --- | --- | --- |
| train | 21115 | 181 |
| validation | 4777 | 38 |
| test | 6108 | 40 |
- Training-only TF-IDF; comments and identifiers retained.
- **The same inferred programming problem never appears in both training and testing.**
- Limitation: no authoritative IDs; semantic paraphrase leakage cannot be ruled out.

# Slide 3 - Algorithms
```text
C source -> shared character TF-IDF (3-5 grams; 50,000 features)
                  +-- Multinomial Naive Bayes (simple baseline)
                  +-- Logistic Regression
                  +-- Linear SVM
                  +-- Random Forest
```
- Same split, preprocessing, features and seed where supported.
- No heavy tuning; select by validation Macro-F1.
- Chance accuracy: 12.5%; majority test accuracy: 11.90%.

# Slide 4 - Results
| Model | Accuracy | Macro-F1 | Training (s) | Model (MB) |
| --- | --- | --- | --- | --- |
| Naive Bayes | 0.4468 | 0.4309 | 0.22 | 4.640 |
| Logistic Regression | 0.5787 | 0.5731 | 22.78 | 1.383 |
| Linear SVM | 0.5896 | 0.5797 | 11.36 | 2.835 |
| Random Forest | 0.5034 | 0.4645 | 53.34 | 24.326 |
- **Validation-selected winner: Linear SVM.** Table shows held-out test results.
- Time excludes feature extraction; size is compressed classifier only.
- Optional chart: `results/figures/macro_f1_comparison.png`.

# Slide 5 - Why + Ablation
- Winner: **Linear SVM**; 50,000 sparse lexical features (density 3.81%).
- Sparse character cues suit a regularized linear model; top features include comment placement and compact control syntax.
- Removing comments reduced Macro-F1, consistent with useful information in comment wording/style.
- Comments removed; same classifier/settings/split, new training-only TF-IDF.
- Test Macro-F1: **0.5797 -> 0.4287** (-15.10 pp).
- Conclusion: attribution is measurable on held-out inferred task families, with limited generalization claims.
