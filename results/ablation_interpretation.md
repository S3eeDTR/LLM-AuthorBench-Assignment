# Four-model ablation interpretation

SVM's validation lead over Logistic Regression widened from 2.40 to 2.81 percentage points. This contradicts the follow-up expectation that removing comments would narrow its validation lead. SVM remained first on both validation and test. Random Forest and Logistic Regression swapped validation ranks, but their ablated F1 values differ by only 0.00028. Test ranking stayed unchanged. No significance or robust rank reversal is claimed.

The measured property is a sparse, overlapping vocabulary of source and comment fragments: 50,000 features, 3.81% training density. Strong SVM coefficients include brace-adjacent comments, inline comment markers, and compact control syntax. Every classifier lost Macro-F1 when comments were removed. This supports the usefulness of comments, but the persistent SVM lead means comments alone do not explain why SVM won. Other lexical patterns remain a plausible contributor, not a demonstrated cause. A single split and refitted vocabulary limit causal and generalization claims.

| Model | Original validation F1 | No-comment validation F1 | Rank before / after |
|---|---:|---:|---:|
| Linear SVM | 0.6182 | 0.4903 | 1 / 1 |
| Logistic Regression | 0.5942 | 0.4622 | 2 / 3 |
| Random Forest | 0.5429 | 0.4625 | 3 / 2 |
| Naive Bayes | 0.4670 | 0.3335 | 4 / 4 |
