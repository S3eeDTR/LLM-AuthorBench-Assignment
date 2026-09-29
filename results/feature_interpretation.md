# Feature-limit ablation outcome

Prediction committed before execution in 5cdab0d: reducing 50,000 features to 500 would narrow SVM's validation Macro-F1 lead over Logistic Regression and might reverse the pair's order.

Observed validation gap: 1.4869 to 1.3882 percentage points. Narrowing: 0.0987 percentage points. All validation and test ranks remain unchanged. The directional gap prediction holds weakly, but the ranking reversal did not occur. This is not a demonstrated causal explanation of the winner or a statistically established effect.

SVM validation Macro-F1 falls from 0.8062 to 0.7336. Logistic Regression falls from 0.7913 to 0.7197. SVM test Macro-F1 falls from 0.7970 to 0.7169. All models lose F1, consistent with information lost when restricting the vocabulary.

The limit was fixed at 500 before running. No limit sweep was performed. Data, labels, sample assignments, seed and classifier settings remained fixed; comments were retained. Feature selection uses training data only. Vocabulary restriction also changes TF-IDF normalization and the number of candidate features implied by the forest's sqrt setting. Single split, no uncertainty interval, exploratory follow-up after seeing earlier results.
