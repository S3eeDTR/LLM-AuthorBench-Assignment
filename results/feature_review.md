# Feature and leakage review

Reviewed the validation-selected Linear SVM's 15 strongest positive character n-grams for each of eight classes (120 entries). No obvious generator name, task ID, file path, JSON metadata field, or compiler banner appears in this set.

Observed patterns include comment openings adjacent to braces for Claude, capitalized comment fragments such as `For` for Gemini, compact `if(` and `for(` forms for GPT-4.1, inline comments after semicolons for GPT-4o-mini, and pointer/cast syntax for Llama. Short fragments overlap heavily and must not be treated as independent evidence. The character vectorizer collapses repeated whitespace, so these are not exact indentation-width measurements.

The separate whole-source audit contains 26 matches. Manual context inspection found ordinary comments about generated objects or files, ordinary document-author output fields, one generic author-name placeholder, one substring in Hungarian text, and two API model references inside Gemini-generated LLM-client programs. These API references are task content and are not the generator's label. No sample filtering or label redaction was justified by those hits.

Source code is the only model input. Prompt text is used for grouping and is never concatenated with code. Dataset fields such as compiler complexity metrics, checksum, token size and model_name do not enter TF-IDF. Natural source comments are deliberately retained in the main condition; they are not automatically considered accidental dataset headers.

The ablation removes comments with the same estimator settings and a fresh training-only vectorizer. Its measured effect must be read from ablation_results.csv. This is evidence about the contribution of comments under one inferred-task split, not a causal proof of model identity or universal stylistic fingerprints.

The grouping review found and fixed numeric/filename variants, unquoted random histogram input, and several semantic aliases before the final model run. The independent upstream-template audit matched 298 of 300 generator templates and found no matched template spanning multiple inferred groups. Broad families intentionally sometimes combine related but distinct tasks (for example sorting/search variants connected by text similarity); this is conservative for overlap prevention, but means the 259 families are not an authoritative count of unique semantic problems.

Residual limitation: arbitrary paraphrases and algorithmic similarities cannot be ruled out without authoritative IDs or exhaustive expert annotation. Assertions establish disjoint implemented groups and exact prompts/source hashes.


Update after assignment review: the comment-removal condition now includes all four original algorithms, with common ablated matrices. Validation SVM/LR gap widened, so the directional follow-up hypothesis was not supported. See ablation_interpretation.md. Member details remain pending at the user's request.
