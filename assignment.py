"""Run the four-model assignment and its comment-removal experiment."""
from pathlib import Path
import hashlib
import json
import os
import re
import zipfile
import urllib.request
from time import perf_counter

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
OUTPUT.mkdir(exist_ok=True)
# Keep temporary files beside the project and limit CPU thread usage.
(ROOT / "tmp").mkdir(exist_ok=True)
for name in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER"):
    os.environ[name] = str(ROOT / "tmp")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "4"

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42

# Match strings and character literals as well as comments, so quoted // stays intact.
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*(?:\n|$)|/\*.*?\*/', re.S)


def remove_comments(code):
    code = code.replace('\\\r\n', '').replace('\\\n', '')

    def replace(match):
        text = match.group()
        if text.startswith('//') or text.startswith('/*'):
            return ' ' + '\n' * text.count('\n')
        return text

    return TOKEN.sub(replace, code)


def run_experiment(data, condition):
    """Use the same steps for both experiments so the comparison is fair."""
    labels = sorted(data.label.unique())
    results = []
    folder_name = "experiment_1_original" if condition == "original" else "experiment_2_no_comments"
    folder = OUTPUT / folder_name
    folder.mkdir(exist_ok=True)
    model_folder = ROOT / "models" / folder_name
    model_folder.mkdir(parents=True, exist_ok=True)
    code = data.code if condition == "original" else data.code.map(remove_comments)
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), max_features=50000,
        sublinear_tf=True, lowercase=False, dtype=np.float32,
    )
    started = perf_counter()
    x = {"train": vectorizer.fit_transform(code[data.split == "train"])}
    feature_seconds = perf_counter() - started
    for part in ("validation", "test"):
        x[part] = vectorizer.transform(code[data.split == part])
    joblib.dump(vectorizer, model_folder / "vectorizer.joblib", compress=3)
    (folder / "features.json").write_text(json.dumps({
        "training_feature_seconds": feature_seconds,
        "features": len(vectorizer.vocabulary_),
        "vectorizer_MB": (model_folder / "vectorizer.joblib").stat().st_size / 1e6,
    }, indent=2))

    # Train the four models on exactly the same features.
    models = {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", random_state=SEED),
        "Linear SVM": LinearSVC(C=1.0, dual="auto", max_iter=10000, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=40, min_samples_leaf=2,
            max_features="sqrt", n_jobs=4, random_state=SEED,
        ),
    }
    for name, model in models.items():
        print(f"{condition}: training {name}", flush=True)
        started = perf_counter()
        model.fit(x["train"], data.loc[data.split == "train", "label"])
        training_seconds = perf_counter() - started
        model_path = model_folder / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)

        # Measure each model and save its predictions.
        for part in ("validation", "test"):
            rows = data[data.split == part]
            started = perf_counter()
            predicted = model.predict(x[part])
            inference_seconds = perf_counter() - started
            precision, recall, f1, _ = precision_recall_fscore_support(
                rows.label, predicted, average="macro", zero_division=0,
            )
            results.append({
                "condition": condition, "model": name, "split": part,
                "accuracy": accuracy_score(rows.label, predicted),
                "macro_f1": f1, "macro_precision": precision, "macro_recall": recall,
                "training_seconds": training_seconds, "inference_seconds": inference_seconds,
                "model_MB": model_path.stat().st_size / 1e6,
            })
            pd.DataFrame({"row_id": rows.row_id, "true": rows.label, "predicted": predicted}).to_csv(
                folder / f"{name}_{part}_predictions.csv", index=False,
            )
            pd.DataFrame(classification_report(rows.label, predicted, output_dict=True, zero_division=0)).T.to_csv(
                folder / f"{name}_{part}_classes.csv",
            )
            pd.DataFrame(confusion_matrix(rows.label, predicted, labels=labels), index=labels, columns=labels).to_csv(
                folder / f"{name}_{part}_confusion.csv",
            )
    return results


def main():
    # DATA: load the five classes and check the saved split.
    archive = ROOT / "data/LLM-AuthorBench.json.zip"
    if not archive.exists():
        archive.parent.mkdir(exist_ok=True)
        url = "https://raw.githubusercontent.com/LLMauthorbench/LLMauthorbench/6a1c2ac173c774cfbd012f4c81e8d0a51bb61eb7/LLM-AuthorBench.json.zip"
        urllib.request.urlretrieve(url, archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == "e24399b7b05b812c5a1148ef44245348859eaa74b1140523cb695f984337f24c"
    with zipfile.ZipFile(archive) as file:
        records = json.loads(file.read("LLM-AuthorBench.json"))
    data = pd.DataFrame({
        "row_id": range(len(records)),
        "code": [row["c_code"] for row in records],
        "label": [row["model_name"] for row in records],
        "prompt": [row["prompt"] for row in records],
    })
    data["source_hash"] = data.code.map(lambda code: hashlib.sha256(code.encode()).hexdigest())
    data = data[data.label.isin(["gpt-4.1", "deepseek-chat", "claude-3.5-haiku",
                                    "gemini-2.5-flash-preview-05-20", "llama-3.3-70b-instruct"])]
    assert data.code.str.strip().ne("").all()
    assert not data.source_hash.duplicated().any()
    split = pd.read_csv(ROOT / "data/splits.csv")
    data = data.merge(split, on=["row_id", "label", "source_hash"], validate="one_to_one")
    assert len(data) == len(split) == 20000
    labels = sorted(data.label.unique())
    assert len(labels) == 5
    assert set(data.split) == {"train", "validation", "test"}
    for part in ("train", "validation", "test"):
        assert set(data.loc[data.split == part, "label"]) == set(labels)
        for column in ("problem_id", "prompt", "source_hash"):
            assert not set(data.loc[data.split == part, column]) & set(data.loc[data.split != part, column])
    print(data.groupby("split").size(), flush=True)
    pd.crosstab(data.split, data.label).to_csv(OUTPUT / "class_counts.csv")
    data.groupby("split").agg(samples=("row_id", "size"), groups=("problem_id", "nunique")).to_csv(OUTPUT / "split_counts.csv")

    # EXPERIMENT 1: Original source code, including comments.
    print("\nEXPERIMENT 1: ORIGINAL CODE", flush=True)
    results = run_experiment(data, "original")

    # EXPERIMENT 2: Remove comments and repeat with the same settings.
    print("\nEXPERIMENT 2: CODE WITHOUT COMMENTS", flush=True)
    results += run_experiment(data, "comments_removed")

    # FINAL COMPARISON: select with validation scores and report changes.
    results = pd.DataFrame(results)
    results.to_csv(OUTPUT / "metrics.csv", index=False)
    validation = results[(results.condition == "original") & (results.split == "validation")]
    winner = validation.sort_values(["macro_f1", "model"], ascending=[False, True]).iloc[0].model
    (OUTPUT / "winner.txt").write_text(f"Highest original validation Macro-F1: {winner}\n")
    comparison = results.pivot(index=["model", "split"], columns="condition", values="macro_f1")
    comparison["change_percentage_points"] = 100 * (comparison.comments_removed - comparison.original)
    comparison["original_rank"] = comparison.groupby(level="split").original.rank(ascending=False)
    comparison["comments_removed_rank"] = comparison.groupby(level="split").comments_removed.rank(ascending=False)
    comparison.to_csv(OUTPUT / "comment_comparison.csv")
    print("Validation-selected winner:", winner)
    print(results[results.split == "test"][["condition", "model", "accuracy", "macro_f1"]].to_string(index=False))
    print("Results saved in", OUTPUT)


if __name__ == "__main__":
    main()
