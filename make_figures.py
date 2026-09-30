"""Make charts from saved results. This file does not train any models."""

from pathlib import Path
import os

# Keep the plotting cache inside this project.
ROOT = Path(__file__).resolve().parent
(ROOT / "tmp").mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(ROOT / "tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")  # Save images without opening plot windows.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FIGURE_FOLDER = ROOT / "results/figures"
FIGURE_FOLDER.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 17})

# FIGURE 1: SVM confusion matrix from Experiment 1
# Rows = actual authors. Columns = predicted authors.
confusion = pd.read_csv(
    ROOT / "results/experiment_1_original/Linear SVM_test_confusion.csv",
    index_col=0,
)
short_names = {
    "claude-3.5-haiku": "Claude",
    "deepseek-chat": "DeepSeek",
    "gemini-2.5-flash-preview-05-20": "Gemini",
    "gpt-4.1": "GPT-4.1",
    "llama-3.3-70b-instruct": "Llama",
}
predicted_names = [short_names[name] for name in confusion.columns]
actual_names = [short_names[name] for name in confusion.index]
largest_count = confusion.values.max()

figure, axes = plt.subplots(figsize=(6.4, 6.4), layout="constrained")
figure.patch.set_facecolor("#F7FAFA")
axes.imshow(confusion.values, cmap="Blues", vmin=0, vmax=largest_count)
axes.set_xticks(range(5), predicted_names, rotation=35, ha="right")
axes.set_yticks(range(5), actual_names)
axes.set_xlabel("Predicted LLM")
axes.set_ylabel("Actual LLM")

# Write the number of programs in every cell.
for row in range(5):
    for column in range(5):
        count = confusion.iloc[row, column]
        if count > largest_count / 2:
            text_color = "white"
        else:
            text_color = "#152E38"
        axes.text(
            column, row, str(count),
            ha="center", va="center", fontsize=21, color=text_color,
        )

axes.set_title("SVM test confusion matrix", fontsize=22, pad=16)
figure.savefig(FIGURE_FOLDER / "svm_confusion_matrix.png", dpi=200)
plt.close(figure)

assert confusion.values.sum() == 3872
print("Matrix total:", confusion.values.sum(), "Correct:", np.trace(confusion.values))

# FIGURE 2: Compare 50,000 and 500 features using validation Macro-F1.
comparison = pd.read_csv(ROOT / "results/feature_comparison.csv")
comparison = comparison[comparison["split"] == "validation"]
comparison = comparison.sort_values("original_rank")
positions = np.arange(4)

figure, axes = plt.subplots(figsize=(12, 3.4), layout="constrained")
figure.patch.set_facecolor("#F7FAFA")
axes.set_facecolor("#F7FAFA")
axes.barh(
    positions - 0.18, comparison["original"],
    height=0.32, label="50,000 features", color="#117C83",
)
axes.barh(
    positions + 0.18, comparison["limited_features"],
    height=0.32, label="500 features", color="#9CABB7",
)
axes.set_yticks(positions, comparison["model"])
axes.invert_yaxis()
axes.set_xlim(0, 1)
axes.set_xlabel("Validation Macro-F1")
axes.legend(loc="lower right", frameon=False, fontsize=15)

# Write each score beside its bar.
for position, result in enumerate(comparison.itertuples()):
    axes.text(
        result.original + 0.012, position - 0.18,
        f"{result.original:.4f}", va="center", fontsize=14,
    )
    axes.text(
        result.limited_features + 0.012, position + 0.18,
        f"{result.limited_features:.4f}", va="center", fontsize=14,
    )
axes.spines[["top", "right"]].set_visible(False)
figure.savefig(FIGURE_FOLDER / "feature_ablation.png", dpi=200)
plt.close(figure)

# EQUATIONS: These strings control mathematical appearance only.
# They do not calculate predictions or train the models.
def save_equation(filename, equation, width, height):
    """Render one formula as a transparent image for the slides."""
    figure = plt.figure(figsize=(width, height))
    figure.text(
        0.5, 0.5, equation,
        ha="center", va="center", fontsize=27,
        color="#117C83", math_fontfamily="stix",
    )
    figure.savefig(
        FIGURE_FOLDER / filename,
        dpi=240, transparent=True, bbox_inches="tight", pad_inches=0.08,
    )
    plt.close(figure)

# SVM prediction: choose the author with the largest score.
score_equation = (
    r"$f_k(\mathbf{x})=\mathbf{w}_k^{\mathsf{T}}\mathbf{x}+b_k,\qquad "
    r"\hat{y}=\underset{k}{\mathrm{arg\,max}}\; f_k(\mathbf{x})$"
)
save_equation("svm_score.png", score_equation, 11, 1.1)

# SVM training: regularization plus squared-hinge loss.
training_equation = (
    r"$\min_{\mathbf{w},b}\;\dfrac{1}{2}\left(\|\mathbf{w}\|_2^2+b^2\right)"
    r"+C\sum_{i=1}^{n}\left[\max\left(0,1-y_i(\mathbf{w}^{\mathsf{T}}"
    r"\mathbf{x}_i+b)\right)\right]^2$"
)
save_equation("svm_objective.png", training_equation, 12, 1.5)

# Evaluation: F1 for one class, then the average across five classes.
f1_equation = (
    r"$F_1=\dfrac{2PR}{P+R},\qquad "
    r"\mathrm{Macro}\!\!-\!F_1=\dfrac{1}{5}\sum_{k=1}^{5}F_{1,k}$"
)
save_equation("f1_equation.png", f1_equation, 8, 1.4)
print("Figures saved in", FIGURE_FOLDER)
