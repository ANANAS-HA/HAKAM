"""F2 — Headline test-set accuracy on the 1000-question stratified test sample.

Four vertical bars: vanilla 0.8B, 1-epoch trained, 3-epoch trained, 27B-AWQ-Int4
teacher. Each bar carries a ±3 pp 95% binomial CI error bar. Δ-arrow from
vanilla to 3-epoch (+17.2 pp) and a bracket between 3-epoch and teacher
labelled "Δ = 0.2 pp (within ±3.0 pp 95% CI)".

Numbers re-derived from the four prediction jsonls; halts if any drifts >0.5 pp
from the report's reference.

Source: predictions_test_{vanilla,trained,trained_3ep,teacher}.jsonl.
Target report section: §4.5, replaces Table 4.4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _metric import accuracy, load_jsonl
from _style import COLORS, FIGSIZE_SINGLE, apply_style


ROOT = Path(__file__).parent.parent
PRED_FILES = {
    "Vanilla 0.8B":              ROOT / "predictions_test_vanilla.jsonl",
    "1-epoch trained":           ROOT / "predictions_test_trained.jsonl",
    "3-epoch trained":           ROOT / "predictions_test_trained_3ep.jsonl",
    "Teacher 27B-AWQ-Int4":      ROOT / "predictions_test_teacher.jsonl",
}
EXPECTED = {
    "Vanilla 0.8B":           0.227,
    "1-epoch trained":        0.365,
    "3-epoch trained":        0.399,
    "Teacher 27B-AWQ-Int4":   0.397,
}
COLORS_BY_LABEL = {
    "Vanilla 0.8B":         COLORS["vanilla"],
    "1-epoch trained":      COLORS["epoch1"],
    "3-epoch trained":      COLORS["epoch3"],
    "Teacher 27B-AWQ-Int4": COLORS["teacher"],
}
OUT = Path(__file__).parent / "F2_headline_accuracy.png"


def main() -> None:
    apply_style()

    accuracies: dict[str, float] = {}
    for label, path in PRED_FILES.items():
        rows = load_jsonl(path)
        n_correct, n_total = accuracy(rows)
        acc = n_correct / n_total
        accuracies[label] = acc
        if abs(acc - EXPECTED[label]) > 0.005:
            print(f"FAIL: {label}  expected={EXPECTED[label]:.3f}  got={acc:.3f}")
            sys.exit(1)

    n = 1000
    se = np.sqrt(np.array(list(accuracies.values())) * (1 - np.array(list(accuracies.values()))) / n)
    ci95 = 1.96 * se
    labels = list(accuracies.keys())
    values = [accuracies[l] for l in labels]
    cols = [COLORS_BY_LABEL[l] for l in labels]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, yerr=ci95, capsize=5, color=cols,
                  edgecolor="black", linewidth=0.5, alpha=0.95)

    for x, v, ci in zip(x_pos, values, ci95):
        ax.text(x, v + ci + 0.012, f"{v*100:.1f}%", ha="center", fontsize=9.5,
                fontweight="bold")

    delta_van_3ep = (accuracies["3-epoch trained"] - accuracies["Vanilla 0.8B"]) * 100
    ax.annotate(
        f"+{delta_van_3ep:.1f} pp",
        xy=(2, accuracies["3-epoch trained"] + 0.06),
        xytext=(0.5, 0.42),
        fontsize=10, color=COLORS["delta_pos"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["delta_pos"], lw=1.4),
    )

    y_bracket = max(values) + 0.10
    delta_3ep_tch = (accuracies["3-epoch trained"] - accuracies["Teacher 27B-AWQ-Int4"]) * 100
    ax.plot([2, 3], [y_bracket, y_bracket], color="black", lw=1)
    ax.plot([2, 2], [y_bracket - 0.01, y_bracket], color="black", lw=1)
    ax.plot([3, 3], [y_bracket - 0.01, y_bracket], color="black", lw=1)
    ax.text(2.5, y_bracket + 0.012,
            f"Δ = {delta_3ep_tch:+.1f} pp (within ±3.0 pp 95% CI)",
            ha="center", fontsize=9, fontstyle="italic")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (gold-fuzzy ≥ 0.80)")
    ax.set_ylim(0, 0.6)
    ax.set_yticks(np.arange(0, 0.61, 0.1))
    ax.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0, 0.61, 0.1)])
    ax.set_title("Test-set accuracy on 1 000 stratified test questions (n = 1 000)")

    plt.tight_layout()
    plt.savefig(OUT)
    plt.close()

    print(f"F2: vanilla={accuracies['Vanilla 0.8B']*100:.1f}%, "
          f"1ep={accuracies['1-epoch trained']*100:.1f}%, "
          f"3ep={accuracies['3-epoch trained']*100:.1f}%, "
          f"teacher={accuracies['Teacher 27B-AWQ-Int4']*100:.1f}%  "
          f"[matches report ±0.5pp]")


if __name__ == "__main__":
    main()
